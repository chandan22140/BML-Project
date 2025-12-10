"""
Fine-tune LoRA adapters on CIFAR-100 using ViT-B/16.
Trains to obtain MAP estimate of LoRA parameters.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import argparse
from tqdm import tqdm
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import create_vit_lora
from src.utils import set_seed, save_checkpoint, count_parameters


def get_cifar100_dataloaders(batch_size=128, num_workers=4):
    """Load CIFAR-100 dataset and create dataloaders."""
    
    # Load dataset from HuggingFace
    dataset = load_dataset("cifar100")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def train_transforms_fn(examples):
        # print("examples",)

        
        examples['pixel_values'] = [train_transform(img.convert("RGB")) for img in examples['img']]
        # print(len(examples['pixel_values']))
        # print(examples['pixel_values'][0].shape)

        examples['labels'] = examples['fine_label']
        return examples
    
    def test_transforms_fn(examples):
        examples['pixel_values'] = [test_transform(img.convert("RGB")) for img in examples['img']]
        examples['labels'] = examples['fine_label']
        return examples
    
    # Apply transforms
    train_dataset = dataset['train'].with_transform(train_transforms_fn)
    test_dataset = dataset['test'].with_transform(test_transforms_fn)
    
    def collate_fn(batch):
        """Custom collate to handle transformed data."""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])
        return {'pixel_values': pixel_values, 'labels': labels}
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device, prior_precision=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        inputs = batch['pixel_values'].to(device)
        targets = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Cross-entropy loss
        ce_loss = criterion(logits, targets)
        
        # L2 regularization on LoRA parameters (prior)
        l2_reg = 0.0
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                l2_reg += (param ** 2).sum()
        
        loss = ce_loss + (prior_precision / 2) * l2_reg
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = batch['pixel_values'].to(device)
            targets = batch['labels'].to(device)
            
            outputs = model(inputs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = criterion(logits, targets)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(test_loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train LoRA adapters on CIFAR-100')
    parser.add_argument('--output_dir', type=str, default='checkpoints/vit_lora_cifar100',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (optimizer)')
    parser.add_argument('--prior_precision', type=float, default=1.0,
                        help='Prior precision (lambda) for L2 regularization')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha (scaling)')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    train_loader, test_loader = get_cifar100_dataloaders(
        batch_size=args.batch_size,
        num_workers=4
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating ViT-B/16 with LoRA adapters...")
    model = create_vit_lora(
        num_classes=100,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    model = model.to(device)
    
    print(f"\nTotal parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, only_trainable=True):,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Training loop
    print("\nStarting training...")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            prior_precision=args.prior_precision
        )
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        save_checkpoint(
            model, optimizer, epoch + 1, test_loss, checkpoint_path,
            metadata={
                'train_acc': train_acc,
                'test_acc': test_acc,
                'lora_r': args.lora_r,
                'lora_alpha': args.lora_alpha,
                'prior_precision': args.prior_precision
            }
        )
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_path = os.path.join(args.output_dir, 'model_map.pt')
            save_checkpoint(
                model, None, epoch + 1, test_loss, best_path,
                metadata={
                    'best_test_acc': best_acc,
                    'lora_r': args.lora_r,
                    'lora_alpha': args.lora_alpha,
                    'prior_precision': args.prior_precision
                }
            )
            print(f"✓ Best model saved (Test Acc: {best_acc:.2f}%)")
    
    print("\n" + "=" * 50)
    print(f"Training completed! Best test accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

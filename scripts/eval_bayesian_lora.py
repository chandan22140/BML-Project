"""
Evaluate layer-wise Bayesian posteriors and compute calibration metrics.
Generates predictive samples for each layer subset and computes ECE.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
print("🎯 Configured to use GPU 6.")

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import create_vit_lora, get_lora_layer_names
from src.laplace import LaplaceLoRA
from src.metrics import compute_top_label_ece, plot_reliability_diagram, compute_accuracy
from src.utils import load_checkpoint, save_results, set_seed


def get_cifar100_test_loader(batch_size=128, num_workers=4):
    """Load CIFAR-100 test set."""
    dataset = load_dataset("cifar100")
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def test_transforms_fn(examples):
        examples['pixel_values'] = [test_transform(img.convert("RGB")) for img in examples['img']]
        examples['labels'] = examples['fine_label']
        return examples
    
    test_dataset = dataset['test'].with_transform(test_transforms_fn)
    
    def collate_fn(batch):
        """Custom collate to handle transformed data."""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])
        return {'pixel_values': pixel_values, 'labels': labels}
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return test_loader


def evaluate_deterministic(model, test_loader, device):
    """Evaluate deterministic (MAP) model."""
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating deterministic model"):
            inputs = batch['pixel_values'].to(device)
            targets = batch['labels'].to(device)
            
            outputs = model(inputs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            all_logits.append(logits.cpu())
            all_labels.append(targets.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_logits, all_labels


def main():
    parser = argparse.ArgumentParser(description='Evaluate Bayesian LoRA with layer-wise analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to MAP checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of posterior samples for prediction')
    parser.add_argument('--prior_precision', type=float, default=1.0,
                        help='Prior precision (lambda)')
    parser.add_argument('--laplace_type', type=str, default='diagonal',
                        choices=['diagonal', 'kfac'],
                        help='Type of Laplace approximation')
    parser.add_argument('--n_bins', type=int, default=15,
                        help='Number of bins for ECE computation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank (must match training)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha (must match training)')
    parser.add_argument('--save_laplace', type=str, default=None,
                        help='Path to save fitted Laplace approximation')
    parser.add_argument('--load_laplace', type=str, default=None,
                        help='Path to load pre-fitted Laplace approximation')
    parser.add_argument('--use_glm', action='store_true',
                        help='Use GLM predictive instead of MC sampling')
    parser.add_argument('--optimize_prior', action='store_true',
                        help='Optimize prior precision using marginal likelihood')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    
    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load test data
    print("\nLoading CIFAR-100 test set...")
    test_loader = get_cifar100_test_loader(batch_size=args.batch_size)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating ViT-B/16 with LoRA...")
    model = create_vit_lora(
        num_classes=100,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Load MAP checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint, device=device)
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    
    # Get layer names
    layer_names = get_lora_layer_names(model)
    print(f"\nFound {len(layer_names)} layers with LoRA: {layer_names}")
    
    # Evaluate deterministic model
    print("\n" + "="*60)
    print("Evaluating deterministic (MAP) model...")
    print("="*60)
    det_logits, labels = evaluate_deterministic(model, test_loader, device)
    
    det_ece = compute_top_label_ece(det_logits, labels, n_bins=args.n_bins)
    det_acc = compute_accuracy(det_logits.argmax(dim=1).numpy(), labels.numpy())
    
    print(f"Deterministic ECE: {det_ece:.4f}")
    print(f"Deterministic Accuracy: {det_acc:.2f}%")
    
    # Plot deterministic reliability diagram
    probs = torch.softmax(det_logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    plot_reliability_diagram(
        confidences.numpy(),
        predictions.numpy(),
        labels.numpy(),
        n_bins=args.n_bins,
        title="Deterministic Model (MAP)",
        save_path=os.path.join(args.output_dir, 'plots', 'reliability_det.png')
    )
    
    # Fit Laplace approximation
    print("\n" + "="*60)
    print(f"Fitting {args.laplace_type} Laplace approximation...")
    print("="*60)
    
    laplace = LaplaceLoRA(
        model=model,
        likelihood='classification',
        prior_precision=args.prior_precision,
        backend=args.laplace_type
    )
    
    if args.load_laplace and os.path.exists(args.load_laplace):
        # Load pre-fitted Laplace approximation
        print(f"Loading Laplace approximation from: {args.load_laplace}")
        laplace.load(args.load_laplace)
    else:
        # For Laplace fitting, we need training data (use test as proxy for demo)
        # In practice, use actual training set
        train_loader = get_cifar100_test_loader(batch_size=args.batch_size)
        laplace.fit(train_loader, device=device)
        
        # Optionally optimize prior precision
        if args.optimize_prior:
            print("\nOptimizing prior precision...")
            laplace.optimize_prior_precision(train_loader, n_steps=50, verbose=True)
        
        # Optionally save the fitted Laplace approximation
        if args.save_laplace:
            laplace.save(args.save_laplace)
    
    # Print posterior statistics
    try:
        stats = laplace.get_posterior_stats()
        print(f"\nPosterior statistics:")
        print(f"  Backend: {stats['backend']}")
        print(f"  Prior precision: {stats['prior_precision']:.4f}")
        print(f"  Total LoRA parameters: {stats['n_parameters']:,}")
        print(f"  Number of layers: {stats['n_layers']}")
        if 'n_kfac_factors' in stats:
            print(f"  KFAC factors computed: {stats['n_kfac_factors']}")
    except Exception as e:
        print(f"Could not get posterior stats: {e}")
    
    # Layer-wise evaluation
    print("\n" + "="*60)
    print("Evaluating layer-wise Bayesian posteriors...")
    print("="*60)
    
    results = {
        'deterministic': {
            'ece': float(det_ece),
            'accuracy': float(det_acc)
        },
        'layers': {},
        'config': vars(args)
    }
    
    # Evaluate each layer individually
    for layer_idx, layer_name in enumerate(layer_names):
        print(f"\n[{layer_idx+1}/{len(layer_names)}] Evaluating layer: {layer_name}")
        
        # Generate predictive samples with only this layer Bayesian
        bayes_logits, _ = laplace.predictive_samples(
            test_loader,
            n_samples=args.num_samples,
            layer_subset=[layer_name],
            device=device
        )
        
        # Compute metrics
        bayes_ece = compute_top_label_ece(bayes_logits, labels, n_bins=args.n_bins)
        bayes_acc = compute_accuracy(bayes_logits.argmax(dim=1).numpy(), labels.numpy())
        delta_ece = det_ece - bayes_ece
        
        print(f"  Bayesian ECE: {bayes_ece:.4f}")
        print(f"  Bayesian Accuracy: {bayes_acc:.2f}%")
        print(f"  Δ_ECE: {delta_ece:.4f} {'↑' if delta_ece > 0 else '↓'}")
        
        # Store results
        results['layers'][layer_name] = {
            'ece': float(bayes_ece),
            'accuracy': float(bayes_acc),
            'delta_ece': float(delta_ece)
        }
        
        # Plot reliability diagram for significant improvements
        if abs(delta_ece) > 0.005:  # Threshold for plotting
            probs = torch.softmax(bayes_logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            plot_reliability_diagram(
                confidences.numpy(),
                predictions.numpy(),
                labels.numpy(),
                n_bins=args.n_bins,
                title=f"Bayesian Model ({layer_name})",
                save_path=os.path.join(args.output_dir, 'plots', f'reliability_{layer_name}.png')
            )
    
    # Evaluate full Bayesian (all layers)
    print("\n" + "="*60)
    print("Evaluating full Bayesian posterior (all layers)...")
    print("="*60)
    
    full_bayes_logits, _ = laplace.predictive_samples(
        test_loader,
        n_samples=args.num_samples,
        layer_subset=None,  # All layers
        device=device
    )
    
    full_ece = compute_top_label_ece(full_bayes_logits, labels, n_bins=args.n_bins)
    full_acc = compute_accuracy(full_bayes_logits.argmax(dim=1).numpy(), labels.numpy())
    delta_ece_full = det_ece - full_ece
    
    print(f"Full Bayesian ECE: {full_ece:.4f}")
    print(f"Full Bayesian Accuracy: {full_acc:.2f}%")
    print(f"Δ_ECE: {delta_ece_full:.4f}")
    
    results['full_bayesian'] = {
        'ece': float(full_ece),
        'accuracy': float(full_acc),
        'delta_ece': float(delta_ece_full)
    }
    
    # Plot full Bayesian reliability diagram
    probs = torch.softmax(full_bayes_logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    plot_reliability_diagram(
        confidences.numpy(),
        predictions.numpy(),
        labels.numpy(),
        n_bins=args.n_bins,
        title="Full Bayesian Model (All Layers)",
        save_path=os.path.join(args.output_dir, 'plots', 'reliability_full.png')
    )
    
    # Save results
    save_results(results, args.output_dir, 'layer_wise_results.json')
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Layer-wise Δ_ECE (sorted)")
    print("="*60)
    
    layer_deltas = [(name, data['delta_ece']) for name, data in results['layers'].items()]
    layer_deltas.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (layer, delta) in enumerate(layer_deltas, 1):
        sign = '↑' if delta > 0 else '↓'
        print(f"{rank:2d}. {layer:15s}: {delta:+.4f} {sign}")
    
    print(f"\nFull Bayesian: {delta_ece_full:+.4f}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

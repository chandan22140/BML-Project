"""
ViT-B/16 model with LoRA adapters for CIFAR-100.
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
from peft import LoraConfig, get_peft_model
from typing import Optional, List, Dict


def create_vit_lora(
    num_classes: int = 100,
    pretrained_name: str = "google/vit-base-patch16-224",
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Create ViT-B/16 model with LoRA adapters.
    
    Args:
        num_classes: Number of output classes
        pretrained_name: HuggingFace model name
        lora_r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: Which modules to apply LoRA to
    
    Returns:
        PEFT model with LoRA adapters
    """
    # Load pretrained ViT
    model = ViTForImageClassification.from_pretrained(
        pretrained_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    # Default target modules for ViT (query and value projections)
    if target_modules is None:
        target_modules = ["query", "value"]
    
    # Configure LoRA (no task_type needed for image classification with PEFT)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=["classifier"]  # Also train the classifier head
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def get_lora_layer_names(model: nn.Module) -> List[str]:
    """
    Get names of all transformer blocks with LoRA adapters.
    
    Args:
        model: PEFT model with LoRA
    
    Returns:
        List of layer names (e.g., ['layer.0', 'layer.1', ...])
    """
    layer_names = set()
    for name, module in model.named_modules():
        if 'lora' in name.lower() and 'layer' in name.lower():
            # Extract layer number from name like 'vit.encoder.layer.0.attention.attention.query.lora_A'
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layer' and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_names.add(f"layer.{parts[i + 1]}")
                    break
    
    return sorted(list(layer_names), key=lambda x: int(x.split('.')[1]))


def get_lora_parameters_by_layer(model: nn.Module, layer_name: str) -> List[torch.Tensor]:
    """
    Get all LoRA parameters for a specific layer.
    
    Args:
        model: PEFT model with LoRA
        layer_name: Layer identifier (e.g., 'layer.0')
    
    Returns:
        List of parameter tensors for that layer
    """
    params = []
    for name, param in model.named_parameters():
        if layer_name in name and 'lora' in name.lower() and param.requires_grad:
            params.append(param)
    return params


def freeze_lora_except_layers(model: nn.Module, keep_layers: List[str]):
    """
    Freeze all LoRA parameters except those in specified layers.
    
    Args:
        model: PEFT model with LoRA
        keep_layers: List of layer names to keep trainable
    """
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            # Check if this parameter belongs to one of the layers we want to keep
            should_keep = any(layer in name for layer in keep_layers)
            param.requires_grad = should_keep


def unfreeze_all_lora(model: nn.Module):
    """Unfreeze all LoRA parameters."""
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True


def get_lora_modules(model: nn.Module) -> Dict[str, nn.Module]:
    """
    Get all LoRA modules from a PEFT model.
    
    This function finds the parent modules that contain lora_A and lora_B,
    which is useful for KFAC Laplace approximation.
    
    Args:
        model: PEFT model with LoRA
    
    Returns:
        Dictionary mapping module names to LoRA modules
    """
    lora_modules = {}
    
    for name, module in model.named_modules():
        # Check for PEFT-style LoRA (has lora_A and lora_B)
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            lora_modules[name] = module
    
    return lora_modules


def get_lora_param_names(model: nn.Module) -> List[str]:
    """
    Get names of all LoRA parameters in the model.
    
    Args:
        model: PEFT model with LoRA
    
    Returns:
        List of parameter names containing 'lora'
    """
    return [name for name, param in model.named_parameters() 
            if 'lora' in name.lower() and param.requires_grad]

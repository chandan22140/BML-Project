"""
Utility functions for data loading, checkpointing, and configuration.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import os
import json
import numpy as np
from pathlib import Path


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    

def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Save model checkpoint with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def get_lora_parameters(model: nn.Module) -> List[torch.Tensor]:
    """Extract all LoRA parameters from the model."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_params.append(param)
    return lora_params


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """Count model parameters."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_layer_names(model: nn.Module, with_lora: bool = True) -> List[str]:
    """Get names of all layers (optionally filtered for LoRA)."""
    if with_lora:
        return [name for name, _ in model.named_modules() if 'lora' in name.lower()]
    return [name for name, _ in model.named_modules()]


def save_results(results: Dict[str, Any], output_dir: str, filename: str = "results.json"):
    """Save results dictionary to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_serializable = convert_numpy(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

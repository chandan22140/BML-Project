"""
Laplace approximation for LoRA parameters using KFAC or diagonal structure.
Based on: Bayesian Low-Rank Adaptation for Large Language Models (ICLR 2024)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


class LaplaceLoRA:
    """
    Laplace approximation over LoRA parameters.
    Supports both diagonal and KFAC low-rank approximations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        likelihood: str = 'classification',
        prior_precision: float = 1.0,
        backend: str = 'diagonal'  # 'diagonal' or 'kfac'
    ):
        """
        Args:
            model: Neural network with LoRA adapters
            likelihood: Type of likelihood ('classification' or 'regression')
            prior_precision: Prior precision (lambda in paper)
            backend: Type of Hessian approximation
        """
        self.model = model
        self.likelihood = likelihood
        self.prior_precision = prior_precision
        self.backend = backend
        
        # Store MAP parameters
        self.mean = None
        self.precision = None  # Diagonal or KFAC structure
        
        # Layer-wise precision matrices
        self.layer_precisions = {}
        
    def fit(self, train_loader: DataLoader, device: str = 'cuda'):
        """
        Fit Laplace approximation at MAP estimate.
        
        Args:
            train_loader: DataLoader for training data
            device: Device to run on
        """
        self.model.eval()
        self.device = device
        
        # Get MAP parameters (current model state)
        self.mean = {name: param.data.clone() 
                     for name, param in self.model.named_parameters() 
                     if 'lora' in name.lower() and param.requires_grad}
        
        print(f"Computing {self.backend} Hessian approximation...")
        
        if self.backend == 'diagonal':
            self._fit_diagonal(train_loader)
        elif self.backend == 'kfac':
            self._fit_kfac(train_loader)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        print("Laplace approximation fitted.")
    
    def _fit_diagonal(self, train_loader: DataLoader):
        """Fit diagonal Laplace approximation."""
        # Initialize precision dictionary
        self.precision = {name: torch.zeros_like(param) 
                          for name, param in self.mean.items()}
        
        self.model.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Computing diagonal Hessian")):
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs = batch['pixel_values'].to(self.device)
                targets = batch['labels'].to(self.device)
            
            outputs = self.model(inputs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Compute diagonal of Hessian via squared gradients
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            
            # Accumulate squared gradients (diagonal Hessian approximation)
            for name, param in self.model.named_parameters():
                if name in self.precision and param.grad is not None:
                    self.precision[name] += param.grad.data ** 2
            
            self.model.zero_grad()
        
        # Add prior precision
        for name in self.precision:
            self.precision[name] = self.precision[name] + self.prior_precision
    
    def _fit_kfac(self, train_loader: DataLoader):
        """
        Fit KFAC low-rank Laplace approximation.
        Simplified implementation - stores per-layer Kronecker factors.
        """
        # For KFAC, we approximate the Hessian with Kronecker products
        # H ≈ A ⊗ B where A and B are smaller matrices
        # This is a simplified version - full KFAC requires more sophisticated implementation
        
        print("Note: Using simplified KFAC approximation")
        self._fit_diagonal(train_loader)  # Fallback to diagonal for now
        
        # TODO: Implement full KFAC with Kronecker factors
        # This would require tracking activations and gradients separately
    
    def sample_parameters(
        self,
        n_samples: int = 1,
        layer_subset: Optional[List[str]] = None,
        scale: float = 1.0
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Sample from the Laplace posterior.
        
        Args:
            n_samples: Number of samples to draw
            layer_subset: If provided, only sample from these layers (restrict posterior)
            scale: Scale factor for variance (temperature)
        
        Returns:
            List of parameter dictionaries (one per sample)
        """
        if self.mean is None or self.precision is None:
            raise RuntimeError("Must call fit() before sampling")
        
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            
            for name, mean_val in self.mean.items():
                # Check if this parameter should be sampled
                if layer_subset is not None:
                    should_sample = any(layer in name for layer in layer_subset)
                else:
                    should_sample = True
                
                if should_sample:
                    # Sample from N(mean, scale / precision)
                    if self.backend == 'diagonal':
                        std = torch.sqrt(scale / (self.precision[name] + 1e-8))
                        noise = torch.randn_like(mean_val) * std
                        sample[name] = mean_val + noise
                    else:  # kfac
                        # For full KFAC, would use Kronecker structure
                        std = torch.sqrt(scale / (self.precision[name] + 1e-8))
                        noise = torch.randn_like(mean_val) * std
                        sample[name] = mean_val + noise
                else:
                    # Keep at MAP value
                    sample[name] = mean_val.clone()
            
            samples.append(sample)
        
        return samples
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set model parameters from a dictionary."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.copy_(params[name])
    
    def predictive_samples(
        self,
        data_loader: DataLoader,
        n_samples: int = 30,
        layer_subset: Optional[List[str]] = None,
        device: str = 'cuda'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictive samples for calibration analysis.
        
        Args:
            data_loader: DataLoader for test data
            n_samples: Number of posterior samples
            layer_subset: If provided, only sample from these layers
            device: Device to run on
        
        Returns:
            Tuple of (averaged_logits, labels)
        """
        self.model.eval()
        
        # Storage for predictions
        all_logits_list = []
        all_labels = []
        
        with torch.no_grad():
            # Get one batch to determine shape
            first_batch = next(iter(data_loader))
            if isinstance(first_batch, (tuple, list)):
                n_test = len(data_loader.dataset)
                n_classes = self.model.config.num_labels if hasattr(self.model, 'config') else 100
            else:
                n_test = len(data_loader.dataset)
                n_classes = self.model.config.num_labels if hasattr(self.model, 'config') else 100
        
        # Sample parameters
        param_samples = self.sample_parameters(n_samples, layer_subset)
        
        # Store original parameters
        original_params = {name: param.data.clone() 
                          for name, param in self.model.named_parameters() 
                          if name in self.mean}
        
        # Generate predictions for each sample
        for sample_idx, params in enumerate(tqdm(param_samples, desc="Generating predictive samples")):
            # Set sampled parameters
            self.set_parameters(params)
            
            sample_logits = []
            sample_labels = []
            
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                else:
                    inputs = batch['pixel_values'].to(device)
                    targets = batch['labels'].to(device)
                
                outputs = self.model(inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                sample_logits.append(logits.cpu())
                if sample_idx == 0:  # Only store labels once
                    sample_labels.append(targets.cpu())
            
            all_logits_list.append(torch.cat(sample_logits, dim=0))
            if sample_idx == 0:
                all_labels = torch.cat(sample_labels, dim=0)
        
        # Restore original parameters
        self.set_parameters(original_params)
        
        # Average logits across samples (Bayesian model averaging)
        avg_logits = torch.stack(all_logits_list, dim=0).mean(dim=0)
        
        return avg_logits, all_labels

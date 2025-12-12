"""
Laplace approximation for LoRA parameters using KFAC or diagonal structure.
Based on: Bayesian Low-Rank Adaptation for Large Language Models (ICLR 2024)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from math import sqrt, pi, log


class KroneckerFactor:
    """
    Stores Kronecker factors (A, B) for a single layer.
    The Hessian is approximated as: H ≈ A ⊗ B
    """
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        self.A = A  # Input activation covariance
        self.B = B  # Output gradient covariance
    
    def to(self, device):
        self.A = self.A.to(device)
        self.B = self.B.to(device)
        return self
    
    def add_(self, other: 'KroneckerFactor'):
        """Accumulate another Kronecker factor."""
        self.A += other.A
        self.B += other.B
        return self
    
    def scale_(self, factor: float):
        """Scale the factors."""
        self.A *= factor
        self.B *= factor
        return self


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
        
        # KFAC-specific storage
        self.kfac_factors = {}  # Dict[layer_name -> KroneckerFactor]
        self.kfac_eigenvectors = {}  # Cached eigendecompositions for sampling
        self.kfac_eigenvalues = {}
        
        # For tracking activations during KFAC computation
        self._activations = {}
        self._hooks = []
        
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
        
        For each LoRA layer, we approximate the Hessian using Kronecker factors:
        H ≈ A ⊗ B where:
        - A: Covariance of input activations (pre-activation)
        - B: Covariance of output gradients (backpropagated gradients)
        
        This allows efficient sampling and posterior computation.
        """
        print("Computing KFAC Hessian approximation...")
        
        # Initialize KFAC factors for each LoRA parameter
        self.kfac_factors = {}
        
        # Find LoRA layers and register hooks
        lora_layers = self._find_lora_layers()
        
        if len(lora_layers) == 0:
            print("Warning: No LoRA layers found, falling back to diagonal")
            self._fit_diagonal(train_loader)
            return
        
        # Register forward hooks to capture activations
        self._register_kfac_hooks(lora_layers)
        
        n_batches = 0
        
        try:
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Computing KFAC factors")):
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    inputs = batch['pixel_values'].to(self.device)
                    targets = batch['labels'].to(self.device)
                
                # Clear activation cache
                self._activations = {}
                
                # Forward pass (hooks will capture activations)
                self.model.zero_grad()
                outputs = self.model(inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Compute loss and backward pass
                loss = nn.functional.cross_entropy(logits, targets)
                loss.backward()
                
                # Update KFAC factors
                self._update_kfac_factors(lora_layers, inputs.size(0))
                
                n_batches += 1
                self.model.zero_grad()
                
        finally:
            # Remove hooks
            self._remove_hooks()
        
        # Normalize and add prior precision
        self._finalize_kfac_factors(n_batches)
        
        # Also maintain diagonal precision for compatibility
        self._compute_diagonal_from_kfac()
        
        print(f"KFAC approximation computed for {len(self.kfac_factors)} LoRA layers")
    
    def _find_lora_layers(self) -> Dict[str, nn.Module]:
        """
        Find all LoRA layers in the model.
        
        Handles both:
        1. PEFT-style LoRA with lora_A/lora_B sub-modules
        2. Direct Linear layers with 'lora' in name
        """
        lora_layers = {}
        
        for name, module in self.model.named_modules():
            # Check for PEFT-style LoRA (has lora_A and lora_B as sub-modules)
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_layers[name] = module
            # Check for individual lora_A or lora_B Linear layers (PEFT structure)
            elif ('lora_A' in name or 'lora_B' in name) and isinstance(module, nn.Linear):
                # Get parent module name (the actual LoRA layer)
                parent_name = '.'.join(name.split('.')[:-2])  # Remove '.lora_A.default' or similar
                if parent_name and parent_name not in lora_layers:
                    # Try to get parent module
                    try:
                        parent = self.model
                        for part in parent_name.split('.'):
                            parent = getattr(parent, part)
                        if hasattr(parent, 'lora_A') and hasattr(parent, 'lora_B'):
                            lora_layers[parent_name] = parent
                    except AttributeError:
                        # If parent access fails, register the Linear layer directly
                        lora_layers[name] = module
            # Fallback: direct Linear layers with 'lora' in name
            elif 'lora' in name.lower() and isinstance(module, nn.Linear):
                lora_layers[name] = module
        
        return lora_layers
    
    def _register_kfac_hooks(self, lora_layers: Dict[str, nn.Module]):
        """Register forward hooks to capture activations for KFAC."""
        self._hooks = []
        
        for name, layer in lora_layers.items():
            # Hook to capture input activations
            def make_hook(layer_name):
                def hook(module, input, output):
                    if len(input) > 0 and input[0] is not None:
                        # Store input activation (detached to avoid memory issues)
                        self._activations[layer_name] = input[0].detach()
                return hook
            
            handle = layer.register_forward_hook(make_hook(name))
            self._hooks.append(handle)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def _update_kfac_factors(self, lora_layers: Dict[str, nn.Module], batch_size: int):
        """Update KFAC factors with current batch's activations and gradients."""
        
        for name, layer in lora_layers.items():
            if name not in self._activations:
                continue
            
            activation = self._activations[name]
            
            # Get the gradient for this layer's parameters
            if hasattr(layer, 'lora_A') and hasattr(layer, 'lora_B'):
                # Handle PEFT-style LoRA
                grad_A = self._get_lora_grad(layer.lora_A)
                grad_B = self._get_lora_grad(layer.lora_B)
                
                if grad_A is not None and grad_B is not None:
                    self._update_single_lora_kfac(name + '.lora_A', activation, grad_A, batch_size)
                    # For lora_B, use the output of lora_A as input
                    self._update_single_lora_kfac(name + '.lora_B', None, grad_B, batch_size, use_identity=True)
            elif hasattr(layer, 'weight') and layer.weight.grad is not None:
                # Handle single linear layer
                self._update_single_layer_kfac(name, activation, layer.weight.grad, batch_size)
    
    def _get_lora_grad(self, lora_module) -> Optional[torch.Tensor]:
        """
        Get gradient from a LoRA module (handles various PEFT structures).
        
        PEFT can structure LoRA as:
        - lora_A.default.weight
        - lora_A.weight
        - lora_A (if it's directly a Linear)
        """
        # Try PEFT structure: lora_A.default.weight
        if hasattr(lora_module, 'default'):
            if hasattr(lora_module.default, 'weight'):
                return lora_module.default.weight.grad
        
        # Try direct weight attribute
        if hasattr(lora_module, 'weight'):
            return lora_module.weight.grad
        
        # If lora_module itself is a parameter
        if isinstance(lora_module, nn.Parameter):
            return lora_module.grad
        
        # Check if it's a ModuleDict (PEFT uses this)
        if isinstance(lora_module, nn.ModuleDict):
            for key in lora_module.keys():
                submodule = lora_module[key]
                if hasattr(submodule, 'weight') and submodule.weight.grad is not None:
                    return submodule.weight.grad
        
        return None
    
    def _update_single_layer_kfac(self, name: str, activation: torch.Tensor, 
                                   grad: torch.Tensor, batch_size: int):
        """Update KFAC factors for a single layer."""
        # Reshape activation: (batch, seq, hidden) -> (batch*seq, hidden)
        if activation.dim() == 3:
            activation = activation.reshape(-1, activation.size(-1))
        elif activation.dim() == 2:
            pass  # Already (batch, hidden)
        else:
            return  # Skip unsupported shapes
        
        # Compute A: input covariance (with bias term)
        # A = E[a a^T] where a is input activation
        a = activation
        if a.size(0) > 0:
            A = (a.T @ a) / a.size(0)
        else:
            return
        
        # Compute B: gradient covariance
        # B = E[g g^T] where g is the gradient w.r.t. pre-activation
        # For linear layer: grad_weight = g^T @ a, so g has shape (batch, out_dim)
        # We use the gradient directly as an approximation
        g = grad  # (out_dim, in_dim)
        B = (g @ g.T)  # Outer product of gradient
        
        # Initialize or accumulate
        if name not in self.kfac_factors:
            self.kfac_factors[name] = KroneckerFactor(
                A.detach().clone(),
                B.detach().clone()
            )
        else:
            self.kfac_factors[name].A += A.detach()
            self.kfac_factors[name].B += B.detach()
    
    def _update_single_lora_kfac(self, name: str, activation: Optional[torch.Tensor],
                                  grad: torch.Tensor, batch_size: int, 
                                  use_identity: bool = False):
        """Update KFAC factors for a single LoRA matrix (A or B)."""
        
        if use_identity or activation is None:
            # For lora_B, use identity-like covariance
            in_dim = grad.size(1) if grad.dim() > 1 else grad.numel()
            A = torch.eye(in_dim, device=grad.device, dtype=grad.dtype)
        else:
            # Reshape activation
            if activation.dim() == 3:
                activation = activation.reshape(-1, activation.size(-1))
            
            if activation.size(0) > 0:
                A = (activation.T @ activation) / activation.size(0)
            else:
                in_dim = activation.size(-1) if activation.dim() > 1 else activation.numel()
                A = torch.eye(in_dim, device=grad.device, dtype=grad.dtype)
        
        # Compute B from gradient
        if grad.dim() == 2:
            B = grad @ grad.T
        else:
            g = grad.reshape(-1)
            B = torch.outer(g, g)
        
        # Initialize or accumulate
        if name not in self.kfac_factors:
            self.kfac_factors[name] = KroneckerFactor(
                A.detach().clone(),
                B.detach().clone()
            )
        else:
            self.kfac_factors[name].A += A.detach()
            self.kfac_factors[name].B += B.detach()
    
    def _finalize_kfac_factors(self, n_batches: int):
        """Normalize KFAC factors and add prior precision (damping)."""
        for name, factor in self.kfac_factors.items():
            # Normalize by number of batches
            factor.A /= n_batches
            factor.B /= n_batches
            
            # Add damping (prior precision) for numerical stability
            # This corresponds to adding λI to the Kronecker factors
            damping = sqrt(self.prior_precision)
            factor.A += damping * torch.eye(factor.A.size(0), device=factor.A.device, dtype=factor.A.dtype)
            factor.B += damping * torch.eye(factor.B.size(0), device=factor.B.device, dtype=factor.B.dtype)
            
            # Compute and cache eigendecomposition for efficient sampling
            try:
                eigenvalues_A, eigenvectors_A = torch.linalg.eigh(factor.A)
                eigenvalues_B, eigenvectors_B = torch.linalg.eigh(factor.B)
                
                # Clamp eigenvalues to be positive for numerical stability
                eigenvalues_A = torch.clamp(eigenvalues_A, min=1e-8)
                eigenvalues_B = torch.clamp(eigenvalues_B, min=1e-8)
                
                self.kfac_eigenvectors[name] = (eigenvectors_A, eigenvectors_B)
                self.kfac_eigenvalues[name] = (eigenvalues_A, eigenvalues_B)
            except Exception as e:
                print(f"Warning: Eigendecomposition failed for {name}: {e}")
                # Fall back to identity
                self.kfac_eigenvectors[name] = None
                self.kfac_eigenvalues[name] = None
    
    def _compute_diagonal_from_kfac(self):
        """Compute diagonal precision from KFAC for compatibility."""
        self.precision = {}
        
        for param_name, mean_val in self.mean.items():
            # Find matching KFAC factor
            matching_kfac = None
            for kfac_name in self.kfac_factors:
                if kfac_name in param_name or param_name in kfac_name:
                    matching_kfac = kfac_name
                    break
            
            if matching_kfac is not None and matching_kfac in self.kfac_eigenvalues:
                factor = self.kfac_factors[matching_kfac]
                # Diagonal of Kronecker product: diag(A ⊗ B) = diag(A) ⊗ diag(B)
                diag_A = torch.diag(factor.A)
                diag_B = torch.diag(factor.B)
                
                # Compute outer product and reshape to match parameter shape
                diag_kron = torch.outer(diag_B, diag_A).reshape(-1)
                
                if diag_kron.numel() >= mean_val.numel():
                    self.precision[param_name] = diag_kron[:mean_val.numel()].reshape(mean_val.shape)
                else:
                    # Pad with prior precision
                    self.precision[param_name] = torch.full_like(mean_val, self.prior_precision)
                    self.precision[param_name].view(-1)[:diag_kron.numel()] = diag_kron
            else:
                # No KFAC factor found, use prior precision
                self.precision[param_name] = torch.full_like(mean_val, self.prior_precision)
    
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
                    if self.backend == 'kfac':
                        # Use KFAC structure for sampling if available
                        sample[name] = self._sample_kfac_parameter(name, mean_val, scale)
                    else:
                        # Diagonal sampling
                        std = torch.sqrt(scale / (self.precision[name] + 1e-8))
                        noise = torch.randn_like(mean_val) * std
                        sample[name] = mean_val + noise
                else:
                    # Keep at MAP value
                    sample[name] = mean_val.clone()
            
            samples.append(sample)
        
        return samples
    
    def _sample_kfac_parameter(self, name: str, mean_val: torch.Tensor, 
                                scale: float) -> torch.Tensor:
        """
        Sample a parameter using KFAC structure.
        
        For KFAC, we have H ≈ A ⊗ B, so
        H^{-1/2} = A^{-1/2} ⊗ B^{-1/2}
        
        To sample: θ = μ + H^{-1/2} @ ε where ε ~ N(0, I)
        """
        # Find matching KFAC factor
        matching_kfac = None
        for kfac_name in self.kfac_factors:
            if kfac_name in name or name in kfac_name:
                matching_kfac = kfac_name
                break
        
        if matching_kfac is None or self.kfac_eigenvectors.get(matching_kfac) is None:
            # Fall back to diagonal sampling
            std = torch.sqrt(scale / (self.precision[name] + 1e-8))
            noise = torch.randn_like(mean_val) * std
            return mean_val + noise
        
        # Use KFAC eigendecomposition for sampling
        eigvecs_A, eigvecs_B = self.kfac_eigenvectors[matching_kfac]
        eigvals_A, eigvals_B = self.kfac_eigenvalues[matching_kfac]
        
        # Compute inverse square root of eigenvalues
        inv_sqrt_eigvals_A = torch.sqrt(scale / eigvals_A)
        inv_sqrt_eigvals_B = torch.sqrt(scale / eigvals_B)
        
        # Sample noise in the eigenspace
        # For matrix-shaped parameters (out_dim, in_dim)
        param_shape = mean_val.shape
        
        if len(param_shape) == 2:
            out_dim, in_dim = param_shape
            
            # Check if dimensions match
            if eigvecs_B.size(0) == out_dim and eigvecs_A.size(0) == in_dim:
                # Sample standard normal
                noise = torch.randn(out_dim, in_dim, device=mean_val.device, dtype=mean_val.dtype)
                
                # Transform: V_B @ diag(λ_B^{-1/2}) @ noise @ diag(λ_A^{-1/2}) @ V_A^T
                noise = eigvecs_B @ (inv_sqrt_eigvals_B.unsqueeze(1) * noise)
                noise = noise @ (inv_sqrt_eigvals_A.unsqueeze(0) * eigvecs_A.T)
                
                return mean_val + noise
        
        # Fall back to diagonal for non-matching dimensions
        std = torch.sqrt(scale / (self.precision[name] + 1e-8))
        noise = torch.randn_like(mean_val) * std
        return mean_val + noise
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set model parameters from a dictionary."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.copy_(params[name])
    
    def log_marginal_likelihood(self, prior_precision: Optional[float] = None) -> torch.Tensor:
        """
        Compute the Laplace approximation to the log marginal likelihood.
        
        log p(D) ≈ log p(D|θ_MAP) + log p(θ_MAP) - 1/2 log det(H + λI) + const
        
        Args:
            prior_precision: Prior precision (overrides self.prior_precision if provided)
        
        Returns:
            Log marginal likelihood estimate
        """
        if self.mean is None or self.precision is None:
            raise RuntimeError("Must call fit() before computing log marginal likelihood")
        
        if prior_precision is None:
            prior_precision = self.prior_precision
        
        # Count total parameters
        n_params = sum(p.numel() for p in self.mean.values())
        
        if self.backend == 'diagonal':
            # For diagonal: log det(H) = sum of log of diagonal elements
            log_det_posterior = sum(
                torch.log(prec + 1e-8).sum() 
                for prec in self.precision.values()
            )
            log_det_prior = n_params * np.log(prior_precision)
            
        elif self.backend == 'kfac':
            # For KFAC: log det(A ⊗ B) = n * log det(A) + m * log det(B)
            log_det_posterior = 0.0
            
            for name, factor in self.kfac_factors.items():
                n_A, n_B = factor.A.size(0), factor.B.size(0)
                
                # Use eigenvalues if available
                if name in self.kfac_eigenvalues and self.kfac_eigenvalues[name] is not None:
                    eigvals_A, eigvals_B = self.kfac_eigenvalues[name]
                    log_det_A = torch.log(eigvals_A + 1e-8).sum()
                    log_det_B = torch.log(eigvals_B + 1e-8).sum()
                    log_det_posterior += n_B * log_det_A + n_A * log_det_B
                else:
                    # Compute directly (less stable)
                    log_det_A = torch.logdet(factor.A + 1e-6 * torch.eye(n_A, device=factor.A.device))
                    log_det_B = torch.logdet(factor.B + 1e-6 * torch.eye(n_B, device=factor.B.device))
                    log_det_posterior += n_B * log_det_A + n_A * log_det_B
            
            log_det_prior = n_params * np.log(prior_precision)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        # Log determinant ratio (part of marginal likelihood)
        log_det_ratio = log_det_posterior - log_det_prior
        
        # Scatter term: (θ_MAP - μ_0)^T P_0 (θ_MAP - μ_0)
        # Assuming zero prior mean
        scatter = prior_precision * sum(
            (mean_val ** 2).sum() 
            for mean_val in self.mean.values()
        )
        
        # Approximate log marginal likelihood
        # Note: This doesn't include the log likelihood term (would need loss)
        log_marglik = -0.5 * (log_det_ratio + scatter)
        
        return log_marglik
    
    def functional_variance(self, jacobians: torch.Tensor) -> torch.Tensor:
        """
        Compute functional variance for GLM predictive.
        
        f_var[i] = J[i] @ P^{-1} @ J[i].T
        
        Args:
            jacobians: Jacobians of shape (batch, outputs, parameters)
        
        Returns:
            Functional variance of shape (batch, outputs, outputs)
        """
        if self.backend == 'diagonal':
            # For diagonal: J @ diag(1/p) @ J.T
            # Flatten precision to vector
            prec_diag = torch.cat([p.view(-1) for p in self.precision.values()])
            var_diag = 1.0 / (prec_diag + 1e-8)
            
            # Compute J @ diag(var) @ J.T
            # jacobians: (batch, outputs, params)
            # var_diag: (params,)
            f_var = torch.einsum('ncp,p,nkp->nck', jacobians, var_diag, jacobians)
            return f_var
            
        elif self.backend == 'kfac':
            # For KFAC, use the inverse Kronecker structure
            # (A ⊗ B)^{-1} = A^{-1} ⊗ B^{-1}
            # This is more complex for jacobians - fall back to diagonal approx
            prec_diag = torch.cat([p.view(-1) for p in self.precision.values()])
            var_diag = 1.0 / (prec_diag + 1e-8)
            f_var = torch.einsum('ncp,p,nkp->nck', jacobians, var_diag, jacobians)
            return f_var
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def optimize_prior_precision(
        self,
        train_loader: DataLoader,
        n_steps: int = 100,
        lr: float = 0.1,
        verbose: bool = False
    ) -> float:
        """
        Optimize prior precision using marginal likelihood.
        
        Args:
            train_loader: Training data loader
            n_steps: Number of optimization steps
            lr: Learning rate for optimization
            verbose: Whether to print progress
        
        Returns:
            Optimized prior precision
        """
        log_prior_prec = torch.tensor(np.log(self.prior_precision), 
                                       requires_grad=True, 
                                       device=self.device)
        optimizer = torch.optim.Adam([log_prior_prec], lr=lr)
        
        for step in tqdm(range(n_steps), desc="Optimizing prior precision", disable=not verbose):
            optimizer.zero_grad()
            
            # Update prior precision
            prior_prec = log_prior_prec.exp()
            
            # Recompute Hessian with new prior
            old_prior = self.prior_precision
            self.prior_precision = prior_prec.item()
            
            # Compute negative log marginal likelihood
            neg_log_marglik = -self.log_marginal_likelihood()
            
            neg_log_marglik.backward()
            optimizer.step()
            
            if verbose and (step + 1) % 20 == 0:
                print(f"Step {step+1}: prior_prec = {prior_prec.item():.4f}, "
                      f"neg_log_marglik = {neg_log_marglik.item():.4f}")
        
        self.prior_precision = log_prior_prec.exp().item()
        
        # Re-fit with optimized prior
        print(f"Optimized prior precision: {self.prior_precision:.4f}")
        
        return self.prior_precision
    
    def predictive_samples(
        self,
        data_loader: DataLoader,
        n_samples: int = 30,
        layer_subset: Optional[List[str]] = None,
        device: str = 'cuda'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictive samples for calibration analysis.
        Memory-efficient version that accumulates predictions incrementally.
        
        Args:
            data_loader: DataLoader for test data
            n_samples: Number of posterior samples
            layer_subset: If provided, only sample from these layers
            device: Device to run on
        
        Returns:
            Tuple of (averaged_logits, labels)
        """
        self.model.eval()
        
        # First pass: collect labels and allocate storage
        all_labels = []
        n_batches = 0
        batch_sizes = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    targets = batch[1]
                else:
                    targets = batch['labels']
                all_labels.append(targets)
                batch_sizes.append(targets.size(0))
                n_batches += 1
        
        all_labels = torch.cat(all_labels, dim=0)
        n_test = all_labels.size(0)
        n_classes = self.model.config.num_labels if hasattr(self.model, 'config') else 100
        
        # Accumulator for average logits (running mean)
        avg_logits = torch.zeros(n_test, n_classes)
        
        # Store original parameters
        original_params = {name: param.data.clone() 
                          for name, param in self.model.named_parameters() 
                          if name in self.mean}
        
        # Generate predictions for each sample
        for sample_idx in tqdm(range(n_samples), desc="Generating predictive samples"):
            # Sample new parameters
            sampled_params = self.sample_parameters(1, layer_subset)[0]
            
            # Set sampled parameters
            self.set_parameters(sampled_params)
            
            # Collect predictions for this sample
            sample_logits = []
            
            with torch.no_grad():
                for batch in data_loader:
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0].to(device)
                    else:
                        inputs = batch['pixel_values'].to(device)
                    
                    outputs = self.model(inputs)
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    
                    sample_logits.append(logits.cpu())
                    
                    # Clear GPU cache after each batch
                    del outputs, logits
                    if device == 'cuda':
                        torch.cuda.empty_cache()
            
            # Update running average
            sample_logits = torch.cat(sample_logits, dim=0)
            avg_logits = avg_logits + (sample_logits - avg_logits) / (sample_idx + 1)
            
            # Clear memory
            del sample_logits, sampled_params
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Restore original parameters
        self.set_parameters(original_params)
        
        return avg_logits, all_labels
    
    def glm_predictive(
        self,
        data_loader: DataLoader,
        link_approx: str = 'probit',
        device: str = 'cuda'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GLM predictive distribution using linearization.
        
        This provides a more efficient alternative to MC sampling by using
        a closed-form approximation to the predictive distribution.
        
        Args:
            data_loader: DataLoader for test data
            link_approx: Link approximation method ('probit' or 'bridge')
            device: Device to run on
        
        Returns:
            Tuple of (probabilities, labels)
        """
        self.model.eval()
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Computing GLM predictive"):
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0].to(device), batch[1]
                else:
                    inputs = batch['pixel_values'].to(device)
                    targets = batch['labels']
                
                # Get mean prediction
                outputs = self.model(inputs)
                if hasattr(outputs, 'logits'):
                    f_mu = outputs.logits
                else:
                    f_mu = outputs
                
                # For GLM predictive, we need the functional variance
                # f_var = J @ P^{-1} @ J^T
                # This requires computing Jacobians, which is expensive
                # Here we use a simpler approximation based on output variance
                
                if link_approx == 'probit':
                    # Probit approximation: scale logits by kappa
                    # kappa = 1 / sqrt(1 + pi/8 * sigma^2)
                    # Use average precision as variance estimate
                    avg_var = 1.0 / self.prior_precision  # Simplified variance estimate
                    kappa = 1.0 / sqrt(1.0 + pi / 8.0 * avg_var)
                    probs = torch.softmax(kappa * f_mu, dim=-1)
                    
                elif link_approx == 'bridge':
                    # Laplace bridge approximation
                    probs = torch.softmax(f_mu, dim=-1)
                    
                else:
                    # Default to standard softmax
                    probs = torch.softmax(f_mu, dim=-1)
                
                all_probs.append(probs.cpu())
                all_labels.append(targets)
        
        return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)
    
    def get_posterior_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the fitted posterior.
        
        Returns:
            Dictionary containing posterior statistics
        """
        if self.mean is None or self.precision is None:
            raise RuntimeError("Must call fit() before getting posterior stats")
        
        stats = {
            'backend': self.backend,
            'prior_precision': self.prior_precision,
            'n_parameters': sum(p.numel() for p in self.mean.values()),
            'n_layers': len(self.mean),
        }
        
        # Per-layer statistics
        layer_stats = {}
        for name, mean_val in self.mean.items():
            prec = self.precision[name]
            layer_stats[name] = {
                'n_params': mean_val.numel(),
                'mean_magnitude': mean_val.abs().mean().item(),
                'mean_precision': prec.mean().item(),
                'min_precision': prec.min().item(),
                'max_precision': prec.max().item(),
            }
        
        stats['layer_stats'] = layer_stats
        
        if self.backend == 'kfac':
            stats['n_kfac_factors'] = len(self.kfac_factors)
            kfac_stats = {}
            for name, factor in self.kfac_factors.items():
                kfac_stats[name] = {
                    'A_shape': list(factor.A.shape),
                    'B_shape': list(factor.B.shape),
                    'A_trace': factor.A.trace().item(),
                    'B_trace': factor.B.trace().item(),
                }
            stats['kfac_stats'] = kfac_stats
        
        return stats
    
    def save(self, path: str):
        """Save the Laplace approximation to a file."""
        state = {
            'mean': self.mean,
            'precision': self.precision,
            'prior_precision': self.prior_precision,
            'backend': self.backend,
            'likelihood': self.likelihood,
        }
        
        if self.backend == 'kfac':
            state['kfac_factors'] = {
                name: {'A': f.A, 'B': f.B} 
                for name, f in self.kfac_factors.items()
            }
            state['kfac_eigenvectors'] = self.kfac_eigenvectors
            state['kfac_eigenvalues'] = self.kfac_eigenvalues
        
        torch.save(state, path)
        print(f"Laplace approximation saved to {path}")
    
    def load(self, path: str):
        """Load the Laplace approximation from a file."""
        state = torch.load(path)
        
        self.mean = state['mean']
        self.precision = state['precision']
        self.prior_precision = state['prior_precision']
        self.backend = state['backend']
        self.likelihood = state['likelihood']
        
        if self.backend == 'kfac' and 'kfac_factors' in state:
            self.kfac_factors = {
                name: KroneckerFactor(f['A'], f['B'])
                for name, f in state['kfac_factors'].items()
            }
            self.kfac_eigenvectors = state.get('kfac_eigenvectors', {})
            self.kfac_eigenvalues = state.get('kfac_eigenvalues', {})
        
        print(f"Laplace approximation loaded from {path}")

"""
Unit tests and small demo for the pipeline.
Tests core functionality with a minimal subset of data.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import create_vit_lora, get_lora_layer_names
from src.laplace import LaplaceLoRA
from src.metrics import compute_ece, compute_top_label_ece, plot_reliability_diagram
from src.utils import set_seed, count_parameters


def create_dummy_data(n_samples=100, image_size=224, n_classes=100):
    """Create dummy data for testing."""
    images = torch.randn(n_samples, 3, image_size, image_size)
    labels = torch.randint(0, n_classes, (n_samples,))
    return TensorDataset(images, labels)


def test_model_creation():
    """Test ViT-LoRA model creation."""
    print("\n" + "="*60)
    print("TEST 1: Model Creation")
    print("="*60)
    
    model = create_vit_lora(num_classes=100, lora_r=8, lora_alpha=8)
    
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, only_trainable=True)
    
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Check layer names
    layer_names = get_lora_layer_names(model)
    print(f"  LoRA layers found: {len(layer_names)}")
    print(f"  Layer names: {layer_names[:3]}... (showing first 3)")
    
    assert trainable_params < total_params, "All parameters are trainable!"
    assert len(layer_names) > 0, "No LoRA layers found!"
    
    return model


def test_forward_pass(model):
    """Test forward pass."""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)
    
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output range: [{logits.min():.2f}, {logits.max():.2f}]")
    
    assert logits.shape == (2, 100), f"Unexpected output shape: {logits.shape}"
    
    return logits


def test_ece_computation():
    """Test ECE computation."""
    print("\n" + "="*60)
    print("TEST 3: ECE Computation")
    print("="*60)
    
    # Create mock predictions
    n_samples = 1000
    logits = torch.randn(n_samples, 100)
    labels = torch.randint(0, 100, (n_samples,))
    
    ece = compute_top_label_ece(logits, labels, n_bins=10)
    
    print(f"✓ ECE computation successful")
    print(f"  Test samples: {n_samples}")
    print(f"  ECE value: {ece:.4f}")
    
    assert 0 <= ece <= 1, f"ECE out of range: {ece}"
    
    # Test with perfect predictions
    perfect_logits = torch.zeros(n_samples, 100)
    perfect_logits[range(n_samples), labels] = 10.0
    perfect_ece = compute_top_label_ece(perfect_logits, labels, n_bins=10)
    
    print(f"  ECE (perfect predictions): {perfect_ece:.4f}")
    print(f"  → Should be close to 0")
    
    return ece


def test_laplace_fitting(model):
    """Test Laplace approximation fitting."""
    print("\n" + "="*60)
    print("TEST 4: Laplace Approximation")
    print("="*60)
    
    # Create small dataset
    dataset = create_dummy_data(n_samples=50)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    
    # Initialize Laplace
    laplace = LaplaceLoRA(
        model=model,
        prior_precision=1.0,
        backend='diagonal'
    )
    
    print("  Fitting Laplace approximation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    laplace.fit(loader, device=device)
    
    print(f"✓ Laplace fitting successful")
    print(f"  Device: {device}")
    print(f"  Backend: diagonal")
    print(f"  Parameters tracked: {len(laplace.mean)}")
    
    assert laplace.mean is not None, "Mean not set after fitting"
    assert laplace.precision is not None, "Precision not set after fitting"
    
    return laplace


def test_parameter_sampling(model, laplace):
    """Test parameter sampling from Laplace posterior."""
    print("\n" + "="*60)
    print("TEST 5: Parameter Sampling")
    print("="*60)
    
    # Sample parameters
    n_samples = 5
    samples = laplace.sample_parameters(n_samples=n_samples)
    
    print(f"✓ Parameter sampling successful")
    print(f"  Number of samples: {len(samples)}")
    print(f"  Parameters per sample: {len(samples[0])}")
    
    # Test layer-subset sampling
    layer_names = get_lora_layer_names(model)
    if len(layer_names) > 0:
        subset = [layer_names[0]]
        subset_samples = laplace.sample_parameters(n_samples=3, layer_subset=subset)
        print(f"  Subset sampling (layer {subset[0]}): ✓")
    
    assert len(samples) == n_samples, f"Expected {n_samples} samples, got {len(samples)}"
    
    return samples


def test_predictive_sampling(model, laplace):
    """Test predictive sampling."""
    print("\n" + "="*60)
    print("TEST 6: Predictive Sampling")
    print("="*60)
    
    # Create test dataset
    dataset = create_dummy_data(n_samples=20)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("  Generating predictive samples...")
    logits, labels = laplace.predictive_samples(
        loader,
        n_samples=3,
        device=device
    )
    
    print(f"✓ Predictive sampling successful")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # Compute ECE
    ece = compute_top_label_ece(logits, labels, n_bins=5)
    print(f"  ECE on dummy data: {ece:.4f}")
    
    assert logits.shape[0] == 20, "Wrong number of samples"
    assert logits.shape[1] == 100, "Wrong number of classes"
    
    return ece


def test_kfac_laplace(model):
    """Test KFAC Laplace approximation fitting."""
    print("\n" + "="*60)
    print("TEST 7: KFAC Laplace Approximation")
    print("="*60)
    
    # Create small dataset
    dataset = create_dummy_data(n_samples=50)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    
    # Initialize KFAC Laplace
    laplace = LaplaceLoRA(
        model=model,
        prior_precision=1.0,
        backend='kfac'
    )
    
    print("  Fitting KFAC Laplace approximation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    laplace.fit(loader, device=device)
    
    print(f"✓ KFAC Laplace fitting successful")
    print(f"  Backend: kfac")
    print(f"  Parameters tracked: {len(laplace.mean)}")
    print(f"  KFAC factors computed: {len(laplace.kfac_factors)}")
    
    # Test posterior stats
    try:
        stats = laplace.get_posterior_stats()
        print(f"  Posterior stats: n_params={stats['n_parameters']}, n_layers={stats['n_layers']}")
    except Exception as e:
        print(f"  Warning: Could not get posterior stats: {e}")
    
    # Test sampling with KFAC
    samples = laplace.sample_parameters(n_samples=3)
    print(f"  KFAC sampling: {len(samples)} samples generated")
    
    assert laplace.mean is not None, "Mean not set after fitting"
    assert laplace.precision is not None, "Precision not set after fitting"
    
    return laplace


def test_laplace_save_load(model, laplace, tmp_path="/tmp/test_laplace.pt"):
    """Test saving and loading Laplace approximation."""
    print("\n" + "="*60)
    print("TEST 8: Save/Load Laplace Approximation")
    print("="*60)
    
    # Save
    laplace.save(tmp_path)
    print(f"  Saved to: {tmp_path}")
    
    # Create new Laplace and load
    laplace_loaded = LaplaceLoRA(
        model=model,
        prior_precision=1.0,
        backend=laplace.backend
    )
    laplace_loaded.load(tmp_path)
    print(f"  Loaded from: {tmp_path}")
    
    # Verify
    assert len(laplace_loaded.mean) == len(laplace.mean), "Mean mismatch after loading"
    assert laplace_loaded.prior_precision == laplace.prior_precision, "Prior precision mismatch"
    print(f"✓ Save/Load successful")
    
    # Cleanup
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    
    return True


def test_end_to_end():
    """Run complete end-to-end test."""
    print("\n" + "="*70)
    print("RUNNING END-TO-END PIPELINE TEST")
    print("="*70)
    
    set_seed(42)
    
    try:
        # Test 1: Model creation
        model = test_model_creation()
        
        # Test 2: Forward pass
        test_forward_pass(model)
        
        # Test 3: ECE computation
        test_ece_computation()
        
        # Test 4: Laplace fitting (diagonal)
        laplace = test_laplace_fitting(model)
        
        # Test 5: Parameter sampling
        test_parameter_sampling(model, laplace)
        
        # Test 6: Predictive sampling
        test_predictive_sampling(model, laplace)
        
        # Test 7: KFAC Laplace
        kfac_laplace = test_kfac_laplace(model)
        
        # Test 8: Save/Load
        test_laplace_save_load(model, laplace)
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nThe pipeline is working correctly.")
        print("You can now proceed with full-scale experiments.")
        
    except Exception as e:
        print("\n" + "="*70)
        print("✗ TEST FAILED")
        print("="*70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = test_end_to_end()
    sys.exit(0 if success else 1)

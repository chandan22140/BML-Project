"""
Calibration metrics: ECE, reliability diagrams, etc.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        confidences: Max softmax probabilities, shape (n_samples,)
        predictions: Predicted class labels, shape (n_samples,)
        labels: True labels, shape (n_samples,)
        n_bins: Number of bins for calibration
    
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get indices of samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def compute_top_label_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15
) -> float:
    """
    Compute top-label ECE from logits and labels.
    
    Args:
        logits: Model output logits, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples,)
        n_bins: Number of bins
    
    Returns:
        ECE value
    """
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    
    confidences = confidences.cpu().numpy()
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    return compute_ece(confidences, predictions, labels, n_bins)


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute classification accuracy."""
    return float((predictions == labels).mean())


def plot_reliability_diagram(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None
):
    """
    Plot reliability diagram showing calibration.
    
    Args:
        confidences: Max softmax probabilities
        predictions: Predicted labels
        labels: True labels
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure (optional)
    """
    # Compute bin statistics
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.sum()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(prop_in_bin)
        else:
            bin_accuracies.append(0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    
    # Plot reliability diagram
    mask = bin_counts > 0
    ax.bar(
        bin_confidences[mask],
        bin_accuracies[mask],
        width=1.0/n_bins,
        alpha=0.7,
        edgecolor='black',
        label='Model calibration'
    )
    
    # Compute and display ECE
    ece = compute_ece(confidences, predictions, labels, n_bins)
    ax.text(0.05, 0.95, f'ECE = {ece:.4f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reliability diagram saved to {save_path}")
    
    return fig


def compute_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute negative log-likelihood.
    
    Args:
        logits: Model output, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples,)
    
    Returns:
        Mean NLL
    """
    log_probs = torch.log_softmax(logits, dim=1)
    nll = -log_probs[range(len(labels)), labels].mean()
    return float(nll.item())


def compute_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Brier score.
    
    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples,)
    
    Returns:
        Brier score
    """
    n_classes = probs.shape[1]
    targets_onehot = torch.zeros_like(probs)
    targets_onehot[range(len(labels)), labels] = 1.0
    
    brier = ((probs - targets_onehot) ** 2).sum(dim=1).mean()
    return float(brier.item())

# Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PROJECT WORKFLOW OVERVIEW                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: SETUP (5 minutes)                                          │
├─────────────────────────────────────────────────────────────────────┤
│ ./quickstart.sh                                                     │
│   ├─► Create virtual environment                                   │
│   ├─► Install dependencies (PyTorch, Transformers, PEFT, etc.)     │
│   └─► Run unit tests                                               │
│                                                                     │
│ Output: ✓ Verified working environment                             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: TRAINING (20-30 minutes)                                   │
├─────────────────────────────────────────────────────────────────────┤
│ python scripts/train_lora_vit.py                                   │
│                                                                     │
│ CIFAR-100 Train Set                                                │
│        │                                                            │
│        ▼                                                            │
│ ┌─────────────────┐                                                │
│ │ ViT-B/16 (Pre.) │  ←── Add LoRA adapters (r=16, α=16)            │
│ └─────────────────┘                                                │
│        │                                                            │
│        ▼                                                            │
│ ┌─────────────────┐                                                │
│ │  Fine-tune LoRA │  ←── Cross-entropy + L2 prior (λ=1.0)          │
│ └─────────────────┘                                                │
│        │                                                            │
│        ▼                                                            │
│ ┌─────────────────┐                                                │
│ │ θ_MAP (Best)    │  ←── Saved to checkpoints/model_map.pt         │
│ └─────────────────┘                                                │
│                                                                     │
│ Output: MAP checkpoint (~85% test accuracy expected)               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: LAPLACE FITTING (5-10 minutes)                             │
├─────────────────────────────────────────────────────────────────────┤
│ python scripts/eval_bayesian_lora.py                               │
│                                                                     │
│ θ_MAP + Training Data                                              │
│        │                                                            │
│        ▼                                                            │
│ ┌─────────────────────────────────────────┐                        │
│ │  Compute Hessian (Diagonal/KFAC)        │                        │
│ │  H = ∇²_θ [-log p(D|θ) - log p(θ)]     │                        │
│ └─────────────────────────────────────────┘                        │
│        │                                                            │
│        ▼                                                            │
│ ┌─────────────────────────────────────────┐                        │
│ │  Posterior: p(θ) = N(θ_MAP, H^-1)      │                        │
│ └─────────────────────────────────────────┘                        │
│                                                                     │
│ Output: Fitted Laplace approximation                               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 4: LAYER-WISE EVALUATION (1-2 hours)                          │
├─────────────────────────────────────────────────────────────────────┤
│ For each layer ℓ ∈ {0, 1, ..., 11}:                                │
│                                                                     │
│   1. Construct restricted posterior p_ℓ(θ)                         │
│      ├─► Only layer ℓ parameters are stochastic                    │
│      └─► All other layers fixed at θ_MAP                           │
│                                                                     │
│   2. Sample N=30 parameter sets from p_ℓ(θ)                        │
│                                                                     │
│   3. Generate predictions on CIFAR-100 test set                    │
│      ├─► For each sample: logits_i = f(x; θ_i)                     │
│      └─► Average: logits_avg = (1/N) Σ logits_i                    │
│                                                                     │
│   4. Compute calibration metrics                                   │
│      ├─► ECE_Bayes(ℓ) with M=15 bins                               │
│      ├─► Accuracy                                                  │
│      └─► Δ_ECE(ℓ) = ECE_det - ECE_Bayes(ℓ)                         │
│                                                                     │
│ Additionally:                                                       │
│   • Evaluate deterministic (MAP) baseline                          │
│   • Evaluate full Bayesian (all layers)                            │
│                                                                     │
│ Output: results/layer_wise_results.json                            │
│         results/plots/reliability_*.png                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 5: STATISTICAL ANALYSIS (< 5 minutes)                         │
├─────────────────────────────────────────────────────────────────────┤
│ jupyter notebook notebooks/analysis.ipynb                          │
│                                                                     │
│ 1. Load layer-wise results                                         │
│                                                                     │
│ 2. Rank layers by Δ_ECE                                            │
│    ┌────────────────────────────────────┐                          │
│    │ Layer    │ Δ_ECE    │ Rank         │                          │
│    ├──────────┼──────────┼──────────────┤                          │
│    │ layer.5  │ +0.0234  │ 1  (Best)    │                          │
│    │ layer.8  │ +0.0198  │ 2            │                          │
│    │ layer.11 │ +0.0156  │ 3            │                          │
│    │ ...      │ ...      │ ...          │                          │
│    └────────────────────────────────────┘                          │
│                                                                     │
│ 3. Generate visualizations                                         │
│    ├─► Bar chart: Δ_ECE by layer (sorted)                          │
│    ├─► Line plot: Δ_ECE vs layer depth                             │
│    ├─► Comparison: ECE_det vs ECE_Bayes                            │
│    └─► Reliability diagrams for top layers                         │
│                                                                     │
│ 4. Statistical tests                                               │
│    ├─► Correlation: layer depth ↔ Δ_ECE                            │
│    ├─► Summary statistics (mean, std, min, max)                    │
│    └─► Significance analysis                                       │
│                                                                     │
│ 5. Export results                                                  │
│    ├─► summary_table.csv                                           │
│    ├─► analysis_summary.json                                       │
│    └─► All plots saved to results/plots/                           │
│                                                                     │
│ Output: Comprehensive analysis with actionable insights            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FINAL DELIVERABLES                                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ ✓ Ranked table: Which layers improve calibration most?             │
│ ✓ Visualizations: ECE improvements across layers                   │
│ ✓ Statistical analysis: Correlations and significance              │
│ ✓ Deployment recommendation: Best single layer for efficiency      │
│                                                                     │
│ Key Finding Example:                                               │
│   "Layer 5 achieves 73% of full Bayesian calibration benefit       │
│    while requiring uncertainty quantification for only 8% of       │
│    parameters → optimal for low-cost deployment"                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════
                      RESEARCH CONTRIBUTIONS
═══════════════════════════════════════════════════════════════════════

1. Layer-wise attribution of calibration improvements
2. Identification of most impactful layers for uncertainty quantification
3. Cost-benefit analysis for practical Bayesian deployment
4. Quantitative ranking with statistical rigor
5. Reproducible experimental framework

═══════════════════════════════════════════════════════════════════════
```

## Quick Command Reference

```bash
# Complete pipeline (one command)
python run_experiment.py --config config.yaml

# Individual stages
python run_experiment.py --stage train    # Training only
python run_experiment.py --stage eval     # Evaluation only
python run_experiment.py --stage analyze  # Analysis only

# Manual execution
python scripts/train_lora_vit.py --output_dir checkpoints/vit_lora_cifar100
python scripts/eval_bayesian_lora.py --checkpoint checkpoints/vit_lora_cifar100/model_map.pt
jupyter notebook notebooks/analysis.ipynb
```

## Expected Timeline

| Phase | Duration | Bottleneck |
|-------|----------|------------|
| Setup | 5 min | Network (downloading deps) |
| Training | 20-30 min | GPU compute |
| Laplace Fitting | 5-10 min | GPU compute |
| Layer-wise Eval | 1-2 hrs | Multiple inference passes |
| Analysis | < 5 min | I/O |
| **Total** | **~2.5 hrs** | Evaluation phase |

## Resource Requirements

- **GPU Memory:** 16GB minimum (24GB recommended)
- **Disk Space:** ~5GB (models + results)
- **CPU Memory:** 16GB+
- **Network:** Required for dataset/model download

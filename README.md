# Layer-wise Analysis of Laplace Approximations in Low-Rank Adaptation

This project implements a systematic measurement of individual transformer layer contributions to model calibration when Bayesian posterior uncertainty is applied via Laplace approximation over LoRA adapters.

## Goal

Measure which transformer layers, when treated as Bayesian via a Laplace approximation over LoRA parameters, produce the largest improvement in calibration (ECE) on CIFAR-100 (ViT-B/16), and produce a ranked, statistically sound attribution of calibration gains to layers.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── laplace.py          # Laplace approximation (KFAC, diagonal)
│   ├── models.py           # ViT-B/16 with LoRA
│   ├── metrics.py          # ECE computation, reliability diagrams
│   └── utils.py            # Helper functions
├── scripts/
│   ├── train_lora_vit.py   # Fine-tune LoRA on CIFAR-100
│   └── eval_bayesian_lora.py  # Evaluate layer-wise Bayesian posteriors
├── notebooks/
│   └── analysis.ipynb      # Statistical analysis and visualization
└── tests/
    └── test_pipeline.py    # Unit tests and small demo

```

## Installation

```bash
pip install -r requirements.txt
```

### Hardware Requirements

- GPU with at least 16GB VRAM recommended for ViT-B/16
- CPU fallback supported but slower

## Usage

### 1. Fine-tune LoRA adapters on CIFAR-100

```bash
python scripts/train_lora_vit.py \
    --output_dir checkpoints/vit_lora_cifar100 \
    --epochs 20 \
    --batch_size 128 \
    --lora_r 16 \
    --lora_alpha 16
```

### 2. Evaluate layer-wise Bayesian posteriors

```bash
python scripts/eval_bayesian_lora.py \
    --checkpoint checkpoints/vit_lora_cifar100/model_map.pt \
    --output_dir results/ \
    --num_samples 50 \
    --laplace_type kfac
```

### 3. Analyze results

Open and run `notebooks/analysis.ipynb` to:
- Compute Δ_ECE per layer
- Generate bootstrap confidence intervals
- Create ranked visualizations
- Produce reliability diagrams

## Methodology

### Model and Training
- **Base model:** ViT-B/16 pretrained checkpoint
- **PEFT:** LoRA adapters in attention/MLP projections
- **Dataset:** CIFAR-100 (train/test split)
- **Optimization:** MAP estimation with isotropic Gaussian prior

### Layer-wise Restricted Posterior

For subset S ⊆ {1,...,L}, construct restricted posterior:
```
p_S(θ) = N(θ_MAP, Σ_S)
```
where Σ_S has non-zero blocks only for parameters in S.

### Primary Metric: ECE

Expected Calibration Error with M bins:
```
ECE = Σ_{m=1}^M (|B_m|/n) |acc(B_m) - conf(B_m)|
```

Delta ECE per layer:
```
Δ_ECE(ℓ) = ECE_det - ECE_Bayes(ℓ)
```

## Deliverables

1. ✓ Code repository with complete pipeline
2. ✓ LoRA fine-tuning script
3. ✓ Bayesian evaluation and sampling
4. ✓ Statistical analysis notebook
5. Results package:
   - Table of Δ_ECE(ℓ) for all layers
   - Reliability diagrams
   - Ranked bar plots and scatter plots
   - Interpretation and deployment recommendations

## References

- Bayesian Low-Rank Adaptation for Large Language Models (ICLR 2024)
- LoRA: Low-Rank Adaptation of Large Language Models
- On Calibration of Modern Neural Networks

## Citation

If you use this code, please cite:
```
[To be added after publication]
```

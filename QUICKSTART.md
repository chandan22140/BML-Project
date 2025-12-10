# Layer-wise Laplace LoRA Analysis

## Quick Setup

```bash
# Make quickstart script executable
chmod +x quickstart.sh

# Run quickstart (creates venv, installs deps, runs tests)
./quickstart.sh
```

## Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/test_pipeline.py
```

## Usage Workflow

### 1. Train LoRA on CIFAR-100

```bash
python scripts/train_lora_vit.py \
  --output_dir checkpoints/vit_lora_cifar100 \
  --epochs 20 \
  --batch_size 128 \
  --lora_r 16 \
  --lora_alpha 16
```

**Time:** ~20-30 minutes on single GPU (V100/A100)

### 2. Evaluate Layer-wise Posteriors

```bash
python scripts/eval_bayesian_lora.py \
  --checkpoint checkpoints/vit_lora_cifar100/model_map.pt \
  --output_dir results/ \
  --num_samples 30 \
  --laplace_type diagonal
```

**Time:** ~1-2 hours (depends on num_samples and number of layers)

### 3. Analyze Results

```bash
jupyter notebook notebooks/analysis.ipynb
```

Or run programmatically:
```bash
jupyter nbconvert --execute notebooks/analysis.ipynb
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── quickstart.sh
├── src/
│   ├── __init__.py
│   ├── laplace.py       # Laplace approximation (KFAC/diagonal)
│   ├── models.py        # ViT-B/16 with LoRA
│   ├── metrics.py       # ECE, reliability diagrams
│   └── utils.py         # Utilities
├── scripts/
│   ├── train_lora_vit.py      # Training script
│   └── eval_bayesian_lora.py  # Evaluation script
├── notebooks/
│   └── analysis.ipynb   # Statistical analysis
└── tests/
    └── test_pipeline.py # Unit tests
```

## Expected Outputs

### Training
- `checkpoints/vit_lora_cifar100/model_map.pt` - Best MAP model
- Training logs showing loss and accuracy

### Evaluation
- `results/layer_wise_results.json` - Raw ECE data per layer
- `results/plots/reliability_*.png` - Reliability diagrams
- Console output with Δ_ECE rankings

### Analysis
- `results/summary_table.csv` - Ranked summary table
- `results/analysis_summary.json` - Key findings
- `results/plots/delta_ece_ranked.png` - Bar chart
- `results/plots/delta_ece_sequential.png` - Line plot
- `results/plots/ece_comparison.png` - Comparison plot

## Key Metrics

- **ECE**: Expected Calibration Error (lower is better)
- **Δ_ECE**: Calibration improvement (ECE_det - ECE_Bayes)
- Positive Δ_ECE means the Bayesian approach improves calibration

## Hardware Requirements

- **Minimum**: 16GB GPU (e.g., V100, RTX 3090)
- **Recommended**: 24GB+ GPU (e.g., A100, RTX 4090)
- **CPU**: Can run on CPU but ~10x slower

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (e.g., 64 or 32)
- Reduce `--num_samples` (e.g., 20 or 10)
- Use `--laplace_type diagonal` instead of `kfac`

### Slow Training
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU usage: `nvidia-smi`
- Increase batch size if memory allows

### Import Errors
- Verify all dependencies installed: `pip list`
- Reinstall if needed: `pip install -r requirements.txt --force-reinstall`

## Citation

Based on:
- Bayesian Low-Rank Adaptation for Large Language Models (ICLR 2024)
- LoRA: Low-Rank Adaptation of Large Language Models

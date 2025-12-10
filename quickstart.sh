#!/bin/bash
# Quick start script for the project

echo "=========================================="
echo "Layer-wise Laplace LoRA Analysis"
echo "Quick Start Script"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo ""
echo "Step 1: Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo ""
echo "Step 2: Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Step 3: Running pipeline tests..."
python tests/test_pipeline.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Tests passed! Pipeline is ready."
    echo ""
    echo "=========================================="
    echo "Next steps:"
    echo "=========================================="
    echo ""
    echo "1. Train LoRA adapters (20-30 mins on GPU):"
    echo "   python scripts/train_lora_vit.py \\"
    echo "     --output_dir checkpoints/vit_lora_cifar100 \\"
    echo "     --epochs 20 \\"
    echo "     --batch_size 128"
    echo ""
    echo "2. Evaluate layer-wise Bayesian posteriors (1-2 hours):"
    echo "   python scripts/eval_bayesian_lora.py \\"
    echo "     --checkpoint checkpoints/vit_lora_cifar100/model_map.pt \\"
    echo "     --output_dir results/ \\"
    echo "     --num_samples 30"
    echo ""
    echo "3. Analyze results:"
    echo "   jupyter notebook notebooks/analysis.ipynb"
    echo ""
else
    echo ""
    echo "✗ Tests failed. Please check the error messages above."
    exit 1
fi

# Makefile for Layer-wise Laplace LoRA Analysis
# Quick commands for common tasks

.PHONY: help setup test train eval analyze clean all

# Default target
help:
	@echo "Layer-wise Laplace LoRA Analysis - Available Commands:"
	@echo ""
	@echo "  make setup       - Setup environment and install dependencies"
	@echo "  make test        - Run unit tests"
	@echo "  make train       - Train LoRA on CIFAR-100"
	@echo "  make eval        - Evaluate layer-wise Bayesian posteriors"
	@echo "  make analyze     - Run statistical analysis"
	@echo "  make all         - Run complete pipeline (train + eval + analyze)"
	@echo "  make clean       - Remove generated files (checkpoints, results)"
	@echo "  make clean-all   - Remove everything including venv"
	@echo ""

# Setup environment
setup:
	@echo "Setting up environment..."
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "✓ Setup complete. Activate with: source venv/bin/activate"

# Run tests
test:
	@echo "Running tests..."
	python tests/test_pipeline.py

# Train LoRA
train:
	@echo "Training LoRA on CIFAR-100..."
	python scripts/train_lora_vit.py \
		--output_dir checkpoints/vit_lora_cifar100 \
		--epochs 20 \
		--batch_size 128 \
		--lora_r 16 \
		--lora_alpha 16

# Evaluate Bayesian
eval:
	@echo "Evaluating layer-wise Bayesian posteriors..."
	python scripts/eval_bayesian_lora.py \
		--checkpoint checkpoints/vit_lora_cifar100/model_map.pt \
		--output_dir results/ \
		--num_samples 30 \
		--laplace_type diagonal

# Run analysis
analyze:
	@echo "Running statistical analysis..."
	jupyter nbconvert --execute --to notebook --inplace notebooks/analysis.ipynb
	@echo "✓ Analysis complete. Results in results/"

# Run complete pipeline
all:
	@echo "Running complete pipeline..."
	python run_experiment.py --config config.yaml --stage all

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf checkpoints/
	rm -rf results/
	rm -rf logs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cleaned"

# Clean everything including venv
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf venv/
	@echo "✓ All cleaned"

# Quick test (minimal data, fast)
quick-test:
	@echo "Running quick test with minimal settings..."
	python scripts/train_lora_vit.py \
		--output_dir checkpoints/quick_test \
		--epochs 2 \
		--batch_size 32
	python scripts/eval_bayesian_lora.py \
		--checkpoint checkpoints/quick_test/model_map.pt \
		--output_dir results_test/ \
		--num_samples 5
	@echo "✓ Quick test complete"

# Install in development mode
dev-install:
	pip install -e .
	pip install pytest black flake8 mypy

# Format code
format:
	black src/ scripts/ tests/
	@echo "✓ Code formatted"

# Lint code
lint:
	flake8 src/ scripts/ tests/ --max-line-length=100
	@echo "✓ Linting complete"

# Show project info
info:
	@echo "Project: Layer-wise Laplace LoRA Analysis"
	@echo "Python: $$(python --version 2>&1)"
	@echo "PyTorch: $$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA Available: $$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
	@echo "GPU: $$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"

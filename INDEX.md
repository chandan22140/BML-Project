# 📚 Project File Index

## Quick Navigation Guide

---

## 🚀 Getting Started (Start Here!)

1. **STATUS.md** - Project completion status and overview
2. **QUICKSTART.md** - 5-minute quick start guide
3. **README.md** - Main documentation
4. **quickstart.sh** - One-command setup script

---

## 📖 Documentation Files

| File | Purpose | Size |
|------|---------|------|
| **README.md** | Main project documentation | 3.4K |
| **QUICKSTART.md** | Quick start instructions | 3.4K |
| **STATUS.md** | Implementation status report | 8.4K |
| **PROJECT_SUMMARY.md** | Comprehensive overview | 8.1K |
| **WORKFLOW.md** | Visual workflow diagram | 16K |
| **LICENSE** | MIT License | 1.1K |

**Reading Order:** STATUS.md → QUICKSTART.md → README.md → WORKFLOW.md

---

## 💻 Source Code

### Core Library (`src/`)

| File | Purpose | LOC | Description |
|------|---------|-----|-------------|
| **models.py** | Model architecture | 127 | ViT-B/16 + LoRA adapters |
| **laplace.py** | Laplace approximation | 239 | Diagonal/KFAC, sampling |
| **metrics.py** | Calibration metrics | 179 | ECE, reliability diagrams |
| **utils.py** | Utilities | 98 | Checkpointing, I/O |
| **__init__.py** | Package init | 3 | Version info |

**Total:** 646 lines

### Execution Scripts (`scripts/`)

| File | Purpose | LOC | Runtime |
|------|---------|-----|---------|
| **train_lora_vit.py** | Training | 271 | 20-30 min |
| **eval_bayesian_lora.py** | Evaluation | 272 | 1-2 hours |

**Total:** 543 lines

### Testing (`tests/`)

| File | Purpose | LOC | Description |
|------|---------|-----|-------------|
| **test_pipeline.py** | Integration tests | 217 | End-to-end verification |

**Total:** 217 lines

### Analysis (`notebooks/`)

| File | Purpose | Sections | Description |
|------|---------|----------|-------------|
| **analysis.ipynb** | Statistical analysis | 10 | Plots, stats, recommendations |

---

## ⚙️ Configuration & Automation

| File | Purpose | Description |
|------|---------|-------------|
| **config.yaml** | Configuration | Experiment parameters |
| **run_experiment.py** | Orchestration | Automated pipeline runner |
| **Makefile** | Build automation | Common commands (make all, etc.) |
| **quickstart.sh** | Setup script | Environment setup |
| **requirements.txt** | Dependencies | Python packages |
| **.gitignore** | Git exclusions | Ignore patterns |

---

## 📊 File Statistics

### By Category

| Category | Files | Lines | Size |
|----------|-------|-------|------|
| Documentation | 6 | ~5,000 | 40K |
| Source Code | 5 | 646 | 22K |
| Scripts | 2 | 543 | 20K |
| Tests | 1 | 217 | 7K |
| Config/Auto | 6 | ~300 | 11K |
| **Total** | **20** | **~6,700** | **100K** |

### By Language

| Language | Files | Lines | Percentage |
|----------|-------|-------|------------|
| Python | 9 | 1,522 | 23% |
| Markdown | 6 | ~5,000 | 75% |
| YAML | 1 | 50 | <1% |
| Shell | 1 | 60 | <1% |
| Makefile | 1 | 100 | 1% |

---

## 🎯 Usage Scenarios

### Scenario 1: Quick Test (First Time User)
```
Read: STATUS.md → QUICKSTART.md
Run: ./quickstart.sh
```

### Scenario 2: Understand Implementation
```
Read: PROJECT_SUMMARY.md → WORKFLOW.md
Browse: src/*.py
```

### Scenario 3: Run Experiments
```
Read: README.md (Usage section)
Configure: config.yaml
Run: python run_experiment.py
```

### Scenario 4: Analyze Results
```
Run: jupyter notebook notebooks/analysis.ipynb
View: results/plots/*.png
Read: results/summary_table.csv
```

### Scenario 5: Extend/Modify
```
Study: src/laplace.py (for algorithm changes)
Study: src/models.py (for architecture changes)
Test: python tests/test_pipeline.py
```

---

## 🔍 Key File Details

### Most Important Files

1. **scripts/train_lora_vit.py** (271 lines)
   - Entry point for training
   - MAP estimation
   - CIFAR-100 data loading

2. **scripts/eval_bayesian_lora.py** (272 lines)
   - Entry point for evaluation
   - Layer-wise analysis
   - Δ_ECE computation

3. **src/laplace.py** (239 lines)
   - Core algorithm implementation
   - Posterior sampling
   - Predictive inference

4. **notebooks/analysis.ipynb** (10 sections)
   - Statistical analysis
   - Visualization
   - Recommendations

5. **WORKFLOW.md** (16K)
   - Complete pipeline visualization
   - Timeline and resource estimates

### Quick Reference

| Need | File |
|------|------|
| Install dependencies | requirements.txt |
| Configure experiment | config.yaml |
| Understand workflow | WORKFLOW.md |
| Run training | scripts/train_lora_vit.py |
| Evaluate results | scripts/eval_bayesian_lora.py |
| Analyze data | notebooks/analysis.ipynb |
| Common commands | Makefile |
| API reference | src/*.py (docstrings) |

---

## 🛠️ Development Guide

### Adding New Features

1. **New metric:** Edit `src/metrics.py`
2. **New model:** Edit `src/models.py`
3. **New approximation:** Edit `src/laplace.py`
4. **New visualization:** Edit `notebooks/analysis.ipynb`

### Testing Changes

```bash
# Run tests
python tests/test_pipeline.py

# Quick experiment
make quick-test

# Full pipeline
python run_experiment.py
```

### Code Style

- Follow PEP 8
- Add docstrings to all functions
- Include type hints where helpful
- Update tests when modifying code

---

## 📦 Dependencies (requirements.txt)

Core:
- torch ≥2.0.0
- transformers ≥4.30.0
- peft ≥0.4.0

Laplace:
- laplace-torch ≥0.1.0
- backpack-for-pytorch ≥1.5.0

Data/ML:
- datasets ≥2.14.0
- scikit-learn ≥1.3.0

Visualization:
- matplotlib ≥3.7.0
- seaborn ≥0.12.0

Notebooks:
- jupyter ≥1.0.0

---

## 🎓 Learning Path

### Beginner
1. Read STATUS.md
2. Run quickstart.sh
3. Read QUICKSTART.md
4. Run make test

### Intermediate
1. Read README.md
2. Study WORKFLOW.md
3. Browse src/*.py
4. Run full experiment

### Advanced
1. Read PROJECT_SUMMARY.md
2. Study src/laplace.py in detail
3. Modify config.yaml
4. Extend implementation

---

## 📞 Support Resources

| Question | Answer Location |
|----------|----------------|
| How do I start? | QUICKSTART.md |
| What does this do? | PROJECT_SUMMARY.md |
| How does it work? | WORKFLOW.md |
| How do I configure? | config.yaml + README.md |
| What's the API? | src/*.py docstrings |
| How do I test? | tests/test_pipeline.py |
| How do I contribute? | LICENSE (MIT) |

---

## ✅ Checklist: Have You...

- [ ] Read STATUS.md for overview?
- [ ] Run ./quickstart.sh for setup?
- [ ] Reviewed config.yaml for parameters?
- [ ] Checked requirements.txt for dependencies?
- [ ] Run tests with make test?
- [ ] Understood WORKFLOW.md pipeline?
- [ ] Configured hardware (16GB+ GPU)?

---

**Last Updated:** November 29, 2025  
**Project Status:** ✅ Complete & Ready

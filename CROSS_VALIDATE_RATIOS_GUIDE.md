# 5-Fold Cross-Validation Training Guide

## Overview

This guide explains how to train models on the 3 class ratio datasets using **5-fold cross-validation** for robust, publication-ready results.

## What is 5-Fold Cross-Validation?

Instead of training a single model on a fixed train/val split, 5-fold CV:

- Splits the **entire training dataset** into 5 folds
- Trains 5 separate models (1 per fold)
- Each fold gets a unique validation set
- Reports average metrics + standard deviation across folds
- More robust: reduces variance from random data split
- Better for small-to-medium datasets

## Directory Structure

### Checkpoints (5 folds per ratio)

```
checkpoints/ratio_experiments_cv/
├── ratio_1to1/
│   ├── cv_fold_1/best_model.pt
│   ├── cv_fold_2/best_model.pt
│   ├── cv_fold_3/best_model.pt
│   ├── cv_fold_4/best_model.pt
│   └── cv_fold_5/best_model.pt
├── ratio_3to1/
│   ├── cv_fold_1-5/...
└── ratio_4to1/
    ├── cv_fold_1-5/...
```

### Metrics (aggregated results)

```
metrics/ratio_experiments_cv/
├── ratio_1to1/cv_metrics.json    (fold_results + mean metrics)
├── ratio_3to1/cv_metrics.json
└── ratio_4to1/cv_metrics.json
```

## Quick Start

### 1. Run 5-Fold CV on All Ratios

```bash
python cross_validate_ratios.py --ratio all
```

This trains 15 models total (5 folds × 3 ratios) sequentially.

**Output:**

```
============================================================
5-FOLD CV RATIO EXPERIMENTS SUMMARY
============================================================
  ratio_1to1: ✓ PASSED
  ratio_3to1: ✓ PASSED
  ratio_4to1: ✓ PASSED
============================================================

============================================================
5-FOLD CV RATIO METRICS COMPARISON
============================================================

ratio_1to1:
  Mean PR-AUC:  0.8541 ± 0.0123
  Mean F1:      0.7823 ± 0.0234
  Mean Prec:    0.7654
  Mean Recall:  0.7912
  Folds: 5

ratio_3to1:
  Mean PR-AUC:  0.7912 ± 0.0189
  Mean F1:      0.7234 ± 0.0312
  ...
```

### 2. Run 5-Fold CV on Single Ratio

```bash
python cross_validate_ratios.py --ratio 1to1
python cross_validate_ratios.py --ratio 3to1
python cross_validate_ratios.py --ratio 4to1
```

### 3. Skip Existing Checkpoints

```bash
python cross_validate_ratios.py --ratio all --skip-existing
```

### 4. View Detailed Fold Results

```bash
python cross_validate_ratios.py --compare-only --detailed-table
```

**Output:**

```
============================================================
DETAILED FOLD RESULTS
============================================================

ratio_1to1:
─────────────────────────────────────────────────────────────
Fold   PR-AUC      F1          Prec        Recall      Acc
─────────────────────────────────────────────────────────────
1      0.8512      0.7801      0.7634      0.7923      0.8234
2      0.8623      0.7845      0.7712      0.7945      0.8301
3      0.8512      0.7812      0.7645      0.7934      0.8267
4      0.8598      0.7834      0.7698      0.7921      0.8289
5      0.8467      0.7789      0.7601      0.7876      0.8145

MEAN   0.8542±0.012 0.7816±0.0024 ...
```

## Comparison Tools

### View All Results at a Glance

```bash
# Compare all results (single-fold + CV)
python train_ratio_master.py --compare-all
```

### Analyze Fold Stability

```bash
python ratio_cv_utils.py --stability
```

Shows which ratio has most consistent performance across folds:

```
============================================================
FOLD STABILITY ANALYSIS (Lower std = More stable)
============================================================
Ratio           PR-AUC Std      F1 Std          Recall Std      Avg Std
────────────────────────────────────────────────────────────────────────
ratio_1to1      0.0123          0.0024          0.0031          0.0059
ratio_3to1      0.0189          0.0312          0.0145          0.0215
ratio_4to1      0.0267          0.0456          0.0234          0.0319
```

Lower values = more stable (better generalization)

### Show Details for Specific Ratio

```bash
python ratio_cv_utils.py --ratio 1to1
python ratio_cv_utils.py --ratio 3to1
```

## Loading CV Models in Python

### Load a Specific Fold Model

```python
from ratio_cv_utils import RatioCVModelLoader

# Load fold 1 from ratio_3to1
model = RatioCVModelLoader.load_fold_model("3to1", fold=1, device="cuda")
```

### Load All 5 Folds for a Ratio

```python
from ratio_cv_utils import RatioCVModelLoader

# Load all folds for ensemble predictions
models = RatioCVModelLoader.load_all_folds_for_ratio("3to1")
# models = {1: model1, 2: model2, 3: model3, 4: model4, 5: model5}

# Use for ensemble predictions
predictions = []
for fold_id, model in models.items():
    pred = model(x)
    predictions.append(pred)
ensemble_pred = sum(predictions) / len(predictions)  # Average
```

### Load CV Metrics

```python
from ratio_cv_utils import load_cv_metrics, calculate_fold_statistics

metrics = load_cv_metrics("3to1")
fold_results = metrics["fold_results"]  # List of 5 fold results
stats = calculate_fold_statistics(fold_results)  # Mean ± std
```

## Complete Workflow Example

### Step 1: Quick Single-Fold Baseline

```bash
# Fast baseline with 1 model per ratio
python train_ratio_experiments.py --ratio all
python train_ratio_experiments.py --compare-only
```

### Step 2: Robust 5-Fold Evaluation

```bash
# Run 5-fold CV for publication-ready results
python cross_validate_ratios.py --ratio all
python cross_validate_ratios.py --compare-only --detailed-table
```

### Step 3: Analyze & Compare

```bash
# View fold stability
python ratio_cv_utils.py --stability

# Overall comparison
python train_ratio_master.py --compare-all
```

### Step 4: Select Best Ratio & Load Model

```python
from ratio_cv_utils import load_cv_metrics, calculate_fold_statistics

# Find best ratio by PR-AUC
best_metrics = None
best_ratio = None
for ratio in ["1to1", "3to1", "4to1"]:
    metrics = load_cv_metrics(ratio)
    fold_results = metrics["fold_results"]
    stats = calculate_fold_statistics(fold_results)
    pr_auc_mean = stats["PR-AUC"]["mean"]

    if best_metrics is None or pr_auc_mean > best_metrics["PR-AUC"]["mean"]:
        best_metrics = stats
        best_ratio = ratio

print(f"Best ratio: {best_ratio} (PR-AUC: {best_metrics['PR-AUC']['mean']:.4f})")

# Load the best fold
from ratio_cv_utils import RatioCVModelLoader
best_model = RatioCVModelLoader.load_fold_model(best_ratio, fold=1)
```

## Command Reference

### Training

```bash
# Single ratio
python cross_validate_ratios.py --ratio 1to1

# All ratios (default)
python cross_validate_ratios.py --ratio all

# Skip if already trained
python cross_validate_ratios.py --ratio all --skip-existing

# Override max epochs
python cross_validate_ratios.py --ratio all --max-epochs 100
```

### Analysis

```bash
# Summary comparison
python cross_validate_ratios.py --compare-only

# Detailed fold results
python cross_validate_ratios.py --compare-only --detailed-table

# Fold stability analysis
python ratio_cv_utils.py --stability

# Single ratio details
python ratio_cv_utils.py --ratio 1to1
```

### Master Script (Combined)

```bash
# Single-fold only
python train_ratio_master.py --mode single

# 5-fold CV only
python train_ratio_master.py --mode cv

# Both approaches
python train_ratio_master.py --mode both

# Show all results
python train_ratio_master.py --compare-all

# Print workflow guide
python train_ratio_master.py --guide
```

## Understanding the Metrics

### CV Metrics JSON Structure

```json
{
  "fold_results": [
    {
      "fold": 1,
      "threshold": 0.487,
      "best_epoch": 42,
      "train_rows": 1624,
      "val_rows": 406,
      "validation": {
        "F1": 0.7801,
        "Precision": 0.7634,
        "Recall": 0.7923,
        "PR-AUC": 0.8512,
        "Accuracy": 0.8234,
        "MCC": 0.6234,
        "Loss": 0.1234
      }
    },
    ... (4 more folds)
  ]
}
```

### Interpreting Results

- **PR-AUC (Precision-Recall AUC)**: Main metric for imbalanced data (0-1, higher is better)
- **F1**: Balance between precision and recall (0-1, higher is better)
- **Std Dev**: Shows fold-to-fold consistency (lower = more stable model)
- **MCC (Matthews Correlation Coefficient)**: Correlation metric accounting for all classes (-1 to 1, higher is better)

## Troubleshooting

### "Checkpoint not found" error

```bash
# Make sure training completed successfully
python cross_validate_ratios.py --ratio all
```

### Training is slow (5 folds × 3 ratios = 15 training runs)

Each training takes ~5-15 min depending on max_epochs. Total time ~2-4 hours.

Option: Run in parallel terminals

```bash
# Terminal 1
python cross_validate_ratios.py --ratio 1to1

# Terminal 2
python cross_validate_ratios.py --ratio 3to1

# Terminal 3
python cross_validate_ratios.py --ratio 4to1
```

### Want fresh start

```bash
# Remove old CV results
rm -r checkpoints/ratio_experiments_cv
rm -r metrics/ratio_experiments_cv

# Re-train
python cross_validate_ratios.py --ratio all
```

## Key Differences: Single-Fold vs 5-Fold CV

| Aspect         | Single-Fold                 | 5-Fold CV                     |
| -------------- | --------------------------- | ----------------------------- |
| # Models       | 3 (1 per ratio)             | 15 (5 per ratio)              |
| Training Time  | 15-45 min                   | 2-4 hours                     |
| Robustness     | Medium                      | High                          |
| Fold Stability | N/A                         | Shows variance                |
| Use Case       | Quick baseline, exploration | Publication, final evaluation |
| Metrics        | Single value                | Mean ± Std Dev                |

## Publication Ready Results

For papers/reports, report:

```
Model trained with 5-fold cross-validation:
  ratio_1to1: PR-AUC = 0.854 ± 0.012, F1 = 0.782 ± 0.002
  ratio_3to1: PR-AUC = 0.791 ± 0.019, F1 = 0.723 ± 0.031
  ratio_4to1: PR-AUC = 0.723 ± 0.027, F1 = 0.669 ± 0.046
```

Standard deviations show model reliability and generalization capability.

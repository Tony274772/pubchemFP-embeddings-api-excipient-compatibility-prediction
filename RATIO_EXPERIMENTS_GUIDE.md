# Ratio Experiments Training Guide

## Overview

This guide explains how to train models on 3 different class ratio datasets without affecting your main model.

## Directory Structure

### Data (Input)

```
data/
├── train.csv, val.csv, test.csv  ← MAIN DATA (untouched)
├── ratio_experiments/
│   ├── ratio_1to1/    (balanced: 1:1 positive:negative)
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── ratio_3to1/    (imbalanced: 3:1 positive:negative)
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── ratio_4to1/    (imbalanced: 4:1 positive:negative)
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
```

### Checkpoints (Separate from main model)

```
checkpoints/
├── molformer/           ← MAIN MODEL (untouched)
│   └── best_model.pt
└── ratio_experiments/   ← RATIO EXPERIMENTS
    ├── ratio_1to1/
    │   └── best_model.pt
    ├── ratio_3to1/
    │   └── best_model.pt
    └── ratio_4to1/
        └── best_model.pt
```

### Metrics (Separate Results)

```
metrics/
├── molformer/           ← MAIN RESULTS (untouched)
│   └── run_metrics.json
└── ratio_experiments/   ← RATIO RESULTS
    ├── ratio_1to1/
    │   └── run_metrics.json
    ├── ratio_3to1/
    │   └── run_metrics.json
    └── ratio_4to1/
        └── run_metrics.json
```

## Usage

### 1. Train All Ratios (Recommended)

```bash
python train_ratio_experiments.py --ratio all
```

This trains models on all 3 ratios sequentially and displays a comparison summary.

### 2. Train a Single Ratio

```bash
python train_ratio_experiments.py --ratio 1to1
python train_ratio_experiments.py --ratio 3to1
python train_ratio_experiments.py --ratio 4to1
```

### 3. Skip Existing Checkpoints

If you want to re-train only missing ratios:

```bash
python train_ratio_experiments.py --ratio all --skip-existing
```

### 4. Override Max Epochs

```bash
python train_ratio_experiments.py --ratio all --max-epochs 100
```

### 5. Compare Metrics Only

To display metrics from all completed ratio experiments without training:

```bash
python train_ratio_experiments.py --compare-only
```

## Key Points

✓ **Main model is untouched**: Data in `data/` and checkpoint `checkpoints/molformer/` are never modified

✓ **Isolated checkpoints**: Each ratio has its own checkpoint directory

✓ **Isolated metrics**: Each ratio has its own metrics JSON

✓ **Reproducible**: Uses the same `Config()` and hyperparameters, only changes the input data directory

✓ **Parameterized via CLI**: `main.py` accepts `--data-dir`, `--checkpoint-dir`, and `--metrics-dir` arguments

## What Happens Under the Hood

Each call to `train_single_ratio()`:

1. Verifies the ratio data directory exists
2. Creates checkpoint and metrics directories (if needed)
3. Runs: `python main.py --data-dir <ratio_dir> --checkpoint-dir <cp_dir> --metrics-dir <m_dir>`
4. Each training uses the same config, model architecture, and hyperparameters
5. **Only the training/validation/test data differs**

## Monitoring Training

You can run multiple ratio trainings in parallel by calling the script multiple times in separate terminals:

**Terminal 1:**

```bash
python train_ratio_experiments.py --ratio 1to1
```

**Terminal 2:**

```bash
python train_ratio_experiments.py --ratio 3to1
```

**Terminal 3:**

```bash
python train_ratio_experiments.py --ratio 4to1
```

Or sequentially (one after another) with:

```bash
python train_ratio_experiments.py --ratio all
```

## Expected Output

```
============================================================
Training on ratio_1to1
  Data dir:      data/ratio_experiments/ratio_1to1
  Checkpoint:    checkpoints/ratio_experiments/ratio_1to1
  Metrics:       metrics/ratio_experiments/ratio_1to1
============================================================

[Training logs...]

✓ Successfully trained ratio_1to1

[Repeats for 3to1 and 4to1]

============================================================
RATIO EXPERIMENTS SUMMARY
============================================================
  ratio_1to1: ✓ PASSED
  ratio_3to1: ✓ PASSED
  ratio_4to1: ✓ PASSED
============================================================

============================================================
RATIO METRICS COMPARISON
============================================================

ratio_1to1:
  Val  F1=0.7823, PR-AUC=0.8541
  Test F1=0.7654, PR-AUC=0.8412

ratio_3to1:
  Val  F1=0.7234, PR-AUC=0.7912
  Test F1=0.7012, PR-AUC=0.7823

ratio_4to1:
  Val  F1=0.6891, PR-AUC=0.7234
  Test F1=0.6712, PR-AUC=0.7156
```

## Troubleshooting

**Q: "Ratio directory not found"**

- Ensure your ratio datasets are in `data/ratio_experiments/ratio_1to1/`, etc.
- Check that train.csv, val.csv, test.csv exist in each ratio directory

**Q: Main model checkpoint was modified**

- This shouldn't happen if using the script correctly
- Check that you're not passing `--checkpoint-dir checkpoints/molformer` in the script

**Q: Want to start fresh**

```bash
rm -r checkpoints/ratio_experiments
rm -r metrics/ratio_experiments
python train_ratio_experiments.py --ratio all
```

## Next Steps

1. **Train the models**: `python train_ratio_experiments.py --ratio all`
2. **Compare results**: `python train_ratio_experiments.py --compare-only`
3. **Analyze which ratio works best** for your use case
4. **Load a ratio model** in inference: Use `--checkpoint-dir checkpoints/ratio_experiments/ratio_3to1` in your inference script

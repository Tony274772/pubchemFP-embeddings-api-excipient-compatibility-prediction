"""
ASL hyperparameter sweep from ARCHITECTURE.md Section 7.

Runs the grouped train/val/test split with the current regularization fixes and
reports validation PR-AUC for each Asymmetric Loss setting. Final test evaluation
should be done separately with `python main.py` after selecting a setting.
"""

import argparse
import itertools
import json
import logging
import os
import sys

os.environ["TRITON_DISABLE"] = "1"
os.environ["TORCH_USE_TRITON"] = "0"

import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from src.config import Config
from src.dataset import get_dataloaders
from src.evaluate import evaluate_model_full, calculate_metrics_at_threshold
from src.molformer_featurization import MolFormerFeaturizer
from src.loss import AsymmetricLoss
from src.model import APIExcipientModel
from src.train import train_model
from src.utils import count_parameters, set_seed
from sklearn.metrics import auc, precision_recall_curve
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep ASL hyperparameters.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--checkpoint-root", default="checkpoints_asl")
    return parser.parse_args()


def val_pr_auc_and_metrics(model, val_loader, criterion, device, threshold_step):
    """Compute PR-AUC (threshold-free) and F1/MCC at a val-tuned threshold.
    
    Performs single forward pass to get y_true/y_prob, then:
    - Computes PR-AUC (threshold-independent)
    - Sweeps thresholds to find best F1 on validation set
    - Returns F1/MCC at that tuned threshold
    """
    # Single forward pass to get predictions
    _, _, y_true, y_prob = evaluate_model_full(model, val_loader, criterion, device)
    
    # Compute PR-AUC (threshold-free metric)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    # Sweep thresholds to find the one that maximizes F1 on val set
    thresholds = np.arange(0.001, 1.0, threshold_step)
    best_f1 = -1.0
    best_threshold = 0.5
    for thresh in thresholds:
        metrics = calculate_metrics_at_threshold(y_true, y_prob, thresh)
        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]
            best_threshold = thresh
    
    # Compute F1 and MCC at the tuned threshold
    metrics_dict = calculate_metrics_at_threshold(y_true, y_prob, best_threshold)
    f1 = metrics_dict["F1"]
    mcc = metrics_dict["MCC"]
    
    return pr_auc, f1, mcc, best_threshold


def main():
    args = parse_args()
    base_config = Config()
    if args.max_epochs is not None:
        base_config.max_epochs = args.max_epochs
    if args.early_stop_patience is not None:
        base_config.early_stop_patience = args.early_stop_patience

    device = base_config.get_device()
    logging.info("=== ASL HYPERPARAMETER SWEEP ===")
    logging.info(f"Using device: {device}")

    featurizer = MolFormerFeaturizer(
        model_path=base_config.molformer_model_path,
    ).to(device)

    results = []
    grid = itertools.product([2.0, 3.0, 4.0], [0.0, 1.0], [0.0, 0.05, 0.1])

    for gamma_neg, gamma_pos, clip in grid:
        config = Config()
        config.asl_gamma_neg = gamma_neg
        config.asl_gamma_pos = gamma_pos
        config.asl_clip = clip
        config.checkpoint_dir = os.path.join(
            args.checkpoint_root,
            f"gn{gamma_neg:g}_gp{gamma_pos:g}_clip{clip:g}".replace(".", "p"),
        )
        config.max_epochs = base_config.max_epochs
        config.early_stop_patience = base_config.early_stop_patience
        config.device = base_config.device

        set_seed(config.seed)
        logging.info(
            "Starting ASL combo: "
            f"gamma_neg={gamma_neg:g}, gamma_pos={gamma_pos:g}, clip={clip:g}"
        )

        train_loader, val_loader, _ = get_dataloaders(config, featurizer)
        model = APIExcipientModel(config).to(device)
        trainable, total = count_parameters(model)
        logging.info(f"Model params: {trainable:,} trainable / {total:,} total")

        criterion = AsymmetricLoss(
            gamma_neg=config.asl_gamma_neg,
            gamma_pos=config.asl_gamma_pos,
            clip=config.asl_clip,
        )
        best_model = train_model(config, model, train_loader, val_loader, criterion)
        pr_auc, f1, mcc, best_threshold = val_pr_auc_and_metrics(
            best_model, val_loader, criterion.to(device), device, config.threshold_step
        )

        results.append(
            {
                "gamma_neg": gamma_neg,
                "gamma_pos": gamma_pos,
                "clip": clip,
                "val_pr_auc": pr_auc,
                "val_f1": f1,
                "val_mcc": mcc,
                "threshold": best_threshold,
            }
        )
        logging.info(
            f"ASL combo result: gamma_neg={gamma_neg:g}, gamma_pos={gamma_pos:g}, "
            f"clip={clip:g}, val PR-AUC={pr_auc:.4f}, val F1={f1:.4f}, val MCC={mcc:.4f}"
        )

    results.sort(key=lambda row: row["val_pr_auc"], reverse=True)
    
    # Save results to JSON
    os.makedirs(base_config.metrics_dir, exist_ok=True)
    results_path = os.path.join(base_config.metrics_dir, "asl_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_path}")
    
    print("\n" + "=" * 80)
    print("ASL SWEEP RESULTS (ranked by validation PR-AUC)")
    print("=" * 80)
    print(f"{'rank':>4}  {'gamma_neg':>9}  {'gamma_pos':>9}  {'clip':>6}  {'val_pr_auc':>10}  {'val_f1':>8}  {'val_mcc':>8}  {'threshold':>9}")
    for rank, row in enumerate(results, start=1):
        print(
            f"{rank:>4}  {row['gamma_neg']:>9g}  {row['gamma_pos']:>9g}  "
            f"{row['clip']:>6g}  {row['val_pr_auc']:>10.4f}  {row['val_f1']:>8.4f}  {row['val_mcc']:>8.4f}  {row['threshold']:>9.3f}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

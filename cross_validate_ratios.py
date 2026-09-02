"""
5-fold cross-validation for different class ratio datasets.

Trains 5 folds for each of the 3 ratio datasets:
  - ratio_1to1 (balanced)
  - ratio_3to1 (3:1 positive:negative)
  - ratio_4to1 (4:1 positive:negative)

Each ratio gets its own checkpoint and metrics directories, keeping results isolated.
Results are aggregated and compared across ratios.
"""

import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

RATIOS = ["1to1", "3to1", "4to1"]
BASE_DATA_DIR = "data/ratio_experiments"


def run_cv_for_ratio(ratio: str, max_epochs: int = None, skip_if_exists: bool = False) -> bool:
    """
    Run 5-fold cross-validation for a single ratio dataset.
    
    Args:
        ratio: One of "1to1", "3to1", "4to1"
        max_epochs: Optional max epochs override
        skip_if_exists: If True, skip if any fold checkpoint exists
    
    Returns:
        True if CV succeeded, False otherwise
    """
    ratio_dir = os.path.join(BASE_DATA_DIR, f"ratio_{ratio}")
    checkpoint_dir = f"checkpoints/ratio_experiments_cv/ratio_{ratio}"
    metrics_dir = f"metrics/ratio_experiments_cv/ratio_{ratio}"
    
    if not os.path.exists(ratio_dir):
        logging.error(f"Ratio directory not found: {ratio_dir}")
        return False
    
    # Check if fold 1 checkpoint exists (quick check for completion)
    fold_1_checkpoint = os.path.join(checkpoint_dir, "cv_fold_1", "best_model.pt")
    if skip_if_exists and os.path.exists(fold_1_checkpoint):
        logging.info(f"5-fold CV already exists for ratio {ratio}, skipping...")
        return True
    
    # Create output directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    logging.info(f"\n{'='*70}")
    logging.info(f"5-Fold Cross-Validation on ratio_{ratio}")
    logging.info(f"  Data dir:      {ratio_dir}")
    logging.info(f"  Checkpoint:    {checkpoint_dir}")
    logging.info(f"  Metrics:       {metrics_dir}")
    logging.info(f"{'='*70}\n")
    
    # Build command
    cmd = [
        sys.executable,
        "cross_validate.py",
        "--data-dir", ratio_dir,
        "--checkpoint-dir", checkpoint_dir,
        "--metrics-dir", metrics_dir,
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        logging.info(f"\n✓ Successfully completed 5-fold CV for ratio_{ratio}\n")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"\n✗ Failed 5-fold CV for ratio_{ratio}: {e}\n")
        return False


def run_all_cv_ratios(max_epochs: int = None, skip_if_exists: bool = False):
    """Run 5-fold cross-validation for all ratio datasets."""
    results = {}
    
    for ratio in RATIOS:
        success = run_cv_for_ratio(ratio, max_epochs=max_epochs, skip_if_exists=skip_if_exists)
        results[ratio] = "✓ PASSED" if success else "✗ FAILED"
    
    # Print summary
    logging.info(f"\n{'='*70}")
    logging.info("5-FOLD CV RATIO EXPERIMENTS SUMMARY")
    logging.info(f"{'='*70}")
    for ratio, status in results.items():
        logging.info(f"  ratio_{ratio}: {status}")
    logging.info(f"{'='*70}\n")
    
    return all(v == "✓ PASSED" for v in results.values())


def load_cv_metrics(ratio: str) -> dict:
    """Load 5-fold CV metrics for a specific ratio."""
    metrics_file = f"metrics/ratio_experiments_cv/ratio_{ratio}/cv_metrics.json"
    
    if not os.path.exists(metrics_file):
        return None
    
    with open(metrics_file) as f:
        return json.load(f)


def compare_cv_results():
    """Compare 5-fold CV results across all ratio experiments."""
    logging.info(f"\n{'='*70}")
    logging.info("5-FOLD CV RATIO METRICS COMPARISON")
    logging.info(f"{'='*70}\n")
    
    all_results = {}
    
    for ratio in RATIOS:
        metrics = load_cv_metrics(ratio)
        if metrics is None:
            logging.warning(f"CV metrics not found for ratio_{ratio}")
            continue
        
        all_results[f"ratio_{ratio}"] = metrics
        
        # Extract aggregate metrics
        fold_results = metrics.get("fold_results", [])
        if fold_results:
            pr_aucs = [f["validation"]["PR-AUC"] for f in fold_results]
            f1s = [f["validation"]["F1"] for f in fold_results]
            precisions = [f["validation"]["Precision"] for f in fold_results]
            recalls = [f["validation"]["Recall"] for f in fold_results]
            
            mean_pr_auc = sum(pr_aucs) / len(pr_aucs)
            mean_f1 = sum(f1s) / len(f1s)
            mean_precision = sum(precisions) / len(precisions)
            mean_recall = sum(recalls) / len(recalls)
            
            std_pr_auc = (sum((x - mean_pr_auc) ** 2 for x in pr_aucs) / len(pr_aucs)) ** 0.5
            std_f1 = (sum((x - mean_f1) ** 2 for x in f1s) / len(f1s)) ** 0.5
            
            logging.info(f"ratio_{ratio}:")
            logging.info(f"  Mean PR-AUC:  {mean_pr_auc:.4f} ± {std_pr_auc:.4f}")
            logging.info(f"  Mean F1:      {mean_f1:.4f} ± {std_f1:.4f}")
            logging.info(f"  Mean Prec:    {mean_precision:.4f}")
            logging.info(f"  Mean Recall:  {mean_recall:.4f}")
            logging.info(f"  Folds: {len(fold_results)}")
            logging.info()
    
    logging.info(f"{'='*70}\n")
    return all_results


def print_detailed_fold_table():
    """Print detailed table with all fold results."""
    logging.info(f"\n{'='*70}")
    logging.info("DETAILED FOLD RESULTS")
    logging.info(f"{'='*70}\n")
    
    for ratio in RATIOS:
        metrics = load_cv_metrics(ratio)
        if metrics is None:
            logging.warning(f"CV metrics not found for ratio_{ratio}")
            continue
        
        fold_results = metrics.get("fold_results", [])
        
        logging.info(f"ratio_{ratio}:")
        logging.info("-" * 95)
        logging.info(f"{'Fold':<6} {'PR-AUC':<10} {'F1':<10} {'Prec':<10} {'Recall':<10} {'Acc':<10} {'MCC':<10}")
        logging.info("-" * 95)
        
        for result in fold_results:
            fold = result["fold"]
            val = result["validation"]
            logging.info(
                f"{fold:<6} "
                f"{val.get('PR-AUC', 0):<10.4f} "
                f"{val.get('F1', 0):<10.4f} "
                f"{val.get('Precision', 0):<10.4f} "
                f"{val.get('Recall', 0):<10.4f} "
                f"{val.get('Accuracy', 0):<10.4f} "
                f"{val.get('MCC', 0):<10.4f}"
            )
        
        logging.info()


def main():
    parser = argparse.ArgumentParser(
        description="Run 5-fold cross-validation on different class ratio datasets."
    )
    parser.add_argument(
        "--ratio",
        choices=["all"] + RATIOS,
        default="all",
        help="Which ratio to run CV on (default: all)"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs from config"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if CV checkpoints already exist"
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare existing CV results, don't train"
    )
    parser.add_argument(
        "--detailed-table",
        action="store_true",
        help="Print detailed fold-by-fold results"
    )
    
    args = parser.parse_args()
    
    if args.compare_only:
        compare_cv_results()
        if args.detailed_table:
            print_detailed_fold_table()
        return
    
    if args.ratio == "all":
        success = run_all_cv_ratios(
            max_epochs=args.max_epochs,
            skip_if_exists=args.skip_existing
        )
        compare_cv_results()
        if args.detailed_table:
            print_detailed_fold_table()
        sys.exit(0 if success else 1)
    else:
        success = run_cv_for_ratio(
            args.ratio,
            max_epochs=args.max_epochs,
            skip_if_exists=args.skip_existing
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

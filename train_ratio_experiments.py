"""
Train models on different class ratio datasets without affecting the main model.

Trains on:
  - ratio_1to1 (balanced)
  - ratio_3to1 (3:1 positive:negative)
  - ratio_4to1 (4:1 positive:negative)

Each ratio gets its own checkpoint and metrics directories, keeping results isolated.
"""

import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

RATIOS = ["1to1", "3to1", "4to1"]
BASE_DATA_DIR = "data/ratio_experiments"


def train_single_ratio(ratio: str, max_epochs: int = None, skip_if_exists: bool = False) -> bool:
    """
    Train a model on a single ratio dataset.
    
    Args:
        ratio: One of "1to1", "3to1", "4to1"
        max_epochs: Optional max epochs override
        skip_if_exists: If True, skip if checkpoint already exists
    
    Returns:
        True if training succeeded, False otherwise
    """
    ratio_dir = os.path.join(BASE_DATA_DIR, f"ratio_{ratio}")
    checkpoint_dir = f"checkpoints/ratio_experiments/ratio_{ratio}"
    metrics_dir = f"metrics/ratio_experiments/ratio_{ratio}"
    
    if not os.path.exists(ratio_dir):
        logging.error(f"Ratio directory not found: {ratio_dir}")
        return False
    
    if skip_if_exists and os.path.exists(os.path.join(checkpoint_dir, "best_model.pt")):
        logging.info(f"Checkpoint already exists for ratio {ratio}, skipping...")
        return True
    
    # Create output directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Training on ratio_{ratio}")
    logging.info(f"  Data dir:      {ratio_dir}")
    logging.info(f"  Checkpoint:    {checkpoint_dir}")
    logging.info(f"  Metrics:       {metrics_dir}")
    logging.info(f"{'='*60}\n")
    
    # Build command
    cmd = [
        sys.executable,
        "main.py",
        "--data-dir", ratio_dir,
        "--checkpoint-dir", checkpoint_dir,
        "--metrics-dir", metrics_dir,
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        logging.info(f"\n✓ Successfully trained ratio_{ratio}\n")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"\n✗ Failed to train ratio_{ratio}: {e}\n")
        return False


def train_all_ratios(max_epochs: int = None, skip_if_exists: bool = False):
    """Train models on all ratio datasets."""
    results = {}
    
    for ratio in RATIOS:
        success = train_single_ratio(ratio, max_epochs=max_epochs, skip_if_exists=skip_if_exists)
        results[ratio] = "✓ PASSED" if success else "✗ FAILED"
    
    # Print summary
    logging.info(f"\n{'='*60}")
    logging.info("RATIO EXPERIMENTS SUMMARY")
    logging.info(f"{'='*60}")
    for ratio, status in results.items():
        logging.info(f"  ratio_{ratio}: {status}")
    logging.info(f"{'='*60}\n")
    
    return all(v == "✓ PASSED" for v in results.values())


def compare_ratios_metrics():
    """Compare validation and test metrics across all ratio experiments."""
    import json
    
    logging.info(f"\n{'='*60}")
    logging.info("RATIO METRICS COMPARISON")
    logging.info(f"{'='*60}\n")
    
    all_metrics = {}
    
    for ratio in RATIOS:
        metrics_file = f"metrics/ratio_experiments/ratio_{ratio}/run_metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics = json.load(f)
                all_metrics[f"ratio_{ratio}"] = metrics
                
                val = metrics.get("validation", {})
                test = metrics.get("test", {})
                
                logging.info(f"ratio_{ratio}:")
                logging.info(f"  Val  F1={val.get('F1', 'N/A'):.4f}, PR-AUC={val.get('PR-AUC', 'N/A'):.4f}")
                logging.info(f"  Test F1={test.get('F1', 'N/A'):.4f}, PR-AUC={test.get('PR-AUC', 'N/A'):.4f}")
                logging.info()
        else:
            logging.warning(f"Metrics file not found for ratio_{ratio}: {metrics_file}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train models on different class ratio datasets."
    )
    parser.add_argument(
        "--ratio",
        choices=["all"] + RATIOS,
        default="all",
        help="Which ratio to train on (default: all)"
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
        help="Skip if checkpoint already exists"
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare existing metrics, don't train"
    )
    
    args = parser.parse_args()
    
    if args.compare_only:
        compare_ratios_metrics()
        return
    
    if args.ratio == "all":
        success = train_all_ratios(
            max_epochs=args.max_epochs,
            skip_if_exists=args.skip_existing
        )
        compare_ratios_metrics()
        sys.exit(0 if success else 1)
    else:
        success = train_single_ratio(
            args.ratio,
            max_epochs=args.max_epochs,
            skip_if_exists=args.skip_existing
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

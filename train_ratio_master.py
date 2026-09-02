"""
Master script for training ratio datasets with both single-fold and 5-fold cross-validation.

Usage:
  python train_ratio_master.py --mode single     # Train 1 fold for each ratio
  python train_ratio_master.py --mode cv         # Train 5 folds for each ratio
  python train_ratio_master.py --mode both       # Train both (single + CV)
  python train_ratio_master.py --compare-all     # Compare all results
"""

import argparse
import logging
import subprocess
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def run_single_fold_training(skip_existing: bool = False):
    """Run single-fold training on all 3 ratios."""
    logging.info("\n" + "="*70)
    logging.info("STARTING SINGLE-FOLD TRAINING")
    logging.info("="*70)
    
    cmd = [sys.executable, "train_ratio_experiments.py", "--ratio", "all"]
    if skip_existing:
        cmd.append("--skip-existing")
    
    try:
        subprocess.run(cmd, check=True)
        logging.info("✓ Single-fold training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ Single-fold training failed: {e}")
        return False


def run_cv_fold_training(skip_existing: bool = False):
    """Run 5-fold cross-validation training on all 3 ratios."""
    logging.info("\n" + "="*70)
    logging.info("STARTING 5-FOLD CROSS-VALIDATION TRAINING")
    logging.info("="*70)
    
    cmd = [sys.executable, "cross_validate_ratios.py", "--ratio", "all"]
    if skip_existing:
        cmd.append("--skip-existing")
    
    try:
        subprocess.run(cmd, check=True)
        logging.info("✓ 5-fold CV training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ 5-fold CV training failed: {e}")
        return False


def compare_single_fold_results():
    """Compare single-fold training results."""
    logging.info("\n" + "="*70)
    logging.info("SINGLE-FOLD RESULTS COMPARISON")
    logging.info("="*70 + "\n")
    
    cmd = [sys.executable, "train_ratio_experiments.py", "--compare-only"]
    subprocess.run(cmd)


def compare_cv_results():
    """Compare 5-fold CV results."""
    logging.info("\n" + "="*70)
    logging.info("5-FOLD CV RESULTS COMPARISON")
    logging.info("="*70 + "\n")
    
    cmd = [sys.executable, "cross_validate_ratios.py", "--compare-only", "--detailed-table"]
    subprocess.run(cmd)


def print_fold_stability():
    """Print fold stability analysis."""
    logging.info("\n" + "="*70)
    logging.info("FOLD STABILITY ANALYSIS")
    logging.info("="*70 + "\n")
    
    cmd = [sys.executable, "ratio_cv_utils.py", "--stability"]
    subprocess.run(cmd)


def compare_all_results():
    """Compare all available results (single-fold and CV)."""
    compare_single_fold_results()
    compare_cv_results()
    print_fold_stability()


def print_workflow_guide():
    """Print a guide for using the training scripts."""
    guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RATIO EXPERIMENTS TRAINING WORKFLOW                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

This system provides two training approaches:

┌─ SINGLE-FOLD TRAINING ─────────────────────────────────────────────────────┐
│                                                                              │
│ Train 1 model per ratio (fast baseline)                                    │
│                                                                              │
│   python train_ratio_experiments.py --ratio all                            │
│                                                                              │
│ Results in:                                                                 │
│   - checkpoints/ratio_experiments/ratio_*/best_model.pt                    │
│   - metrics/ratio_experiments/ratio_*/run_metrics.json                     │
│                                                                              │
│ Use for: Quick comparison between ratios, hyperparameter tuning            │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘

┌─ 5-FOLD CROSS-VALIDATION ──────────────────────────────────────────────────┐
│                                                                              │
│ Train 5 folds per ratio (robust evaluation)                                │
│                                                                              │
│   python cross_validate_ratios.py --ratio all                              │
│                                                                              │
│ Results in:                                                                 │
│   - checkpoints/ratio_experiments_cv/ratio_*/cv_fold_*/best_model.pt       │
│   - metrics/ratio_experiments_cv/ratio_*/cv_metrics.json                   │
│                                                                              │
│ Use for: Publication results, model validation, fold stability analysis    │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘

COMMAND REFERENCE:

  Single-Fold Training:
    python train_ratio_experiments.py --ratio all              # All ratios
    python train_ratio_experiments.py --ratio 1to1             # Single ratio
    python train_ratio_experiments.py --compare-only           # Compare results
    python ratio_utils.py --compare                            # Detailed metrics

  5-Fold Cross-Validation:
    python cross_validate_ratios.py --ratio all                # All ratios
    python cross_validate_ratios.py --ratio 1to1               # Single ratio
    python cross_validate_ratios.py --compare-only             # Compare results
    python cross_validate_ratios.py --compare-only --detailed-table
    python ratio_cv_utils.py --stability                       # Fold stability

  Master Script (this script):
    python train_ratio_master.py --mode single                 # Single-fold only
    python train_ratio_master.py --mode cv                     # 5-fold CV only
    python train_ratio_master.py --mode both                   # Both approaches
    python train_ratio_master.py --compare-all                 # Show all results

RECOMMENDED WORKFLOW:

  1. Start with single-fold for quick baseline:
     python train_ratio_experiments.py --ratio all

  2. Compare single-fold results:
     python train_ratio_experiments.py --compare-only

  3. Run 5-fold CV for robust evaluation:
     python cross_validate_ratios.py --ratio all

  4. Analyze fold stability:
     python cross_validate_ratios.py --compare-only --detailed-table
     python ratio_cv_utils.py --stability

  5. Final comparison:
     python train_ratio_master.py --compare-all

NOTES:
  • Single-fold and 5-fold CV results are stored in separate directories
  • Single-fold: checkpoints/ratio_experiments/
  • 5-fold CV:   checkpoints/ratio_experiments_cv/
  • Both can coexist without interfering
  • Use --skip-existing to only train missing models

"""
    print(guide)


def main():
    parser = argparse.ArgumentParser(
        description="Master script for training ratio datasets (single-fold or 5-fold CV)"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "cv", "both"],
        default="single",
        help="Training mode: single-fold, 5-fold CV, or both (default: single)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if checkpoints already exist"
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Only compare existing results, don't train"
    )
    parser.add_argument(
        "--guide",
        action="store_true",
        help="Print workflow guide"
    )
    
    args = parser.parse_args()
    
    if args.guide:
        print_workflow_guide()
        return
    
    if args.compare_all:
        compare_all_results()
        return
    
    if args.mode == "single":
        success = run_single_fold_training(skip_existing=args.skip_existing)
        compare_single_fold_results()
        sys.exit(0 if success else 1)
    
    elif args.mode == "cv":
        success = run_cv_fold_training(skip_existing=args.skip_existing)
        compare_cv_results()
        print_fold_stability()
        sys.exit(0 if success else 1)
    
    elif args.mode == "both":
        logging.info("\n" + "="*70)
        logging.info("RUNNING BOTH SINGLE-FOLD AND 5-FOLD CV")
        logging.info("="*70 + "\n")
        
        success_single = run_single_fold_training(skip_existing=args.skip_existing)
        success_cv = run_cv_fold_training(skip_existing=args.skip_existing)
        
        if success_single or success_cv:
            compare_all_results()
        
        sys.exit(0 if (success_single and success_cv) else 1)


if __name__ == "__main__":
    main()

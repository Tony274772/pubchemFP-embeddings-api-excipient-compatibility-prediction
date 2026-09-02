"""
Utilities for loading and analyzing models trained with 5-fold cross-validation on ratio datasets.

Provides functions to:
- Load a specific fold model from a specific ratio
- Compare fold performance within a ratio
- Analyze consistency across folds
- Visualize fold stability
"""

import json
import os
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RatioCVModelLoader:
    """Load and manage models from 5-fold CV on ratio datasets."""
    
    RATIOS = ["1to1", "3to1", "4to1"]
    N_FOLDS = 5
    
    @staticmethod
    def get_fold_checkpoint_path(ratio: str, fold: int) -> str:
        """Get the checkpoint path for a specific ratio and fold."""
        if ratio not in RatioCVModelLoader.RATIOS:
            raise ValueError(f"Unknown ratio: {ratio}. Must be one of {RatioCVModelLoader.RATIOS}")
        if fold < 1 or fold > RatioCVModelLoader.N_FOLDS:
            raise ValueError(f"Fold must be 1-{RatioCVModelLoader.N_FOLDS}")
        
        checkpoint_dir = f"checkpoints/ratio_experiments_cv/ratio_{ratio}/cv_fold_{fold}"
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train the model first.")
        
        return checkpoint_path
    
    @staticmethod
    def load_fold_model(ratio: str, fold: int, device: str = "auto") -> torch.nn.Module:
        """
        Load a model from a specific ratio and fold.
        
        Args:
            ratio: One of "1to1", "3to1", "4to1"
            fold: Fold number 1-5
            device: "auto" (cuda if available), "cpu", or specific device
        
        Returns:
            Loaded model on the specified device
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint_path = RatioCVModelLoader.get_fold_checkpoint_path(ratio, fold)
        model = torch.load(checkpoint_path, map_location=device)
        model.eval()
        
        logger.info(f"Loaded ratio_{ratio} fold_{fold} model from {checkpoint_path} on device {device}")
        return model
    
    @staticmethod
    def load_all_folds_for_ratio(ratio: str, device: str = "auto") -> Dict[int, torch.nn.Module]:
        """Load all 5 fold models for a specific ratio."""
        models = {}
        for fold in range(1, RatioCVModelLoader.N_FOLDS + 1):
            try:
                models[fold] = RatioCVModelLoader.load_fold_model(ratio, fold, device)
            except FileNotFoundError:
                logger.warning(f"Model for ratio_{ratio} fold_{fold} not found, skipping")
        return models


def load_cv_metrics(ratio: str) -> Dict:
    """Load 5-fold CV metrics for a specific ratio."""
    metrics_file = f"metrics/ratio_experiments_cv/ratio_{ratio}/cv_metrics.json"
    
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"CV metrics file not found: {metrics_file}")
    
    with open(metrics_file) as f:
        return json.load(f)


def compare_all_ratios_cv() -> Dict:
    """Load and compare CV metrics for all ratio experiments."""
    comparison = {}
    
    for ratio in ["1to1", "3to1", "4to1"]:
        try:
            metrics = load_cv_metrics(ratio)
            comparison[f"ratio_{ratio}"] = metrics
        except FileNotFoundError:
            logger.warning(f"CV metrics for ratio_{ratio} not found")
    
    return comparison


def calculate_fold_statistics(fold_results: List[Dict]) -> Dict:
    """Calculate mean and std for fold results."""
    if not fold_results:
        return {}
    
    metrics_to_aggregate = ["PR-AUC", "F1", "Precision", "Recall", "Accuracy", "MCC"]
    stats = {}
    
    for metric in metrics_to_aggregate:
        values = [f["validation"].get(metric, 0) for f in fold_results]
        if values:
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std = variance ** 0.5
            stats[metric] = {
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
                "values": values
            }
    
    return stats


def print_cv_summary_table():
    """Print a summary table of CV results for all ratios."""
    try:
        comparison = compare_all_ratios_cv()
    except Exception as e:
        logger.error(f"Failed to load CV metrics: {e}")
        return
    
    if not comparison:
        logger.warning("No CV metrics found. Run 5-fold CV training first.")
        return
    
    print("\n" + "="*100)
    print("5-FOLD CV SUMMARY - ALL RATIOS")
    print("="*100)
    print(f"{'Ratio':<15} {'PR-AUC':<20} {'F1':<20} {'Precision':<20} {'Recall':<20}")
    print("-"*100)
    
    for name in sorted(comparison.keys()):
        metrics = comparison[name]
        fold_results = metrics.get("fold_results", [])
        stats = calculate_fold_statistics(fold_results)
        
        pr_auc_str = f"{stats.get('PR-AUC', {}).get('mean', 0):.4f} ± {stats.get('PR-AUC', {}).get('std', 0):.4f}"
        f1_str = f"{stats.get('F1', {}).get('mean', 0):.4f} ± {stats.get('F1', {}).get('std', 0):.4f}"
        prec_str = f"{stats.get('Precision', {}).get('mean', 0):.4f} ± {stats.get('Precision', {}).get('std', 0):.4f}"
        recall_str = f"{stats.get('Recall', {}).get('mean', 0):.4f} ± {stats.get('Recall', {}).get('std', 0):.4f}"
        
        print(f"{name:<15} {pr_auc_str:<20} {f1_str:<20} {prec_str:<20} {recall_str:<20}")
    
    print("="*100 + "\n")


def print_fold_details(ratio: str):
    """Print detailed fold-by-fold results for a specific ratio."""
    try:
        metrics = load_cv_metrics(ratio)
    except FileNotFoundError:
        logger.error(f"CV metrics for ratio_{ratio} not found")
        return
    
    fold_results = metrics.get("fold_results", [])
    stats = calculate_fold_statistics(fold_results)
    
    print("\n" + "="*110)
    print(f"5-FOLD CV DETAILS - ratio_{ratio}")
    print("="*110)
    print(f"{'Fold':<6} {'PR-AUC':<12} {'F1':<12} {'Prec':<12} {'Recall':<12} {'Acc':<12} {'MCC':<12} {'Epoch':<6}")
    print("-"*110)
    
    for result in fold_results:
        fold = result["fold"]
        val = result["validation"]
        epoch = result.get("best_epoch", "N/A")
        print(
            f"{fold:<6} "
            f"{val.get('PR-AUC', 0):<12.4f} "
            f"{val.get('F1', 0):<12.4f} "
            f"{val.get('Precision', 0):<12.4f} "
            f"{val.get('Recall', 0):<12.4f} "
            f"{val.get('Accuracy', 0):<12.4f} "
            f"{val.get('MCC', 0):<12.4f} "
            f"{str(epoch):<6}"
        )
    
    print("-"*110)
    print(f"{'MEAN':<6} ", end="")
    for metric in ["PR-AUC", "F1", "Precision", "Recall", "Accuracy", "MCC"]:
        mean = stats.get(metric, {}).get('mean', 0)
        std = stats.get(metric, {}).get('std', 0)
        print(f"{mean:.4f}±{std:.3f}  ", end="")
    print()
    print("="*110 + "\n")


def compare_fold_stability():
    """Compare fold stability (consistency) across ratios."""
    comparison = compare_all_ratios_cv()
    
    if not comparison:
        logger.warning("No CV metrics found.")
        return
    
    print("\n" + "="*80)
    print("FOLD STABILITY ANALYSIS (Lower std = More stable)")
    print("="*80)
    print(f"{'Ratio':<15} {'PR-AUC Std':<15} {'F1 Std':<15} {'Recall Std':<15} {'Avg Std':<15}")
    print("-"*80)
    
    for name in sorted(comparison.keys()):
        metrics = comparison[name]
        fold_results = metrics.get("fold_results", [])
        stats = calculate_fold_statistics(fold_results)
        
        pr_auc_std = stats.get('PR-AUC', {}).get('std', 0)
        f1_std = stats.get('F1', {}).get('std', 0)
        recall_std = stats.get('Recall', {}).get('std', 0)
        avg_std = (pr_auc_std + f1_std + recall_std) / 3
        
        print(f"{name:<15} {pr_auc_std:<15.4f} {f1_std:<15.4f} {recall_std:<15.4f} {avg_std:<15.4f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze 5-fold CV ratio experiments")
    parser.add_argument("--summary", action="store_true", help="Show summary table")
    parser.add_argument("--ratio", type=str, help="Show details for specific ratio (1to1, 3to1, 4to1)")
    parser.add_argument("--stability", action="store_true", help="Compare fold stability across ratios")
    
    args = parser.parse_args()
    
    if args.summary or (not args.ratio and not args.stability):
        print_cv_summary_table()
    
    if args.ratio:
        print_fold_details(args.ratio)
    
    if args.stability:
        compare_fold_stability()

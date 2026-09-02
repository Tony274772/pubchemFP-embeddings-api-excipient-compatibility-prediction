"""
Utilities for loading and comparing models trained on different class ratios.

Provides functions to:
- Load a specific ratio model for inference
- Compare predictions across ratios
- Visualize metrics differences
"""

import json
import os
import torch
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RatioModelLoader:
    """Load and manage models trained on different ratios."""
    
    RATIO_CONFIGS = {
        "1to1": {"checkpoint_dir": "checkpoints/ratio_experiments/ratio_1to1"},
        "3to1": {"checkpoint_dir": "checkpoints/ratio_experiments/ratio_3to1"},
        "4to1": {"checkpoint_dir": "checkpoints/ratio_experiments/ratio_4to1"},
    }
    
    @staticmethod
    def get_checkpoint_path(ratio: str) -> str:
        """Get the checkpoint path for a ratio."""
        if ratio not in RatioModelLoader.RATIO_CONFIGS:
            raise ValueError(f"Unknown ratio: {ratio}. Must be one of {list(RatioModelLoader.RATIO_CONFIGS.keys())}")
        
        checkpoint_dir = RatioModelLoader.RATIO_CONFIGS[ratio]["checkpoint_dir"]
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train the model first.")
        
        return checkpoint_path
    
    @staticmethod
    def load_model(ratio: str, device: str = "auto") -> torch.nn.Module:
        """
        Load a model trained on the specified ratio.
        
        Args:
            ratio: One of "1to1", "3to1", "4to1"
            device: "auto" (cuda if available), "cpu", or specific device
        
        Returns:
            Loaded model on the specified device
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint_path = RatioModelLoader.get_checkpoint_path(ratio)
        model = torch.load(checkpoint_path, map_location=device)
        model.eval()
        
        logger.info(f"Loaded ratio_{ratio} model from {checkpoint_path} on device {device}")
        return model
    
    @staticmethod
    def load_all_models(device: str = "auto") -> Dict[str, torch.nn.Module]:
        """Load all available ratio models."""
        models = {}
        for ratio in RatioModelLoader.RATIO_CONFIGS.keys():
            try:
                models[ratio] = RatioModelLoader.load_model(ratio, device)
            except FileNotFoundError:
                logger.warning(f"Model for ratio_{ratio} not found, skipping")
        return models


def load_ratio_metrics(ratio: str) -> Dict:
    """Load metrics JSON for a specific ratio."""
    metrics_file = f"metrics/ratio_experiments/ratio_{ratio}/run_metrics.json"
    
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    with open(metrics_file) as f:
        return json.load(f)


def compare_all_ratios() -> Dict:
    """Load and compare metrics for all ratio experiments."""
    ratios = ["1to1", "3to1", "4to1"]
    comparison = {}
    
    for ratio in ratios:
        try:
            metrics = load_ratio_metrics(ratio)
            comparison[f"ratio_{ratio}"] = metrics
        except FileNotFoundError:
            logger.warning(f"Metrics for ratio_{ratio} not found")
    
    return comparison


def print_comparison_table():
    """Print a comparison table of all ratio metrics."""
    try:
        comparison = compare_all_ratios()
    except Exception as e:
        logger.error(f"Failed to load metrics: {e}")
        return
    
    if not comparison:
        logger.warning("No metrics found. Run training first.")
        return
    
    print("\n" + "="*80)
    print("RATIO EXPERIMENTS - METRICS COMPARISON")
    print("="*80)
    
    # Extract val and test metrics
    print("\nVALIDATION METRICS:")
    print("-" * 80)
    print(f"{'Ratio':<15} {'F1':<12} {'Precision':<12} {'Recall':<12} {'PR-AUC':<12} {'Threshold':<12}")
    print("-" * 80)
    
    for name, metrics in sorted(comparison.items()):
        val = metrics.get("validation", {})
        threshold = metrics.get("best_threshold", "N/A")
        print(
            f"{name:<15} "
            f"{val.get('F1', 'N/A'):>10.4f}  "
            f"{val.get('Precision', 'N/A'):>10.4f}  "
            f"{val.get('Recall', 'N/A'):>10.4f}  "
            f"{val.get('PR-AUC', 'N/A'):>10.4f}  "
            f"{threshold:>10.4f}"
        )
    
    print("\nTEST METRICS:")
    print("-" * 80)
    print(f"{'Ratio':<15} {'F1':<12} {'Precision':<12} {'Recall':<12} {'PR-AUC':<12} {'Loss':<12}")
    print("-" * 80)
    
    for name, metrics in sorted(comparison.items()):
        test = metrics.get("test", {})
        print(
            f"{name:<15} "
            f"{test.get('F1', 'N/A'):>10.4f}  "
            f"{test.get('Precision', 'N/A'):>10.4f}  "
            f"{test.get('Recall', 'N/A'):>10.4f}  "
            f"{test.get('PR-AUC', 'N/A'):>10.4f}  "
            f"{test.get('Loss', 'N/A'):>10.4f}"
        )
    
    print("="*80 + "\n")


def get_best_ratio(metric: str = "F1") -> Tuple[str, float]:
    """
    Get the best performing ratio based on a test metric.
    
    Args:
        metric: One of "F1", "Precision", "Recall", "PR-AUC"
    
    Returns:
        (ratio_name, metric_value)
    """
    comparison = compare_all_ratios()
    
    best_ratio = None
    best_value = -float('inf')
    
    for name, metrics in comparison.items():
        test = metrics.get("test", {})
        value = test.get(metric, -float('inf'))
        
        if value > best_value:
            best_value = value
            best_ratio = name
    
    return (best_ratio, best_value) if best_ratio else (None, -1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare ratio experiments")
    parser.add_argument("--compare", action="store_true", help="Show comparison table")
    parser.add_argument("--best-f1", action="store_true", help="Show best ratio by F1")
    parser.add_argument("--best-pr-auc", action="store_true", help="Show best ratio by PR-AUC")
    
    args = parser.parse_args()
    
    if args.compare or (not args.best_f1 and not args.best_pr_auc):
        print_comparison_table()
    
    if args.best_f1:
        ratio, f1 = get_best_ratio("F1")
        print(f"Best ratio by F1: {ratio} (F1={f1:.4f})")
    
    if args.best_pr_auc:
        ratio, pr_auc = get_best_ratio("PR-AUC")
        print(f"Best ratio by PR-AUC: {ratio} (PR-AUC={pr_auc:.4f})")

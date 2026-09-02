"""
Compare 6 key metrics across:
1. Main data CV (aggregated)
2. CV aggregated for 3 class ratios

Metrics: PR-AUC, F1, Precision, Recall, Accuracy, MCC
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

def load_main_cv_metrics() -> Dict:
    """Load main data CV metrics (aggregated across folds)."""
    metrics_file = "metrics/molformer/cv_metrics.json"
    
    if not os.path.exists(metrics_file):
        return None
    
    with open(metrics_file) as f:
        data = json.load(f)
    
    folds = data.get("folds", [])
    
    # Aggregate metrics across folds
    aggregated = {}
    metric_names = ["PR-AUC", "F1", "Precision", "Recall", "Accuracy", "MCC"]
    
    for metric in metric_names:
        values = [fold["validation"].get(metric, 0) for fold in folds]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        
        aggregated[metric] = {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values)
        }
    
    return aggregated


def load_cv_metrics(ratio: str) -> Dict:
    """Load aggregated CV metrics for a ratio."""
    metrics_file = f"metrics/ratio_experiments_cv/ratio_{ratio}/cv_metrics.json"
    
    if not os.path.exists(metrics_file):
        return None
    
    with open(metrics_file) as f:
        data = json.load(f)
    
    folds = data.get("folds", [])
    
    # Aggregate metrics across folds
    aggregated = {}
    metric_names = ["PR-AUC", "F1", "Precision", "Recall", "Accuracy", "MCC"]
    
    for metric in metric_names:
        values = [fold["validation"].get(metric, 0) for fold in folds]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        
        aggregated[metric] = {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values)
        }
    
    return aggregated


def print_comparison_table():
    """Print side-by-side comparison of 6 metrics."""
    main_cv = load_main_cv_metrics()
    cv_ratios = {
        "ratio_1to1": load_cv_metrics("1to1"),
        "ratio_3to1": load_cv_metrics("3to1"),
        "ratio_4to1": load_cv_metrics("4to1")
    }
    
    if not main_cv or not all(cv_ratios.values()):
        print("Error: Could not load all metrics")
        return
    
    # Extract 6 key metrics
    metrics_to_show = ["PR-AUC", "F1", "Precision", "Recall", "Accuracy", "MCC"]
    
    print("\n" + "="*180)
    print("6-METRIC COMPARISON: MAIN CV vs RATIO CVs (Mean ± Std)")
    print("="*180)
    
    # Header
    print(f"{'Dataset':<20}", end="")
    for metric in metrics_to_show:
        print(f"{metric:<26}", end="")
    print()
    print("-"*180)
    
    # Main CV
    print(f"{'Main (CV)':<20}", end="")
    for metric in metrics_to_show:
        mean = main_cv[metric]["mean"]
        std = main_cv[metric]["std"]
        print(f"{mean:.4f} ± {std:.4f}        ", end="")
    print()
    
    print("-"*180)
    
    # CV Ratios
    for ratio_name, metrics in cv_ratios.items():
        print(f"{ratio_name:<20}", end="")
        for metric in metrics_to_show:
            mean = metrics[metric]["mean"]
            std = metrics[metric]["std"]
            print(f"{mean:.4f} ± {std:.4f}        ", end="")
        print()
    
    print("="*180 + "\n")


def print_detailed_comparison():
    """Print detailed comparison with min/max range."""
    main_cv = load_main_cv_metrics()
    cv_ratios = {
        "ratio_1to1": load_cv_metrics("1to1"),
        "ratio_3to1": load_cv_metrics("3to1"),
        "ratio_4to1": load_cv_metrics("4to1")
    }
    
    if not main_cv or not all(cv_ratios.values()):
        print("Error: Could not load all metrics")
        return
    
    metrics_to_show = ["PR-AUC", "F1", "Precision", "Recall", "Accuracy", "MCC"]
    
    print("\n" + "="*160)
    print("DETAILED COMPARISON: Mean ± Std (Min-Max)")
    print("="*160)
    
    # Main CV
    print("\nMAIN CV:")
    print("-"*160)
    for metric in metrics_to_show:
        mean = main_cv[metric]["mean"]
        std = main_cv[metric]["std"]
        min_val = main_cv[metric]["min"]
        max_val = main_cv[metric]["max"]
        print(f"  {metric:<15}: {mean:.4f} ± {std:.4f}  (range: {min_val:.4f} - {max_val:.4f})")
    
    print("\nRATIO CVs:")
    print("-"*160)
    
    for ratio_name, metrics in cv_ratios.items():
        print(f"\n{ratio_name}:")
        for metric in metrics_to_show:
            mean = metrics[metric]["mean"]
            std = metrics[metric]["std"]
            min_val = metrics[metric]["min"]
            max_val = metrics[metric]["max"]
            print(f"  {metric:<15}: {mean:.4f} ± {std:.4f}  (range: {min_val:.4f} - {max_val:.4f})")
    
    print("\n" + "="*160 + "\n")


def print_metric_ranking():
    """Rank datasets by each metric."""
    main_cv = load_main_cv_metrics()
    cv_ratios = {
        "ratio_1to1": load_cv_metrics("1to1"),
        "ratio_3to1": load_cv_metrics("3to1"),
        "ratio_4to1": load_cv_metrics("4to1")
    }
    
    if not main_cv or not all(cv_ratios.values()):
        print("Error: Could not load all metrics")
        return
    
    metrics_to_show = ["PR-AUC", "F1", "Precision", "Recall", "Accuracy", "MCC"]
    
    print("\n" + "="*110)
    print("METRIC RANKINGS (Higher is Better)")
    print("="*110)
    
    for metric in metrics_to_show:
        print(f"\n{metric}:")
        print("-"*110)
        
        # Collect all values
        scores = []
        
        # Main CV
        mean = main_cv[metric]["mean"]
        scores.append(("Main (CV)", mean))
        
        # CV ratios (use mean)
        for ratio_name, metrics in cv_ratios.items():
            mean = metrics[metric]["mean"]
            scores.append((ratio_name, mean))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (name, score) in enumerate(scores, 1):
            print(f"  {rank}. {name:<20} {score:.4f}")
    
    print("\n" + "="*110 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare 4 metrics across datasets")
    parser.add_argument("--detailed", action="store_true", help="Show detailed comparison with ranges")
    parser.add_argument("--ranking", action="store_true", help="Show metric rankings")
    parser.add_argument("--all", action="store_true", help="Show all comparisons")
    
    args = parser.parse_args()
    
    if args.all or (not args.detailed and not args.ranking):
        print_comparison_table()
    
    if args.detailed or args.all:
        print_detailed_comparison()
    
    if args.ranking or args.all:
        print_metric_ranking()

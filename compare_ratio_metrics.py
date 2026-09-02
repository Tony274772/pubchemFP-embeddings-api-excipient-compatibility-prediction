"""
compare_ratio_metrics.py
========================
Compares single-run validation and test metrics across:
  - 1:1 ratio
  - 3:1 ratio
  - 4:1 ratio
  - Base / Original (~10.6:1 natural ratio)

Metrics evaluated:
  PR-AUC, F1, Precision, Recall, Accuracy, MCC, Loss, Optimal Threshold
"""

import json
import os
import pandas as pd

METRIC_DIRS = {
    "1:1": "metrics/ratio_1_1",
    "3:1": "metrics/ratio_3_1",
    "4:1": "metrics/ratio_4_1",
    "Base (Original)": "metrics/ratio_original",
}


def load_run_metrics():
    results = []
    for name, dir_path in METRIC_DIRS.items():
        json_path = os.path.join(dir_path, "run_metrics.json")
        if not os.path.exists(json_path):
            # Fall back to default metrics dir if ratio_original is not set yet
            if name == "Base (Original)" and os.path.exists("metrics/molformer/run_metrics.json"):
                json_path = "metrics/molformer/run_metrics.json"
            else:
                continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        thresh = data.get("threshold", 0.5)
        best_epoch = data.get("best_epoch", None)
        val = data.get("validation", {})
        test = data.get("test", {})

        results.append({
            "Ratio": name,
            "Best Epoch": best_epoch,
            "Val Thresh": thresh,
            "Val PR-AUC": val.get("PR-AUC", 0.0),
            "Val F1": val.get("F1", 0.0),
            "Val Prec": val.get("Precision", 0.0),
            "Val Rec": val.get("Recall", 0.0),
            "Val MCC": val.get("MCC", 0.0),
            "Test PR-AUC": test.get("PR-AUC", 0.0),
            "Test F1": test.get("F1", 0.0),
            "Test Prec": test.get("Precision", 0.0),
            "Test Rec": test.get("Recall", 0.0),
            "Test Acc": test.get("Accuracy", 0.0),
            "Test MCC": test.get("MCC", 0.0),
        })

    return results


def print_comparison():
    results = load_run_metrics()
    if not results:
        print("No ratio metrics found. Please train models first.")
        return

    df = pd.DataFrame(results)

    print("\n" + "=" * 110)
    print("SINGLE-RUN RATIO EXPERIMENT METRICS COMPARISON (1:1, 3:1, 4:1 vs Base)")
    print("=" * 110)
    
    # Format table for clean display
    display_cols = [
        "Ratio", "Best Epoch", "Val Thresh",
        "Val PR-AUC", "Val F1", "Val MCC",
        "Test PR-AUC", "Test F1", "Test Prec", "Test Rec", "Test Acc", "Test MCC"
    ]
    
    formatted_df = df[display_cols].copy()
    for col in ["Val PR-AUC", "Val F1", "Val MCC", "Test PR-AUC", "Test F1", "Test Prec", "Test Rec", "Test Acc", "Test MCC"]:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
    formatted_df["Val Thresh"] = formatted_df["Val Thresh"].apply(lambda x: f"{x:.3f}")

    print(formatted_df.to_string(index=False))
    print("=" * 110 + "\n")

    # Determine best model based on Test PR-AUC & Test F1
    best_prauc_row = df.loc[df["Test PR-AUC"].idxmax()]
    best_f1_row = df.loc[df["Test F1"].idxmax()]

    print("BEST PERFORMING MODELS:")
    print(f"  Best Test PR-AUC: {best_prauc_row['Ratio']} (PR-AUC = {best_prauc_row['Test PR-AUC']:.4f})")
    print(f"  Best Test F1:     {best_f1_row['Ratio']} (F1 = {best_f1_row['Test F1']:.4f}, Val Thresh = {best_f1_row['Val Thresh']:.3f})")
    print("-" * 110 + "\n")


if __name__ == "__main__":
    print_comparison()

"""
run_ratio_experiments.py
========================
Runs ratio experiments (1:1, 3:1, 4:1, and original baseline) by:
1. Generating resampled training datasets from data/train.csv while keeping data/val.csv and data/test.csv untouched.
2. Training models on each ratio split using main.py with fixed val and test datasets.
3. Tuning the decision threshold on the val dataset for each run (same method as main model).
4. Evaluating on the test dataset at the val-tuned threshold.
5. Printing a complete comparison table and diagnostic summary.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

DATA_DIR = "data"
RATIO_SPLITS_DIR = "data/ratio_splits"
RUNS = [
    {"name": "1:1", "key": "1_1", "train": os.path.join(RATIO_SPLITS_DIR, "train_1_1.csv")},
    {"name": "3:1", "key": "3_1", "train": os.path.join(RATIO_SPLITS_DIR, "train_3_1.csv")},
    {"name": "4:1", "key": "4_1", "train": os.path.join(RATIO_SPLITS_DIR, "train_4_1.csv")},
    {"name": "Original (~10.6:1)", "key": "original", "train": os.path.join(DATA_DIR, "train.csv")},
]


def generate_ratio_splits():
    logging.info("=== STEP 1: Generating Ratio Training Splits ===")
    cmd = [
        sys.executable,
        os.path.join("data", "ratio_experiments", "make_ratio_splits.py"),
        "--train_path", os.path.join(DATA_DIR, "train.csv"),
        "--val_path", os.path.join(DATA_DIR, "val.csv"),
        "--test_path", os.path.join(DATA_DIR, "test.csv"),
        "--out_dir", RATIO_SPLITS_DIR,
    ]
    subprocess.run(cmd, check=True)


def run_experiments():
    val_csv = os.path.join(DATA_DIR, "val.csv")
    test_csv = os.path.join(DATA_DIR, "test.csv")
    
    logging.info("\n=== STEP 2: Training Models & Tuning Thresholds ===")
    for run in RUNS:
        key = run["key"]
        train_csv = run["train"]
        checkpoint_dir = f"checkpoints/ratio_{key}"
        metrics_dir = f"metrics/ratio_{key}"
        
        logging.info(f"\n{'='*70}")
        logging.info(f"RUNNING EXPERIMENT: {run['name']}")
        logging.info(f"  Train CSV:      {train_csv}")
        logging.info(f"  Val CSV:        {val_csv} (fixed main val)")
        logging.info(f"  Test CSV:       {test_csv} (fixed main test)")
        logging.info(f"  Checkpoint Dir: {checkpoint_dir}")
        logging.info(f"  Metrics Dir:    {metrics_dir}")
        logging.info(f"{'='*70}")
        
        cmd = [
            sys.executable,
            "main.py",
            "--train-csv", train_csv,
            "--val-csv", val_csv,
            "--test-csv", test_csv,
            "--checkpoint-dir", checkpoint_dir,
            "--metrics-dir", metrics_dir,
        ]
        
        subprocess.run(cmd, check=True)


def print_comparison_table():
    logging.info("\n=== STEP 3: Results Summary & Diagnostic Analysis ===")
    results = []
    
    for run in RUNS:
        key = run["key"]
        metrics_file = f"metrics/ratio_{key}/run_metrics.json"
        train_csv = run["train"]
        
        if not os.path.exists(metrics_file):
            logging.error(f"Missing metrics file: {metrics_file}")
            continue
            
        with open(metrics_file, "r") as f:
            m = json.load(f)
            
        df_train = pd.read_csv(train_csv)
        pos = (df_train["Outcome1"] == 1).sum()
        neg = (df_train["Outcome1"] == 0).sum()
        
        val = m.get("validation", {})
        test = m.get("test", {})
        thresh = m.get("threshold", 0.5)
        
        results.append({
            "Ratio": run["name"],
            "Train Pos/Neg": f"{pos}/{neg}",
            "Val Thresh": thresh,
            "Val PR-AUC": val.get("PR-AUC", 0.0),
            "Val F1": val.get("F1", 0.0),
            "Test PR-AUC": test.get("PR-AUC", 0.0),
            "Test F1": test.get("F1", 0.0),
            "Test Prec": test.get("Precision", 0.0),
            "Test Rec": test.get("Recall", 0.0),
            "Test MCC": test.get("MCC", 0.0),
        })

    if not results:
        logging.error("No results to display.")
        return

    res_df = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print("CLASS RATIO EXPERIMENT RESULTS SUMMARY (Tuned on Val, Evaluated on Fixed Test)")
    print("="*100)
    print(res_df.to_string(index=False))
    print("="*100)
    
    # Analysis logic
    baseline_rows = res_df[res_df["Ratio"].str.contains("Original")]
    if not baseline_rows.empty:
        baseline_prauc = baseline_rows["Test PR-AUC"].values[0]
        max_prauc = res_df["Test PR-AUC"].max()
        best_ratio = res_df.loc[res_df["Test PR-AUC"] == max_prauc, "Ratio"].values[0]
        diff = max_prauc - baseline_prauc
        
        print("\nDIAGNOSTIC CONCLUSION:")
        print("-" * 50)
        print(f"  Baseline (Original Train) Test PR-AUC: {baseline_prauc:.4f}")
        print(f"  Best Ratio Split ({best_ratio}) Test PR-AUC: {max_prauc:.4f} (Delta = {diff:+.4f})")
        
        if diff > 0.05:
            print("  -> CLASS IMBALANCE is a major bottleneck! Undersampling negatives significantly improves performance.")
        elif abs(diff) <= 0.02:
            print("  -> ARCHITECTURE is the primary bottleneck! Changing train class ratios has minimal impact on PR-AUC.")
        else:
            print("  -> Mild impact from class balance. Combine mild ratio adjusting with architectural improvements.")
        print("-" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run ratio experiments with fixed main val & test sets.")
    parser.add_argument("--skip-generation", action="store_true", help="Skip split generation step.")
    parser.add_argument("--compare-only", action="store_true", help="Only show comparison table of existing runs.")
    args = parser.parse_args()

    if args.compare_only:
        print_comparison_table()
        return

    if not args.skip_generation:
        generate_ratio_splits()

    run_experiments()
    print_comparison_table()


if __name__ == "__main__":
    main()

"""
make_ratio_splits.py
====================
Creates three resampled training sets from the existing leakage-safe train.csv,
targeting compatible:incompatible ratios of 1:1, 3:1, and 4:1.

Key design decisions
--------------------
* Val and test sets are NEVER touched — they stay at natural distribution
  so your metric comparisons across experiments are apples-to-apples.
* Only the TRAIN set is undersampled (negatives are randomly removed).
* Positives are always kept in full — you have only 191; losing any of
  them would hurt minority-class recall directly.
* Undersampling is stratified by API_CID: for each API that has negatives,
  we remove proportionally, so no single API's negatives dominate.
  This preserves the within-train API distribution as much as possible.
* random_state=42 for reproducibility everywhere.

Outputs (written to ./ratio_splits/)
--------------------------------------
  train_1_1.csv   — 191 pos + 191 neg  = 382  rows  (1:1)
  train_3_1.csv   — 191 pos + 573 neg  = 764  rows  (3:1)
  train_4_1.csv   — 191 pos + 764 neg  = 955  rows  (4:1)
  val.csv         — unchanged copy (convenience)
  test.csv        — unchanged copy (convenience)

Usage
-----
  python make_ratio_splits.py
  python make_ratio_splits.py --train_path /path/to/train.csv
"""

import argparse
import os
import pandas as pd
import numpy as np

RANDOM_STATE = 42
RATIOS = {
    "1_1": 1,
    "3_1": 3,
    "4_1": 4,
}
OUT_DIR = "ratio_splits"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def stratified_undersample_negatives(
    train_df: pd.DataFrame,
    n_neg_target: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Keep ALL positives.
    Sample exactly n_neg_target negatives, stratified by API_CID so that
    each API contributes negatives in proportion to its share of the total
    negative pool.
    Returns the combined (positives + sampled negatives) dataframe.
    """
    rng = np.random.default_rng(random_state)

    pos_df = train_df[train_df["Outcome1"] == 1].copy()
    neg_df = train_df[train_df["Outcome1"] == 0].copy()

    n_neg_available = len(neg_df)
    assert n_neg_target <= n_neg_available, (
        f"Requested {n_neg_target} negatives but only {n_neg_available} available."
    )

    # --- stratified sampling by API_CID ---
    # Compute the "fair share" of n_neg_target for each API proportionally.
    api_neg_counts = neg_df.groupby("API_CID").size()          # series: api -> count
    proportions    = api_neg_counts / api_neg_counts.sum()
    api_allocations = (proportions * n_neg_target).apply(np.floor).astype(int)

    # Distribute the remainder (due to floor) to the APIs with the largest
    # fractional parts, one extra each, until we hit n_neg_target exactly.
    remainder = n_neg_target - api_allocations.sum()
    frac_parts = (proportions * n_neg_target) - api_allocations
    top_apis   = frac_parts.nlargest(int(remainder)).index
    api_allocations.loc[top_apis] += 1

    assert api_allocations.sum() == n_neg_target, "allocation arithmetic error"

    # Draw from each API
    sampled_neg_parts = []
    for api_cid, n_draw in api_allocations.items():
        api_rows = neg_df[neg_df["API_CID"] == api_cid]
        if n_draw == 0:
            continue
        if n_draw >= len(api_rows):
            sampled_neg_parts.append(api_rows)
        else:
            chosen_idx = rng.choice(api_rows.index, size=int(n_draw), replace=False)
            sampled_neg_parts.append(api_rows.loc[chosen_idx])

    sampled_neg_df = pd.concat(sampled_neg_parts)

    # Combine and shuffle
    result = pd.concat([pos_df, sampled_neg_df]).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    return result


def print_split_summary(name: str, df: pd.DataFrame) -> None:
    n    = len(df)
    pos  = int((df["Outcome1"] == 1).sum())
    neg  = int((df["Outcome1"] == 0).sum())
    apis = df["API_CID"].nunique()
    exc  = df["Excipient_CID"].nunique()
    print(
        f"  {name:15s} | rows={n:5d} | pos={pos:4d} ({pos/n:5.1%}) "
        f"| neg={neg:4d} ({neg/n:5.1%}) | ratio={neg/pos:.1f}:1 "
        f"| APIs={apis} | excipients={exc}"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(train_path: str, val_path: str, test_path: str, out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)

    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)
    test  = pd.read_csv(test_path)

    n_pos = int((train["Outcome1"] == 1).sum())
    n_neg = int((train["Outcome1"] == 0).sum())

    print("=" * 70)
    print("INPUT SPLIT STATS")
    print("=" * 70)
    print_split_summary("TRAIN (original)", train)
    print_split_summary("VAL   (kept as-is)", val)
    print_split_summary("TEST  (kept as-is)", test)

    print()
    print("=" * 70)
    print("CREATING RATIO VARIANTS  (train only; val/test unchanged)")
    print("=" * 70)

    results = {}
    for ratio_name, neg_multiplier in RATIOS.items():
        n_neg_target = n_pos * neg_multiplier
        print(f"\n  Ratio {ratio_name.replace('_',':')} -> "
              f"keep all {n_pos} positives + sample {n_neg_target} negatives "
              f"(from {n_neg} available)")

        resampled = stratified_undersample_negatives(train, n_neg_target)
        results[ratio_name] = resampled
        print_split_summary(f"  train_{ratio_name}", resampled)

    # ---------------------------------------------------------------------------
    # Leakage verification across all variants
    # ---------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("LEAKAGE VERIFICATION (all ratio variants share same API grouping)")
    print("=" * 70)

    val_apis  = set(val["API_CID"])
    test_apis = set(test["API_CID"])

    for ratio_name, resampled in results.items():
        train_apis = set(resampled["API_CID"])
        tv = len(train_apis & val_apis)
        tt = len(train_apis & test_apis)
        print(f"  train_{ratio_name}: train<->val overlap={tv}, train<->test overlap={tt}",
              "  OK" if tv == 0 and tt == 0 else "  *** LEAKAGE ***")

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SAVING FILES")
    print("=" * 70)

    for ratio_name, resampled in results.items():
        out_path = os.path.join(out_dir, f"train_{ratio_name}.csv")
        resampled.to_csv(out_path, index=False)
        print(f"  {out_path}  ({len(resampled)} rows)")

    # Copy val and test for convenience (you only need one reference copy)
    val.to_csv(os.path.join(out_dir, "val.csv"),  index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"  {out_dir}/val.csv   ({len(val)} rows)  [unchanged]")
    print(f"  {out_dir}/test.csv  ({len(test)} rows)  [unchanged]")

    print()
    print("=" * 70)
    print("EXPERIMENT GUIDE")
    print("=" * 70)
    print("""
  Train three separate model runs:
    Run A:  train_1_1.csv  +  val.csv  ->  evaluate on test.csv
    Run B:  train_3_1.csv  +  val.csv  ->  evaluate on test.csv
    Run C:  train_4_1.csv  +  val.csv  ->  evaluate on test.csv
    Run D:  train.csv (original, natural ratio) + val.csv -> test.csv  [baseline]

  Keep EVERYTHING else identical across runs:
    - same model architecture
    - same hyperparameters (lr, weight decay, epochs, etc.)
    - same loss function (e.g. ASL or BCE -- pick one for this experiment)
    - same threshold (or tune threshold on val for each run independently)

  Metrics to compare (on test.csv at natural distribution):
    - PR-AUC  (primary, distribution-invariant)
    - ROC-AUC
    - F1 / Precision / Recall @ val-tuned threshold
    - Calibration (ECE) if you care about probability outputs

  Interpretation guide:
    If PR-AUC improves a lot from Run D -> Run A/B/C:
      -> Class imbalance was a major contributor to poor metrics.
         Fix: use undersampling or a strong class-weighting loss on the FULL train set.

    If PR-AUC barely changes across runs:
      -> The architecture / feature representation is the bottleneck.
         Fix: better molecular features, cross-attention, larger backbone, etc.

    If PR-AUC peaks at 3:1 or 4:1 (not 1:1):
      -> Mild imbalance handling helps, but aggressive undersampling (1:1) discards
         too many negatives and the model stops learning the true decision boundary.
         Fix: class weights / focal loss / ASL instead of hard undersampling.
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/train.csv")
    parser.add_argument("--val_path",   default="data/val.csv")
    parser.add_argument("--test_path",  default="data/test.csv")
    parser.add_argument("--out_dir",    default="data/ratio_splits")
    args = parser.parse_args()
    main(args.train_path, args.val_path, args.test_path, args.out_dir)
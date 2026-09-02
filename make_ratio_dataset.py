"""
make_ratio_dataset.py

Builds a class-ratio-controlled SUBSET of the full API-Excipient dataset,
then re-runs the same leakage-safe, API-grouped, stratified split used for
the main dataset (see data/data_split.py) on that subset only.

READ-ONLY on the original data: only ever reads data/start_dataset.csv (and
the two descriptor CSVs). Never writes to data/train.csv, data/val.csv,
data/test.csv, or data/start_dataset.csv. All output goes to a new directory
under data/ratio_experiments/.

Negative selection logic (not pure random):
  Every negative row is scored by how "informative" it is for contrastive
  learning:
    priority 2 - both the API and the Excipient in this negative pair also
                 appear somewhere in the positive set (best case: the model
                 sees the *same* compounds in both a compatible and an
                 incompatible context)
    priority 1 - exactly one of (API, Excipient) also appears in the
                 positive set
    priority 0 - neither compound appears in any positive pair (kept last,
                 purely for chemical diversity)
  Within each priority tier, a soft per-API_CID cap prevents any single API
  from dominating the selection just because it happens to have many logged
  excipient partners in the source data.

Usage:
    python make_ratio_dataset.py --ratio 4      # 4 negatives per 1 positive
    python make_ratio_dataset.py --ratio 3
    python make_ratio_dataset.py --ratio 1      # full 1:1 balance
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold

RDLogger.DisableLog("rdApp.*")

BUTINA_CUTOFF = 0.15          # distance cutoff -> Tanimoto similarity >= 0.85
MERGE_THRESHOLD_PCT = 5.0     # only switch to cluster-based grouping if >this% of APIs merge


def parse_args():
    p = argparse.ArgumentParser(description="Build a class-ratio-controlled dataset subset.")
    p.add_argument("--ratio", type=float, required=True,
                   help="Negatives per positive, e.g. 4 for 4:1, 1 for full 1:1 balance.")
    p.add_argument("--input", default="data/start_dataset.csv",
                   help="Full source pool (read-only). Default: data/start_dataset.csv")
    p.add_argument("--output-dir", default=None,
                   help="Where to write the new subset + splits. "
                        "Default: data/ratio_experiments/ratio_<ratio>to1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--api-descriptors", default="data/api_descriptors.csv")
    p.add_argument("--excipient-descriptors", default="data/excipient_descriptors.csv")
    return p.parse_args()


def select_negatives(pos_df, neg_pool, target_n, seed):
    """Priority- and diversity-aware negative sampling. See module docstring."""
    rng = np.random.default_rng(seed)

    pos_apis = set(pos_df["API_CID"])
    pos_excs = set(pos_df["Excipient_CID"])

    neg_pool = neg_pool.copy()
    neg_pool["_priority"] = (
        neg_pool["API_CID"].isin(pos_apis).astype(int)
        + neg_pool["Excipient_CID"].isin(pos_excs).astype(int)
    )

    n_unique_apis = neg_pool["API_CID"].nunique()
    # Generous soft cap: allows some concentration but blocks any single API
    # from supplying more than ~1.5x its "fair share" of the target count.
    soft_cap = max(1, math.ceil(1.5 * target_n / max(n_unique_apis, 1)))

    selected_idx = []
    api_counts = defaultdict(int)

    for tier in (2, 1, 0):
        if len(selected_idx) >= target_n:
            break
        tier_idx = neg_pool.index[neg_pool["_priority"] == tier].to_numpy().copy()
        rng.shuffle(tier_idx)

        for idx in tier_idx:
            if len(selected_idx) >= target_n:
                break
            api_cid = neg_pool.loc[idx, "API_CID"]
            if api_counts[api_cid] >= soft_cap:
                continue
            selected_idx.append(idx)
            api_counts[api_cid] += 1

    # If soft caps left us short (possible at very high target ratios or with
    # a very skewed API distribution), relax caps and fill the remainder
    # randomly from whatever negatives are left.
    if len(selected_idx) < target_n:
        remaining_pool = neg_pool.index.difference(selected_idx).to_numpy().copy()
        rng.shuffle(remaining_pool)
        n_needed = target_n - len(selected_idx)
        selected_idx.extend(remaining_pool[:n_needed].tolist())

    return neg_pool.loc[selected_idx].drop(columns=["_priority"])


def grouped_stratified_split(df, seed):
    """
    Same leakage-safe strategy as data/data_split.py: group by API_CID
    (merging Butina near-duplicate clusters if >5% of APIs would merge),
    stratify on Outcome1, 60/20/20 train/val/test.
    """
    df = df.reset_index(drop=True)
    api_df = df[["API_CID", "API_Smiles"]].drop_duplicates().reset_index(drop=True)

    mols, valid_idx = [], []
    for i, smi in enumerate(api_df["API_Smiles"]):
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)
            valid_idx.append(i)

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    if n > 1:
        clusters = Butina.ClusterData(dists, n, BUTINA_CUTOFF, isDistData=True)
    elif n == 1:
        clusters = [[0]]
    else:
        clusters = []

    cluster_id_by_pos = {}
    for cid, members in enumerate(clusters):
        for m_idx in members:
            cluster_id_by_pos[m_idx] = cid

    api_df["butina_cluster"] = -1
    for local_pos, orig_idx in enumerate(valid_idx):
        api_df.loc[orig_idx, "butina_cluster"] = cluster_id_by_pos[local_pos]
    invalid_mask = api_df["butina_cluster"] == -1
    api_df.loc[invalid_mask, "butina_cluster"] = (
        api_df.loc[invalid_mask].index + 10_000_000
    )

    cluster_sizes = api_df.groupby("butina_cluster")["API_CID"].nunique()
    multi_clusters = cluster_sizes[cluster_sizes > 1]
    n_apis_merged = int(multi_clusters.sum())
    pct_merged = 100 * n_apis_merged / max(len(api_df), 1)

    group_col_name = "butina_cluster" if pct_merged > MERGE_THRESHOLD_PCT else "API_CID"

    df = df.merge(api_df[["API_CID", "butina_cluster"]], on="API_CID", how="left")
    df["split_group"] = df[group_col_name] if group_col_name == "butina_cluster" else df["API_CID"]

    sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf1.split(df, y=df["Outcome1"], groups=df["split_group"]))
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    sgkf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    train_idx, val_idx = next(
        sgkf2.split(trainval_df, y=trainval_df["Outcome1"], groups=trainval_df["split_group"])
    )
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    cols_to_drop = ["butina_cluster", "split_group"]
    return (
        train_df.drop(columns=cols_to_drop),
        val_df.drop(columns=cols_to_drop),
        test_df.drop(columns=cols_to_drop),
        group_col_name,
    )


def main():
    args = parse_args()
    output_dir = args.output_dir or f"data/ratio_experiments/ratio_{args.ratio:g}to1"
    os.makedirs(output_dir, exist_ok=True)

    full_df = pd.read_csv(args.input)
    pos_df = full_df[full_df["Outcome1"] == 1].reset_index(drop=True)
    neg_pool = full_df[full_df["Outcome1"] == 0].reset_index(drop=True)

    n_pos = len(pos_df)
    target_neg = int(round(args.ratio * n_pos))
    target_neg = min(target_neg, len(neg_pool))  # can't exceed what exists

    selected_neg = select_negatives(pos_df, neg_pool, target_neg, args.seed)
    subset_df = pd.concat([pos_df, selected_neg], ignore_index=True)
    subset_df = subset_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    subset_path = os.path.join(output_dir, "subset_dataset.csv")
    subset_df.to_csv(subset_path, index=False)

    train_df, val_df, test_df, group_col_name = grouped_stratified_split(subset_df, args.seed)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # Regenerate descriptor normalization stats for THIS train split only --
    # writes into output_dir, never touches models/descriptor_norm_stats.json
    from src.descriptors import write_descriptor_norm_stats
    norm_stats_path = os.path.join(output_dir, "descriptor_norm_stats.json")
    write_descriptor_norm_stats(
        train_csv_path=os.path.join(output_dir, "train.csv"),
        api_csv_path=args.api_descriptors,
        exc_csv_path=args.excipient_descriptors,
        norm_stats_path=norm_stats_path,
    )

    manifest = {
        "ratio_requested": args.ratio,
        "n_positive": n_pos,
        "n_negative_selected": len(selected_neg),
        "n_negative_available_in_source": len(neg_pool),
        "actual_ratio": len(selected_neg) / n_pos,
        "grouping_key_used": group_col_name,
        "splits": {
            "train": {"rows": len(train_df), "pos": int((train_df["Outcome1"] == 1).sum())},
            "val": {"rows": len(val_df), "pos": int((val_df["Outcome1"] == 1).sum())},
            "test": {"rows": len(test_df), "pos": int((test_df["Outcome1"] == 1).sum())},
        },
        "negative_priority_breakdown": {
            "both_compounds_seen_in_positives": int(
                (
                    selected_neg["API_CID"].isin(set(pos_df["API_CID"]))
                    & selected_neg["Excipient_CID"].isin(set(pos_df["Excipient_CID"]))
                ).sum()
            ),
        },
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {output_dir}/  (train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")
    print(f"Actual negative:positive ratio = {manifest['actual_ratio']:.2f}:1")


if __name__ == "__main__":
    main()

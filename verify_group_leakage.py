"""
verify_group_leakage.py

Stricter leakage check for the ratio-experiment datasets: verifies not just
that no single API_CID appears in more than one split, but that no group of
*chemically near-duplicate* APIs (Tanimoto similarity >= 0.85 on Morgan
fingerprints, via Butina clustering -- same method used in
data/data_split.py and make_ratio_dataset.py) has its members spread across
more than one split. Two different API_CIDs can be the same molecule in a
different salt form, for example -- if one lands in train and its
near-duplicate lands in test, that's leakage even though the raw API_CID
sets never technically overlap.

This is a read-only AUDIT script. It recomputes clusters itself from each
dataset's full pool (subset_dataset.csv for a ratio experiment,
start_dataset.csv for the original data/ directory) and cross-checks them
against the actual saved train.csv/val.csv/test.csv -- it never modifies any
file, and it doesn't rely on trusting the grouping key the split happened to
use internally.

Checks performed per dataset directory:
  1. Raw API_CID disjointness across train/val/test (baseline leakage check).
  2. Near-duplicate cluster disjointness: every Butina cluster (>=0.85
     Tanimoto) must have ALL of its member API_CIDs inside a single split.
     A cluster spanning two or more splits is flagged explicitly, listing
     the API_CIDs and which split each landed in.

Usage:
    python verify_group_leakage.py data data/ratio_experiments/ratio_4to1 \
        data/ratio_experiments/ratio_3to1 data/ratio_experiments/ratio_1to1

    # or auto-discover data/ plus every ratio_*to1 directory:
    python verify_group_leakage.py --auto
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")

BUTINA_CUTOFF = 0.15  # distance cutoff -> Tanimoto similarity >= 0.85


def parse_args():
    p = argparse.ArgumentParser(description="Verify no near-duplicate-API-group leakage across splits.")
    p.add_argument("dirs", nargs="*", help="Dataset directories to check.")
    p.add_argument("--auto", action="store_true",
                   help="Auto-discover: data/ plus every data/ratio_experiments/ratio_*to1/.")
    return p.parse_args()


def discover_dirs():
    dirs = ["data"]
    dirs.extend(sorted(glob.glob("data/ratio_experiments/ratio_*to1")))
    return dirs


def find_full_pool_path(dir_path):
    for name in ("subset_dataset.csv", "start_dataset.csv"):
        candidate = os.path.join(dir_path, name)
        if os.path.exists(candidate):
            return candidate
    return None


def compute_clusters(full_pool_df):
    """Return {API_CID: cluster_id} for every unique API in the full pool."""
    api_df = full_pool_df[["API_CID", "API_Smiles"]].drop_duplicates().reset_index(drop=True)

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

    api_df["cluster"] = -1
    for local_pos, orig_idx in enumerate(valid_idx):
        api_df.loc[orig_idx, "cluster"] = cluster_id_by_pos[local_pos]
    # invalid SMILES fall back to their own singleton cluster
    invalid_mask = api_df["cluster"] == -1
    api_df.loc[invalid_mask, "cluster"] = api_df.loc[invalid_mask].index + 10_000_000

    return dict(zip(api_df["API_CID"], api_df["cluster"]))


def check_dataset_dir(dir_path):
    print("=" * 78)
    print(f"CHECKING: {dir_path}")
    print("=" * 78)

    required = ["train.csv", "val.csv", "test.csv"]
    missing = [f for f in required if not os.path.exists(os.path.join(dir_path, f))]
    if missing:
        print(f"  SKIPPED - missing files: {missing}")
        return True

    train_df = pd.read_csv(os.path.join(dir_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(dir_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(dir_path, "test.csv"))

    manifest_path = os.path.join(dir_path, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            grouping_key_used = json.load(f).get("grouping_key_used", "unknown")
        print(f"  grouping_key_used (from manifest.json): {grouping_key_used}")

    ok = True

    # --- Check 1: raw API_CID disjointness ----------------------------------
    split_apis = {
        "train": set(train_df["API_CID"]),
        "val": set(val_df["API_CID"]),
        "test": set(test_df["API_CID"]),
    }
    tv = split_apis["train"] & split_apis["val"]
    tt = split_apis["train"] & split_apis["test"]
    vt = split_apis["val"] & split_apis["test"]
    print(f"  API_CID overlap train<->val  : {len(tv)}")
    print(f"  API_CID overlap train<->test : {len(tt)}")
    print(f"  API_CID overlap val<->test   : {len(vt)}")
    if tv or tt or vt:
        print("  *** LEAKAGE: same API_CID appears in more than one split ***")
        ok = False
    else:
        print("  PASS: no raw API_CID leakage.")

    # --- Check 2: near-duplicate cluster disjointness -----------------------
    full_pool_path = find_full_pool_path(dir_path)
    if full_pool_path is None:
        print("  (no subset_dataset.csv / start_dataset.csv found -- skipping cluster check)")
        print()
        return ok

    full_pool_df = pd.read_csv(full_pool_path)
    cluster_by_api = compute_clusters(full_pool_df)

    # which split does each API_CID belong to (an API only appears in one
    # split if check 1 passed, so this is a safe 1:1 lookup)
    api_to_split = {}
    for split_name, apis in split_apis.items():
        for api in apis:
            api_to_split[api] = split_name

    cluster_to_apis = {}
    for api_cid, cluster_id in cluster_by_api.items():
        cluster_to_apis.setdefault(cluster_id, set()).add(api_cid)

    multi_member_clusters = {cid: apis for cid, apis in cluster_to_apis.items() if len(apis) > 1}
    print(f"  Near-duplicate clusters (Tanimoto>=0.85) with >1 API: {len(multi_member_clusters)}")

    leaking_clusters = []
    for cluster_id, apis in multi_member_clusters.items():
        splits_touched = {api_to_split.get(a) for a in apis if a in api_to_split}
        splits_touched.discard(None)
        if len(splits_touched) > 1:
            leaking_clusters.append((cluster_id, apis, splits_touched))

    if leaking_clusters:
        print(f"  *** GROUP LEAKAGE DETECTED: {len(leaking_clusters)} near-duplicate "
              f"cluster(s) span more than one split ***")
        for cluster_id, apis, splits_touched in leaking_clusters:
            detail = ", ".join(f"{a}->{api_to_split.get(a)}" for a in sorted(apis))
            print(f"    cluster {cluster_id}: {detail}  (splits: {sorted(splits_touched)})")
        ok = False
    else:
        print("  PASS: every near-duplicate API cluster stays within a single split.")

    print()
    return ok


def main():
    args = parse_args()
    dirs = discover_dirs() if args.auto else args.dirs
    if not dirs:
        print("No directories given. Use --auto or pass directories explicitly.")
        sys.exit(1)

    results = {d: check_dataset_dir(d) for d in dirs}

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    all_ok = True
    for d, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status:5s} - {d}")
        all_ok = all_ok and ok

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
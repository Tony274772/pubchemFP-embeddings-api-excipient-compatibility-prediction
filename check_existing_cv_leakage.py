"""
Verify existing CVs for leakage by reconstructing folds with the same seed.

This script simulates the original fold splitting and checks if Butina clusters
span across train/validation splits (which would indicate data leakage).
"""

import logging
import os
import sys

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

BUTINA_CUTOFF = 0.15  # distance cutoff -> Tanimoto similarity >= 0.85
RATIOS = ["1to1", "3to1", "4to1"]


def compute_butina_clusters(smiles_list):
    """Compute Butina clusters and return cluster ID for each sample."""
    fps = []
    valid_indices = []
    
    for i, smi in enumerate(smiles_list):
        if pd.isna(smi) or not str(smi).strip():
            fps.append(None)
        else:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                fps.append(None)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
                valid_indices.append(i)
    
    if not valid_indices:
        return list(range(len(smiles_list)))
    
    valid_fps = [fps[i] for i in valid_indices]
    distances = []
    
    for i in range(len(valid_fps)):
        for j in range(i + 1, len(valid_fps)):
            sim = DataStructs.TanimotoSimilarity(valid_fps[i], valid_fps[j])
            dist = 1 - sim
            distances.append(dist)
    
    if not distances:
        return list(range(len(smiles_list)))
    
    clusters = Butina.ClusterData(distances, len(valid_fps), BUTINA_CUTOFF, isDistData=True)
    
    cluster_ids = [-1] * len(smiles_list)
    for cluster_id, members in enumerate(clusters):
        for member_idx in members:
            original_idx = valid_indices[member_idx]
            cluster_ids[original_idx] = cluster_id
    
    max_cluster_id = len(clusters) - 1
    for i in range(len(smiles_list)):
        if cluster_ids[i] == -1:
            max_cluster_id += 1
            cluster_ids[i] = max_cluster_id
    
    return cluster_ids


def check_existing_cv_for_leakage(data_dir, ratio_name, seed=42):
    """
    Reconstruct folds using original seed and check for Butina cluster leakage.
    
    Returns:
        Dict with leakage status
    """
    train_path = os.path.join(data_dir, "train.csv")
    
    if not os.path.exists(train_path):
        logging.warning(f"{ratio_name}: train.csv not found")
        return {"ratio": ratio_name, "status": "SKIP", "reason": "No train.csv"}
    
    df = pd.read_csv(train_path)
    
    # Compute Butina clusters
    if "API_Smiles" not in df.columns:
        logging.warning(f"{ratio_name}: No API_Smiles column")
        return {"ratio": ratio_name, "status": "SKIP", "reason": "No API_Smiles"}
    
    logging.info(f"\nChecking {ratio_name}...")
    logging.info(f"Computing Butina clusters for {len(df)} samples...")
    cluster_ids = compute_butina_clusters(df["API_Smiles"].tolist())
    df["butina_cluster"] = cluster_ids
    n_clusters = len(set(cluster_ids))
    logging.info(f"  → {n_clusters} Butina clusters created")
    
    # Recreate folds using original parameters
    y = df["Outcome1"]
    groups = df["butina_cluster"]  # Use Butina clusters for grouping
    
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    
    result = {
        "ratio": ratio_name,
        "status": "PASS",
        "leakage_found": False,
        "n_clusters": n_clusters,
        "folds": {}
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(df, y, groups), start=1):
        fold_train = df.iloc[train_idx]
        fold_val = df.iloc[val_idx]
        
        train_clusters = set(fold_train["butina_cluster"].unique())
        val_clusters = set(fold_val["butina_cluster"].unique())
        overlap = train_clusters & val_clusters
        
        fold_info = {
            "train_samples": len(fold_train),
            "val_samples": len(fold_val),
            "train_clusters": len(train_clusters),
            "val_clusters": len(val_clusters),
            "cluster_overlap": len(overlap),
            "has_leakage": len(overlap) > 0
        }
        
        if overlap:
            result["status"] = "FAIL"
            result["leakage_found"] = True
            fold_info["overlapping_clusters"] = sorted(list(overlap))
            logging.error(f"  Fold {fold_idx}: ❌ {len(overlap)} clusters span train and validation!")
            # Show a few examples
            for cluster_id in list(overlap)[:3]:
                apis_train = set(fold_train[fold_train["butina_cluster"] == cluster_id]["API_CID"].unique())
                apis_val = set(fold_val[fold_val["butina_cluster"] == cluster_id]["API_CID"].unique())
                logging.error(f"    Cluster {cluster_id}: {len(apis_train)} APIs in train, {len(apis_val)} in val")
        else:
            logging.info(f"  Fold {fold_idx}: ✓ No cluster leakage ({len(train_clusters)} train, {len(val_clusters)} val)")
        
        result["folds"][f"fold_{fold_idx}"] = fold_info
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Check existing CV folds for Butina cluster leakage (reconstructs folds from seed)."
    )
    parser.add_argument(
        "--ratio",
        choices=["all"] + RATIOS,
        default="all",
        help="Which ratio to check"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for original fold splitting"
    )
    
    args = parser.parse_args()
    
    results = {}
    
    if args.ratio == "all":
        for ratio in RATIOS:
            data_dir = f"data/ratio_experiments/ratio_{ratio}"
            result = check_existing_cv_for_leakage(data_dir, f"ratio_{ratio}", args.seed)
            results[ratio] = result
    else:
        data_dir = f"data/ratio_experiments/ratio_{args.ratio}"
        result = check_existing_cv_for_leakage(data_dir, f"ratio_{args.ratio}", args.seed)
        results[args.ratio] = result
    
    # Summary
    print("\n" + "="*70)
    print("CV LEAKAGE CHECK - RECONSTRUCTED FOLDS")
    print("="*70)
    
    for name, result in results.items():
        status = result["status"]
        leakage = "LEAKAGE" if result.get("leakage_found") else "OK"
        print(f"  {name:15s}: {status:6s} [{leakage}]")
    
    print("="*70)
    
    # Detailed results
    print("\nDETAILS:")
    for name, result in results.items():
        if result["status"] != "SKIP":
            print(f"\n{result['ratio']}:")
            print(f"  Total Butina clusters: {result['n_clusters']}")
            for fold, info in result["folds"].items():
                print(f"  {fold}:")
                print(f"    Train: {info['train_samples']} samples, {info['train_clusters']} clusters")
                print(f"    Val:   {info['val_samples']} samples, {info['val_clusters']} clusters")
                print(f"    Overlap: {info['cluster_overlap']} clusters " + 
                      ("❌ LEAKAGE" if info["has_leakage"] else "✓ OK"))
    
    print("\n" + "="*70)
    all_pass = all(r["status"] == "PASS" or r["status"] == "SKIP" for r in results.values() if r["status"] != "FAIL")
    has_leakage = any(r.get("leakage_found", False) for r in results.values())
    
    if has_leakage:
        print("⚠️  DATA LEAKAGE DETECTED - Need to retrain with cross_validate_no_leakage.py")
    else:
        print("✓ NO DATA LEAKAGE - Results are statistically valid")
    
    print("="*70 + "\n")
    
    sys.exit(0 if not has_leakage else 1)


if __name__ == "__main__":
    main()

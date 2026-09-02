"""
Verify that cross-validation folds have NO data leakage via Butina clustering.

This script audits CV folds to ensure:
  1. No exact API_CID appears in both train and validation
  2. No Butina cluster (chemically similar APIs) spans train and validation

Usage:
    python verify_cv_fold_integrity.py --ratio 1to1
    python verify_cv_fold_integrity.py --ratio all
    python verify_cv_fold_integrity.py --data-dir data/ratio_experiments/ratio_1to1 \\
                                       --checkpoint-dir checkpoints/ratio_experiments_cv/ratio_1to1
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina

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


def verify_fold_integrity(data_dir, checkpoint_dir, ratio_name=None):
    """
    Verify a single ratio's 5-fold CV for leakage.
    
    Returns:
        Dict with leakage status and details
    """
    result = {
        "ratio": ratio_name or os.path.basename(data_dir),
        "status": "UNKNOWN",
        "leakage_type": None,
        "details": {},
        "folds": {}
    }
    
    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        result["status"] = "SKIP"
        result["details"]["reason"] = f"No train.csv at {train_path}"
        return result
    
    df = pd.read_csv(train_path)
    
    # Compute Butina clusters once for the entire dataset
    if "API_Smiles" in df.columns:
        logging.info(f"Computing Butina clusters for {ratio_name or 'dataset'}...")
        cluster_ids = compute_butina_clusters(df["API_Smiles"].tolist())
        df["butina_cluster"] = cluster_ids
        n_clusters = len(set(cluster_ids))
        logging.info(f"  → {n_clusters} Butina clusters")
    else:
        logging.warning(f"  No API_Smiles in {ratio_name}, skipping Butina check")
        n_clusters = None
    
    # Check each fold
    all_pass = True
    
    for fold_num in range(1, 6):
        fold_checkpoint = os.path.join(checkpoint_dir, f"cv_fold_{fold_num}")
        fold_result = {
            "checkpoint_exists": os.path.exists(fold_checkpoint),
            "api_cid_leakage": False,
            "butina_leakage": False,
            "overlapping_api_cids": [],
            "overlapping_clusters": [],
            "train_count": 0,
            "val_count": 0,
        }
        
        if not fold_result["checkpoint_exists"]:
            logging.warning(f"  Fold {fold_num}: No checkpoint found")
            result["folds"][f"fold_{fold_num}"] = fold_result
            continue
        
        # Try to reconstruct folds from the training script
        # For now, we'll check if we can find fold indices in metadata
        fold_manifest = os.path.join(fold_checkpoint, "fold_manifest.json")
        
        if os.path.exists(fold_manifest):
            with open(fold_manifest) as f:
                manifest = json.load(f)
                train_indices = set(manifest.get("train_indices", []))
                val_indices = set(manifest.get("val_indices", []))
        else:
            # Fallback: we can't verify without saved indices
            # This is a limitation - we'd need to save fold indices during training
            logging.warning(f"  Fold {fold_num}: No fold_manifest.json, can't verify indices")
            fold_result["checkpoint_exists"] = True
            fold_result["error"] = "No fold manifest saved (need to re-train with save_manifest=True)"
            result["folds"][f"fold_{fold_num}"] = fold_result
            all_pass = False
            continue
        
        fold_train = df.iloc[sorted(train_indices)]
        fold_val = df.iloc[sorted(val_indices)]
        
        fold_result["train_count"] = len(fold_train)
        fold_result["val_count"] = len(fold_val)
        
        # Check exact API_CID leakage
        train_apis = set(fold_train["API_CID"].unique())
        val_apis = set(fold_val["API_CID"].unique())
        api_overlap = train_apis & val_apis
        
        if api_overlap:
            fold_result["api_cid_leakage"] = True
            fold_result["overlapping_api_cids"] = sorted(list(api_overlap))
            all_pass = False
            logging.error(f"  Fold {fold_num}: ❌ {len(api_overlap)} APIs appear in both train and val!")
        
        # Check Butina cluster leakage (if clusters exist)
        if n_clusters is not None:
            train_clusters = set(fold_train["butina_cluster"].unique())
            val_clusters = set(fold_val["butina_cluster"].unique())
            cluster_overlap = train_clusters & val_clusters
            
            if cluster_overlap:
                fold_result["butina_leakage"] = True
                fold_result["overlapping_clusters"] = sorted(list(cluster_overlap))
                all_pass = False
                logging.error(f"  Fold {fold_num}: ❌ {len(cluster_overlap)} Butina clusters span train and val!")
        
        if not fold_result["api_cid_leakage"] and not fold_result["butina_leakage"]:
            logging.info(f"  Fold {fold_num}: ✓ No leakage detected")
        
        result["folds"][f"fold_{fold_num}"] = fold_result
    
    # Summarize
    if all_pass:
        result["status"] = "PASS"
        logging.info(f"{ratio_name or 'Dataset'}: ✓ ALL FOLDS PASS\n")
    else:
        result["status"] = "FAIL"
        if any(f["api_cid_leakage"] for f in result["folds"].values()):
            result["leakage_type"] = "API_CID"
        elif any(f["butina_leakage"] for f in result["folds"].values()):
            result["leakage_type"] = "BUTINA_CLUSTER"
        logging.error(f"{ratio_name or 'Dataset'}: ❌ LEAKAGE DETECTED\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Verify CV fold integrity (check for data leakage via API_CID and Butina clustering)."
    )
    parser.add_argument(
        "--ratio",
        choices=["all"] + RATIOS,
        default="all",
        help="Which ratio to check (default: all)"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory (for manual check)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override checkpoint directory (for manual check)"
    )
    
    args = parser.parse_args()
    
    results = {}
    
    if args.data_dir and args.checkpoint_dir:
        # Manual check
        result = verify_fold_integrity(args.data_dir, args.checkpoint_dir, "custom")
        results["custom"] = result
    elif args.ratio == "all":
        # Check all ratios
        for ratio in RATIOS:
            data_dir = f"data/ratio_experiments/ratio_{ratio}"
            checkpoint_dir = f"checkpoints/ratio_experiments_cv/ratio_{ratio}"
            result = verify_fold_integrity(data_dir, checkpoint_dir, f"ratio_{ratio}")
            results[ratio] = result
    else:
        # Check one ratio
        data_dir = f"data/ratio_experiments/ratio_{args.ratio}"
        checkpoint_dir = f"checkpoints/ratio_experiments_cv/ratio_{args.ratio}"
        result = verify_fold_integrity(data_dir, checkpoint_dir, f"ratio_{args.ratio}")
        results[args.ratio] = result
    
    # Summary
    print("\n" + "="*70)
    print("CV FOLD INTEGRITY CHECK SUMMARY")
    print("="*70)
    
    for name, result in results.items():
        status = result["status"]
        leakage = result.get("leakage_type") or ""
        print(f"  {name:20s}: {status:6s} {leakage}")
    
    print("="*70)
    
    # Return exit code
    all_pass = all(r["status"] == "PASS" for r in results.values())
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

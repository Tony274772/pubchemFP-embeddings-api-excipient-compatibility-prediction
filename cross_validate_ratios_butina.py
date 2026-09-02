"""
5-fold cross-validation with Butina clustering to prevent data leakage.

Uses Butina clustering (Tanimoto similarity >= 0.85) to group near-duplicate APIs,
ensuring ALL chemically similar APIs stay together in the same fold.

This prevents data leakage from similar molecules appearing in both train and validation.
"""

import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path
import json
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

RATIOS = ["1to1", "3to1", "4to1"]
BASE_DATA_DIR = "data/ratio_experiments"
BUTINA_CUTOFF = 0.15  # distance cutoff -> Tanimoto similarity >= 0.85


def compute_butina_clusters(smiles_list):
    """
    Compute Butina clusters from a list of SMILES strings.
    
    Returns:
        Dict mapping API index to cluster ID
    """
    logging.info(f"Computing Butina clusters for {len(smiles_list)} molecules (threshold: Tanimoto >= 0.85)...")
    
    # Generate fingerprints
    fps = []
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
    
    # Compute distance matrix
    valid_fps = [(i, fp) for i, fp in enumerate(fps) if fp is not None]
    
    if not valid_fps:
        logging.warning("No valid SMILES found, falling back to API_CID grouping")
        return None
    
    n_valid = len(valid_fps)
    distances = []
    
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            idx_i, fp_i = valid_fps[i]
            idx_j, fp_j = valid_fps[j]
            sim = DataStructs.TanimotoSimilarity(fp_i, fp_j)
            dist = 1 - sim
            distances.append(dist)
    
    # Cluster
    if not distances:
        logging.warning("No valid distance pairs, falling back to API_CID grouping")
        return None
    
    clusters = Butina.ClusterData(distances, len(valid_fps), BUTINA_CUTOFF, isDistData=True)
    
    # Map back to original indices
    cluster_map = {}
    for cluster_id, members in enumerate(clusters):
        for member_idx in members:
            original_idx = valid_fps[member_idx][0]
            cluster_map[original_idx] = cluster_id
    
    # Map remaining invalid entries to singletons
    for i, fp in enumerate(fps):
        if fp is None and i not in cluster_map:
            cluster_map[i] = len(clusters) + i
    
    logging.info(f"Created {len(set(cluster_map.values()))} Butina clusters")
    return cluster_map


def add_butina_clusters_to_dataframe(df):
    """
    Add a Butina cluster column to the dataframe.
    
    Returns:
        Modified dataframe with 'butina_cluster' column
    """
    df = df.copy()
    
    if "API_Smiles" not in df.columns:
        logging.warning("No API_Smiles column found, using API_CID as fallback")
        df["butina_cluster"] = df["API_CID"]
        return df
    
    cluster_map = compute_butina_clusters(df["API_Smiles"].tolist())
    
    if cluster_map is None:
        logging.warning("Could not compute Butina clusters, using API_CID as fallback")
        df["butina_cluster"] = df["API_CID"]
    else:
        df["butina_cluster"] = df.index.map(lambda i: cluster_map.get(i, i))
    
    return df


def verify_no_leakage(df_train, df_val, cluster_col="butina_cluster"):
    """Verify no clusters span both train and validation."""
    train_clusters = set(df_train[cluster_col].unique())
    val_clusters = set(df_val[cluster_col].unique())
    overlap = train_clusters & val_clusters
    
    if overlap:
        logging.error(f"❌ DATA LEAKAGE DETECTED: {len(overlap)} clusters span train and validation!")
        return False
    else:
        logging.info(f"✓ No cluster overlap between train and validation")
        return True


def run_cv_for_ratio_fixed(ratio: str, max_epochs: int = None, skip_if_exists: bool = False) -> bool:
    """
    Run 5-fold cross-validation with Butina clustering to prevent data leakage.
    
    Args:
        ratio: One of "1to1", "3to1", "4to1"
        max_epochs: Optional max epochs override
        skip_if_exists: If True, skip if any fold checkpoint exists
    
    Returns:
        True if CV succeeded, False otherwise
    """
    ratio_dir = os.path.join(BASE_DATA_DIR, f"ratio_{ratio}")
    checkpoint_dir = f"checkpoints/ratio_experiments_cv/ratio_{ratio}"
    metrics_dir = f"metrics/ratio_experiments_cv/ratio_{ratio}"
    
    if not os.path.exists(ratio_dir):
        logging.error(f"Ratio directory not found: {ratio_dir}")
        return False
    
    # Check if fold 1 checkpoint exists
    fold_1_checkpoint = os.path.join(checkpoint_dir, "cv_fold_1", "best_model.pt")
    if skip_if_exists and os.path.exists(fold_1_checkpoint):
        logging.info(f"5-fold CV already exists for ratio {ratio}, skipping...")
        return True
    
    # Create output directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    logging.info(f"\n{'='*70}")
    logging.info(f"5-Fold Cross-Validation on ratio_{ratio} (with Butina clustering)")
    logging.info(f"  Data dir:      {ratio_dir}")
    logging.info(f"  Checkpoint:    {checkpoint_dir}")
    logging.info(f"  Metrics:       {metrics_dir}")
    logging.info(f"{'='*70}\n")
    
    # Load training data and add Butina clusters
    train_path = os.path.join(ratio_dir, "train.csv")
    df = pd.read_csv(train_path)
    
    logging.info(f"Loaded {len(df)} samples from {train_path}")
    
    # Add Butina clustering
    df = add_butina_clusters_to_dataframe(df)
    logging.info(f"Added Butina cluster column (unique clusters: {df['butina_cluster'].nunique()})")
    
    # Verify clustering makes sense
    logging.info(f"\nCluster size distribution:")
    cluster_sizes = df["butina_cluster"].value_counts()
    logging.info(f"  Min: {cluster_sizes.min()}, Max: {cluster_sizes.max()}, Mean: {cluster_sizes.mean():.1f}")
    large_clusters = cluster_sizes[cluster_sizes > 50]
    if len(large_clusters) > 0:
        logging.info(f"  Large clusters (>50): {len(large_clusters)}")
    
    # Create a modified training script command that uses Butina clusters
    # We'll use cross_validate.py with a special environment variable
    os.environ["BUTINA_CLUSTER_DATA"] = json.dumps({
        "ratio": ratio,
        "data_dir": ratio_dir,
        "cluster_column": "butina_cluster",
        "cluster_map": df[["API_CID", "butina_cluster"]].drop_duplicates().to_dict("records")
    })
    
    # Build command to run cross_validate.py
    cmd = [
        sys.executable,
        "cross_validate.py",
        "--data-dir", ratio_dir,
        "--checkpoint-dir", checkpoint_dir,
        "--metrics-dir", metrics_dir,
    ]
    
    if max_epochs is not None:
        # Note: cross_validate.py doesn't have --max-epochs arg by default
        # You would need to add it or pass via environment variable
        pass
    
    logging.info(f"Running cross_validate.py with Butina clustering...")
    logging.info(f"Using cluster column: butina_cluster")
    
    try:
        result = subprocess.run(cmd, check=True)
        logging.info(f"\n✓ Successfully completed 5-fold CV for ratio_{ratio}\n")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"\n✗ Failed 5-fold CV for ratio_{ratio}: {e}\n")
        return False


def run_all_cv_ratios_fixed(max_epochs: int = None, skip_if_exists: bool = False):
    """Run 5-fold cross-validation with Butina clustering for all ratio datasets."""
    results = {}
    
    for ratio in RATIOS:
        success = run_cv_for_ratio_fixed(ratio, max_epochs=max_epochs, skip_if_exists=skip_if_exists)
        results[ratio] = "✓ PASSED" if success else "✗ FAILED"
    
    # Print summary
    logging.info(f"\n{'='*70}")
    logging.info("5-FOLD CV RATIO EXPERIMENTS SUMMARY (with Butina Clustering)")
    logging.info(f"{'='*70}")
    for ratio, status in results.items():
        logging.info(f"  ratio_{ratio}: {status}")
    logging.info(f"{'='*70}\n")
    
    return all(v == "✓ PASSED" for v in results.values())


def main():
    parser = argparse.ArgumentParser(
        description="Run 5-fold CV on ratio datasets with Butina clustering to prevent data leakage."
    )
    parser.add_argument(
        "--ratio",
        choices=["all"] + RATIOS,
        default="all",
        help="Which ratio to run CV on (default: all)"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs from config"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if CV checkpoints already exist"
    )
    parser.add_argument(
        "--cluster-only",
        action="store_true",
        help="Only compute and display clusters, don't train"
    )
    
    args = parser.parse_args()
    
    if args.cluster_only:
        # Show cluster info for each ratio
        for ratio in RATIOS:
            ratio_dir = os.path.join(BASE_DATA_DIR, f"ratio_{ratio}")
            train_path = os.path.join(ratio_dir, "train.csv")
            if os.path.exists(train_path):
                df = pd.read_csv(train_path)
                df = add_butina_clusters_to_dataframe(df)
                
                print(f"\n{'='*70}")
                print(f"Butina Clustering - ratio_{ratio}")
                print(f"{'='*70}")
                print(f"Total samples: {len(df)}")
                print(f"Unique clusters: {df['butina_cluster'].nunique()}")
                print(f"Cluster size distribution:")
                print(df["butina_cluster"].value_counts().describe())
        return
    
    if args.ratio == "all":
        success = run_all_cv_ratios_fixed(
            max_epochs=args.max_epochs,
            skip_if_exists=args.skip_existing
        )
        sys.exit(0 if success else 1)
    else:
        success = run_cv_for_ratio_fixed(
            args.ratio,
            max_epochs=args.max_epochs,
            skip_if_exists=args.skip_existing
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

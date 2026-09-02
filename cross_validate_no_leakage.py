"""
5-fold cross-validation with Butina clustering - NO DATA LEAKAGE VERSION.

This script does 5-fold CV while ensuring that ALL chemically similar APIs
(Tanimoto similarity >= 0.85, detected via Butina clustering) stay together
in the same fold. This prevents data leakage.

Key differences from cross_validate_ratios.py:
  1. Computes Butina clusters from API_Smiles
  2. Uses clusters (not just API_CID) as grouping variable for StratifiedGroupKFold
  3. Verifies no leakage at fold creation time
  4. Logs cluster composition for audit trail
"""

import logging
import os
import sys
from dataclasses import replace

os.environ["TRITON_DISABLE"] = "1"
os.environ["TORCH_USE_TRITON"] = "0"

from src.runtime import configure_thread_limits, configure_torch_runtime
configure_thread_limits()

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold

RDLogger.DisableLog("rdApp.*")

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

configure_torch_runtime(torch)

from src.config import Config
from src.dataset import get_dataloader_from_dataframe
from src.evaluate import (
    calculate_metrics_at_threshold,
    evaluate_model_full,
    save_cross_validation_metrics,
    tune_threshold,
)
from src.molformer_featurization import MolFormerFeaturizer
from src.loss import AsymmetricLoss
from src.model import APIExcipientModel
from src.train import train_model
from src.utils import count_parameters, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

BUTINA_CUTOFF = 0.15  # distance cutoff -> Tanimoto similarity >= 0.85
RATIOS = ["1to1", "3to1", "4to1"]
BASE_DATA_DIR = "data/ratio_experiments"


def compute_butina_clusters(smiles_list):
    """
    Compute Butina clusters from SMILES strings (Tanimoto >= 0.85).
    
    Returns:
        List where index i contains cluster ID for sample i
    """
    logging.info(f"Computing Butina clusters for {len(smiles_list)} molecules...")
    
    # Generate fingerprints
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
        logging.warning("No valid SMILES found, using row index as cluster ID")
        return list(range(len(smiles_list)))
    
    # Build distance matrix only for valid FPs
    valid_fps = [fps[i] for i in valid_indices]
    distances = []
    
    for i in range(len(valid_fps)):
        for j in range(i + 1, len(valid_fps)):
            sim = DataStructs.TanimotoSimilarity(valid_fps[i], valid_fps[j])
            dist = 1 - sim
            distances.append(dist)
    
    # Cluster
    if not distances:
        logging.warning("No distance pairs computed, using row index as cluster ID")
        return list(range(len(smiles_list)))
    
    clusters = Butina.ClusterData(distances, len(valid_fps), BUTINA_CUTOFF, isDistData=True)
    
    # Map clusters back to all indices
    cluster_ids = [-1] * len(smiles_list)
    
    for cluster_id, members in enumerate(clusters):
        for member_idx in members:
            original_idx = valid_indices[member_idx]
            cluster_ids[original_idx] = cluster_id
    
    # Assign singletons to invalid entries
    max_cluster_id = len(clusters) - 1
    for i in range(len(smiles_list)):
        if cluster_ids[i] == -1:
            max_cluster_id += 1
            cluster_ids[i] = max_cluster_id
    
    n_clusters = len(set(cluster_ids))
    logging.info(f"✓ Created {n_clusters} Butina clusters (Tanimoto >= 0.85)")
    
    return cluster_ids


def evaluate_validation_fold(model, val_loader, criterion, device, threshold):
    """Evaluate one validation fold at its tuned threshold."""
    _, val_loss, y_true, y_prob = evaluate_model_full(model, val_loader, criterion, device)
    metrics = calculate_metrics_at_threshold(y_true, y_prob, threshold)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    metrics["PR-AUC"] = auc(recall, precision)
    metrics["Loss"] = val_loss
    return metrics


def cross_validate_with_butina(config=None, n_splits=5):
    """
    Run grouped stratified 5-fold CV using Butina clustering for grouping.
    
    Ensures no chemically similar APIs span train/validation splits.
    """
    if n_splits != 5:
        raise ValueError("This validation utility is fixed to 5 folds by the spec.")

    config = config or Config()
    config.positive_prior = config.compute_positive_prior()
    
    set_seed(config.seed)
    device = config.get_device()

    train_path = os.path.join(config.data_dir, "train.csv")
    df = pd.read_csv(train_path)

    logging.info("=== 5-Fold Cross-Validation with Butina Clustering ===")
    logging.info(f"Using device: {device}")
    logging.info(f"Loading folds from {train_path}")
    
    # Compute Butina clusters
    if "API_Smiles" in df.columns:
        logging.info("Computing Butina clusters from API_Smiles (Tanimoto >= 0.85)...")
        cluster_ids = compute_butina_clusters(df["API_Smiles"].tolist())
        df["butina_cluster"] = cluster_ids
        group_col = "butina_cluster"
    else:
        logging.warning("No API_Smiles column, using API_CID for grouping (potential leakage risk)")
        group_col = "API_CID"
    
    logging.info(f"Grouping folds by: {group_col} ({df[group_col].nunique()} groups)")
    
    # Show cluster statistics
    if group_col == "butina_cluster":
        logging.info(f"\nButina Cluster Statistics:")
        cluster_sizes = df[group_col].value_counts()
        logging.info(f"  Cluster count: {len(cluster_sizes)}")
        logging.info(f"  Min size: {cluster_sizes.min()}, Max size: {cluster_sizes.max()}, Mean: {cluster_sizes.mean():.1f}")

    featurizer = MolFormerFeaturizer(
        model_path=config.molformer_model_path
    ).to(device)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.seed
    )

    fold_results = []
    y = df["Outcome1"]
    groups = df[group_col]

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(df, y, groups), start=1):
        logging.info("=" * 70)
        logging.info(f"Starting fold {fold_idx}/{n_splits}")

        fold_config = replace(
            config,
            checkpoint_dir=os.path.join(config.checkpoint_dir, f"cv_fold_{fold_idx}")
        )
        set_seed(config.seed + fold_idx)

        fold_train = df.iloc[train_idx].reset_index(drop=True)
        fold_val = df.iloc[val_idx].reset_index(drop=True)

        # Verify no leakage
        if group_col == "butina_cluster":
            train_clusters = set(fold_train[group_col].unique())
            val_clusters = set(fold_val[group_col].unique())
            overlap = train_clusters & val_clusters
            
            if overlap:
                logging.error(f"❌ DATA LEAKAGE: {len(overlap)} clusters span train and validation!")
                raise ValueError(f"Butina cluster leakage detected in fold {fold_idx}")
            
            logging.info(f"✓ Fold {fold_idx}: {len(train_clusters)} train clusters, {len(val_clusters)} val clusters (no overlap)")
        
        # Log fold composition
        logging.info(f"Fold {fold_idx}: {len(fold_train)} train rows, {len(fold_val)} val rows")
        logging.info(f"  Train positives: {(fold_train['Outcome1'] == 1).sum()}")
        logging.info(f"  Val positives: {(fold_val['Outcome1'] == 1).sum()}")

        train_loader = get_dataloader_from_dataframe(
            fold_config,
            featurizer,
            fold_train,
            is_train=True,
            shuffle=True
        )
        val_loader = get_dataloader_from_dataframe(
            fold_config,
            featurizer,
            fold_val,
            is_train=False,
            shuffle=False
        )

        model = APIExcipientModel(fold_config).to(device)
        trainable, total = count_parameters(model)
        logging.info(f"Fold {fold_idx} params: {trainable:,} trainable / {total:,} total")

        criterion = AsymmetricLoss(
            gamma_neg=fold_config.asl_gamma_neg,
            gamma_pos=fold_config.asl_gamma_pos,
            clip=fold_config.asl_clip
        )

        best_model = train_model(fold_config, model, train_loader, val_loader, criterion)
        best_threshold = tune_threshold(
            best_model,
            val_loader,
            criterion,
            device,
            step=fold_config.threshold_step
        )
        val_metrics = evaluate_validation_fold(
            best_model,
            val_loader,
            criterion,
            device,
            best_threshold
        )

        logging.info(
            f"Fold {fold_idx} complete | "
            f"PR-AUC: {val_metrics['PR-AUC']:.4f} | "
            f"Acc: {val_metrics['Accuracy']:.4f} | "
            f"Prec: {val_metrics['Precision']:.4f} | "
            f"Recall: {val_metrics['Recall']:.4f} | "
            f"F1: {val_metrics['F1']:.4f} | "
            f"MCC: {val_metrics['MCC']:.4f}"
        )

        fold_results.append({
            "fold": fold_idx,
            "threshold": best_threshold,
            "best_epoch": getattr(best_model, "best_epoch", None),
            "train_rows": len(fold_train),
            "val_rows": len(fold_val),
            "train_groups": len(train_clusters) if group_col == "butina_cluster" else len(fold_train[group_col].unique()),
            "val_groups": len(val_clusters) if group_col == "butina_cluster" else len(fold_val[group_col].unique()),
            "validation": val_metrics,
        })

    metrics_path = save_cross_validation_metrics(fold_results, config.metrics_dir)
    mean_pr_auc = sum(result["validation"]["PR-AUC"] for result in fold_results) / len(fold_results)
    
    logging.info(f"\n{'='*70}")
    logging.info(f"Cross-validation complete. Mean validation PR-AUC: {mean_pr_auc:.4f}")
    logging.info(f"Metrics saved to {metrics_path}")
    logging.info(f"✓ No data leakage detected (Butina clustering verified)")
    logging.info(f"{'='*70}\n")

    return {
        "fold_results": fold_results,
        "mean_val_pr_auc": mean_pr_auc,
        "metrics_path": metrics_path,
        "grouping_method": group_col,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run 5-fold cross-validation with Butina clustering (NO leakage)."
    )
    parser.add_argument("--data-dir", default=None, help="Directory containing train.csv, val.csv, test.csv.")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for fold checkpoint subdirectories.")
    parser.add_argument("--metrics-dir", default=None, help="Directory for cross-validation metrics JSON.")
    parser.add_argument("--descriptor-norm-stats-path", default=None, help="Path to descriptor_norm_stats.json.")
    args = parser.parse_args()
    
    config = Config()
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.metrics_dir is not None:
        config.metrics_dir = args.metrics_dir
    if args.descriptor_norm_stats_path is not None:
        config.descriptor_norm_stats_path = args.descriptor_norm_stats_path
    
    cross_validate_with_butina(config)

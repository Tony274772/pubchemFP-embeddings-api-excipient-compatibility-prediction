"""Configuration dataclass - all hyperparameters in one place."""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Data paths
    # ------------------------------------------------------------------ #
    data_dir: str = "data"
    train_csv: str = None
    val_csv: str = None
    test_csv: str = None
    molformer_model_path: str = "models/molformer-xl-both-10pct"
    molformer_dim: int = 768

    # ------------------------------------------------------------------ #
    # Model architecture
    # ------------------------------------------------------------------ #
    proj_dim: int = 128
    num_heads: int = 8
    attn_dropout: float = 0.15
    proj_dropout: float = 0.15
    clf_dropout_1: float = 0.5
    clf_dropout_2: float = 0.4
    clf_hidden_dim: int = 128
    clf_hidden_dim_2: int = 64
    num_descriptors: int = 21
    desc_proj_dim: int = 24
    desc_dropout: float = 0.15
    use_descriptors: bool = True
    api_descriptors_path: str = "data/api_descriptors.csv"
    excipient_descriptors_path: str = "data/excipient_descriptors.csv"
    descriptor_norm_stats_path: str = "models/descriptor_norm_stats.json"

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    lr: float = 1.5e-4
    weight_decay: float = 8e-4
    grad_clip_norm: float = 1.0
    batch_size: int = 64
    max_epochs: int = 150
    early_stop_patience: int = 6
    early_stop_min_delta: float = 0.002
    lr_patience: int = 8
    lr_factor: float = 0.5

    # Kept for future missing-SMILES data, inert for the current clean splits.
    modality_dropout_rate: float = 0.0

    # ------------------------------------------------------------------ #
    # ASL loss
    # ------------------------------------------------------------------ #
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05

    # ------------------------------------------------------------------ #
    # Class imbalance handling
    # ------------------------------------------------------------------ #
    # Positive class prior computed from data/train.csv: 191 positives / 2030 total rows
    positive_prior: float = 0.0941
    # Use WeightedRandomSampler to balance batch composition
    use_balanced_sampler: bool = True
    # SMILES randomization augmentation for positive-class training rows only
    smiles_augment_positive_class: bool = False
    # Number of distinct random SMILES variants to cache per molecule
    smiles_augment_n_variants: int = 5

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    threshold_step: float = 0.001

    # ------------------------------------------------------------------ #
    # Reproducibility / system
    # ------------------------------------------------------------------ #
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "checkpoints/molformer"
    metrics_dir: str = "metrics/molformer"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def get_train_csv_path(self) -> str:
        import os
        return self.train_csv if self.train_csv else os.path.join(self.data_dir, "train.csv")

    def get_val_csv_path(self) -> str:
        import os
        return self.val_csv if self.val_csv else os.path.join(self.data_dir, "val.csv")

    def get_test_csv_path(self) -> str:
        import os
        return self.test_csv if self.test_csv else os.path.join(self.data_dir, "test.csv")

    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def compute_positive_prior(self) -> float:
        """Compute positive class prior from the active training data.
        
        Reads the training CSV and computes the fraction of positive
        examples. Used for subset-aware bias initialization in the classifier.
        If file doesn't exist, returns the default prior (0.0941).
        """
        import os
        import pandas as pd
        
        train_path = self.get_train_csv_path()
        if not os.path.exists(train_path):
            return self.positive_prior  # Return default if file doesn't exist
        
        try:
            df = pd.read_csv(train_path)
            n_positive = (df["Outcome1"] == 1).sum()
            n_total = len(df)
            if n_total > 0:
                return n_positive / n_total
        except Exception:
            pass
        
        return self.positive_prior  # Return default on any error

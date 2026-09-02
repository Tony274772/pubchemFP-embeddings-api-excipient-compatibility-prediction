"""
PyTorch Dataset and DataLoader factory.

Loads train/val/test CSVs, applies modality dropout to excipients during training.
Note: Since the dataset is small (~2k rows), and transformer featurization can
be somewhat slow, we'll rely on the caching inside MolFormerFeaturizer.
To make it fast, the dataset just returns raw strings and labels, and a custom
collate function calls the featurizer.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random

from src.descriptors import DescriptorLookup


def generate_smiles_variants(smiles: str, n_variants: int) -> list:
    """Generate up to n_variants distinct chemically valid random SMILES for a molecule.

    Uses RDKit's non-canonical (doRandom=True) SMILES output. If the input cannot
    be parsed, or no variant is produced, the original string is returned unchanged.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    smiles = str(smiles)
    if not smiles.strip():
        return [smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    variants = {smiles}
    max_attempts = n_variants * 10
    attempts = 0
    while len(variants) < n_variants and attempts < max_attempts:
        variants.add(Chem.MolToSmiles(mol, doRandom=True))
        attempts += 1
    return list(variants)


class CompatibilityDataset(Dataset):
    def __init__(self, csv_path: str = None, df: pd.DataFrame = None, is_train: bool = False, modality_dropout_rate: float = 0.2, smiles_augment_positive_class: bool = False, smiles_augment_n_variants: int = 5):
        """
        Args:
            csv_path: path to train.csv, val.csv, or test.csv
            df: in-memory dataframe, used for cross-validation folds
            is_train: if True, applies modality dropout dynamically
            modality_dropout_rate: prob to simulate missing excipient SMILES
            smiles_augment_positive_class: if True, augments positive-class train rows
            smiles_augment_n_variants: number of random SMILES variants to cache per molecule
        """
        if df is None and csv_path is None:
            raise ValueError("Either csv_path or df must be provided.")
        self.df = df.reset_index(drop=True) if df is not None else pd.read_csv(csv_path)
        self.is_train = is_train
        self.modality_dropout_rate = modality_dropout_rate
        self.smiles_augment_positive_class = smiles_augment_positive_class
        self.smiles_augment_n_variants = smiles_augment_n_variants

        # Precompute random SMILES variants once per unique molecule used in
        # positive-class training rows. Descriptor lookup keys off CID, not
        # SMILES, so swapping to a variant does not break descriptors.
        self._smiles_variant_cache = {}
        if self.is_train and self.smiles_augment_positive_class:
            positive_mask = self.df["Outcome1"] == 1
            if positive_mask.any():
                positive_rows = self.df.loc[positive_mask]
                unique_smiles = set()
                for col in ("API_Smiles", "Excipient_Smiles"):
                    unique_smiles.update(positive_rows[col].dropna().astype(str).unique())
                for smi in unique_smiles:
                    self._smiles_variant_cache[smi] = generate_smiles_variants(
                        smi, self.smiles_augment_n_variants
                    )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        api_smi = row["API_Smiles"]
        exc_smi = row["Excipient_Smiles"]
        label = float(row["Outcome1"])
        
        # Missing-flag: 1 = SMILES available, 0 = absent/dropped
        # For start_dataset.csv, all have SMILES.
        # But we simulate missing ones during training.
        exc_smiles_available = 1.0
        
        # If exc_smi is literally missing in data (e.g. NaN) - handle just in case
        if pd.isna(exc_smi) or str(exc_smi).strip() == "":
            exc_smiles_available = 0.0
            exc_smi = "" # Empty string for featurizer (will return 0 tokens)
        elif self.is_train and random.random() < self.modality_dropout_rate:
            # Modality dropout: pretend structure is unavailable, but keep the
            # real SMILES in the batch. The model uses exc_smiles_available=0 to
            # replace the structural branch with its learned placeholder, so the
            # featurizer should never see a fake empty SMILES for valid data.
            exc_smiles_available = 0.0

        # SMILES randomization augmentation: positive-class training rows only.
        # Val/test rows and negative-class train rows keep exact original SMILES.
        if self.is_train and self.smiles_augment_positive_class and label == 1.0:
            api_variants = self._smiles_variant_cache.get(str(api_smi))
            if api_variants:
                api_smi = random.choice(api_variants)
            if exc_smi:
                exc_variants = self._smiles_variant_cache.get(str(exc_smi))
                if exc_variants:
                    exc_smi = random.choice(exc_variants)

        return {
            "api_cid": row["API_CID"],
            "exc_cid": row["Excipient_CID"],
            "api_smi": api_smi,
            "exc_smi": exc_smi,
            "exc_smiles_available": exc_smiles_available,
            "label": label
        }

def create_collate_fn(featurizer, descriptor_lookup=None):
    """Creates a collate function that uses the provided featurizer."""
    
    def collate_fn(batch):
        api_smiles = [item["api_smi"] for item in batch]
        exc_smiles = [item["exc_smi"] for item in batch]
        
        exc_available = torch.tensor([item["exc_smiles_available"] for item in batch], dtype=torch.float32)
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        
        # Featurize
        api_padded, api_global, api_mask, api_num = featurizer(api_smiles)
        exc_padded, exc_global, exc_mask, exc_num = featurizer(exc_smiles)
        
        batch_dict = {
            "api_tokens": api_padded,
            "api_global": api_global,
            "api_mask": api_mask,
            "api_num": api_num,
            
            "exc_tokens": exc_padded,
            "exc_global": exc_global,
            "exc_mask": exc_mask,
            "exc_num": exc_num,
            
            "exc_available": exc_available,
            "labels": labels
        }

        if descriptor_lookup is not None:
            batch_dict["api_desc"] = torch.tensor(
                [descriptor_lookup.get_api(item["api_cid"]) for item in batch],
                dtype=torch.float32
            )
            batch_dict["exc_desc"] = torch.tensor(
                [descriptor_lookup.get_exc(item["exc_cid"]) for item in batch],
                dtype=torch.float32
            )

        return batch_dict
        
    return collate_fn


def create_balanced_sampler(dataset):
    """Create a WeightedRandomSampler that balances class composition in each batch.
    
    Computes per-sample weights as 1.0 / class_count[label] for each row,
    ensuring minority class samples appear with higher probability.
    
    Args:
        dataset: CompatibilityDataset instance
        
    Returns:
        WeightedRandomSampler with weights computed from labels
    """
    labels = dataset.df["Outcome1"].values
    
    # Count positives and negatives
    num_positives = (labels == 1).sum()
    num_negatives = (labels == 0).sum()
    
    # Assign weight: 1.0 / class_count[label]
    weights = []
    for label in labels:
        if label == 1:
            weight = 1.0 / num_positives
        else:
            weight = 1.0 / num_negatives
        weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )
    return sampler


def create_descriptor_lookup(config):
    if not config.use_descriptors:
        return None
    return DescriptorLookup(
        config.api_descriptors_path,
        config.excipient_descriptors_path,
        config.descriptor_norm_stats_path
    )

def get_dataloaders(config, featurizer):
    """Return train, val, and test dataloaders."""
    
    train_dataset = CompatibilityDataset(
        csv_path=config.get_train_csv_path(), 
        is_train=True, 
        modality_dropout_rate=config.modality_dropout_rate,
        smiles_augment_positive_class=config.smiles_augment_positive_class,
        smiles_augment_n_variants=config.smiles_augment_n_variants
    )
    val_dataset = CompatibilityDataset(
        csv_path=config.get_val_csv_path(), 
        is_train=False
    )
    test_dataset = CompatibilityDataset(
        csv_path=config.get_test_csv_path(), 
        is_train=False
    )
    
    descriptor_lookup = create_descriptor_lookup(config)
    collate = create_collate_fn(featurizer, descriptor_lookup)
    
    # Train loader: use balanced sampler if enabled, otherwise shuffle
    if config.use_balanced_sampler:
        train_sampler = create_balanced_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            sampler=train_sampler, 
            collate_fn=collate,
            drop_last=False
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            collate_fn=collate,
            drop_last=False
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def get_dataloader_from_dataframe(config, featurizer, df, is_train=False, shuffle=False):
    """Return a dataloader for an in-memory dataframe."""
    dataset = CompatibilityDataset(
        df=df,
        is_train=is_train,
        modality_dropout_rate=config.modality_dropout_rate,
        smiles_augment_positive_class=config.smiles_augment_positive_class,
        smiles_augment_n_variants=config.smiles_augment_n_variants
    )
    descriptor_lookup = create_descriptor_lookup(config)
    collate = create_collate_fn(featurizer, descriptor_lookup)

    # Use balanced sampler for training data if enabled, otherwise use shuffle parameter
    if is_train and config.use_balanced_sampler:
        sampler = create_balanced_sampler(dataset)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=collate,
            drop_last=False
        )
    else:
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            drop_last=False
        )

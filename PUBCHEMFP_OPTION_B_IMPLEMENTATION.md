# Implementation Spec: PubChemFP Flat-Vector Encoder Ablation ("Option B")

## Audience

This file is written for an AI coding agent (or human engineer) that has
read/write access to this repository but has **not** seen any prior
conversation about it. Everything you need to implement this change without
guessing is below. If something is genuinely ambiguous, stop and ask rather
than assume — but most decisions have already been made explicit on purpose.

## Objective

Add a **new, additive** model variant that replaces the current MoLFormer-XL
token-embedding + cross-attention encoder with a **PubChem Fingerprint (881-bit
binary vector) encoder**, using a **flat MLP projection with no attention
mechanism** — i.e. no token sequences, no cross-attention between API and
excipient. This should let us empirically compare "MoLFormer contextual
embeddings + cross-attention" vs. "PubChem structural-fragment bits + plain
MLP" while holding the rest of the pipeline (interaction features, classifier
head, loss, sampler, training loop, evaluation) constant.

This is Option B in a three-option design discussion (Option A = treat active
fingerprint bits as attention tokens; Option B = this, flat vector, no
attention; Option C = ensemble both encoders). **Only implement Option B.**

## Hard constraints — do not violate these

1. **Do not modify or delete** the existing MoLFormer pipeline: `main.py`,
   `src/model.py`, `src/molformer_featurization.py`, `src/dataset.py`'s
   existing functions (`CompatibilityDataset`, `create_collate_fn`,
   `create_balanced_sampler`, `get_dataloaders`, `get_dataloader_from_dataframe`),
   `src/config.py`'s existing fields, `src/descriptors.py`. All new
   functionality must be **additive** (new files, new functions, new config
   fields with safe defaults) so the existing MoLFormer training/eval run
   (`python main.py`) continues to work byte-for-byte identically after your
   changes.
2. Reuse the existing `_projection(in_dim, hidden_dim, out_dim, dropout)`
   helper from `src/model.py` (import it, do not redefine it).
3. Reuse the existing `AsymmetricLoss` (`src/loss.py`), `train_model`
   (`src/train.py`), and all of `src/evaluate.py` unchanged. These only
   depend on `model(batch) -> logits` and `batch["labels"]`, so they work
   with any encoder as long as your new model's `forward()` returns a
   `[batch_size]` tensor of raw logits.
4. Do not touch `data/train.csv`, `data/val.csv`, `data/test.csv`, or any
   existing descriptor CSVs. You will only **add** new CSV files.
5. Preserve column names exactly as they exist in the CSVs today:
   `API_CID`, `Excipient_CID`, `API_Smiles`, `Excipient_Smiles`, `Outcome1`.

## Background you need (do not skip)

- The dataset CSVs (`data/train.csv`, `data/val.csv`, `data/test.csv`) have
  columns `API_CID, Excipient_CID, Outcome1, API_Smiles, Excipient_Smiles`.
  `API_CID`/`Excipient_CID` are **PubChem Compound IDs (CIDs)** — integers.
- `src/descriptors.py` already implements exactly the lookup pattern you must
  mirror for fingerprints: a CSV keyed by CID, loaded into a dict, with
  `get_api(cid)` / `get_exc(cid)` accessor methods that raise `ValueError` on
  a missing CID (fail loudly, never silently zero-fill).
- The current `src/model.py` (`APIExcipientModel`) computes, per pair:
  - `h_api` = structural embedding (proj_dim) [+ RDKit descriptor embedding]
  - `h_exc` = structural embedding (proj_dim) [+ RDKit descriptor embedding] + `exc_available` flag
  - `interaction = h_api * h_exc[:, :dim_api]`
  - `difference = |h_api - h_exc[:, :dim_api]|`
  - `pair_vec = concat([h_api, h_exc, interaction, difference])` → classifier MLP → 1 logit
  - Missing excipient (`exc_available == 0`) is handled by substituting a
    learned placeholder vector instead of the real structural embedding.
  You are replacing **only** how the "structural embedding" is computed (from
  MoLFormer token attention → from a PubChem fingerprint MLP), and dropping
  the attention layers entirely. Everything downstream (interaction,
  difference, classifier, placeholder-for-missing-excipient logic) is
  conceptually the same and should be re-implemented in the new model file
  with the same shapes so the comparison is fair.

## Step 1 — Generate PubChem fingerprints (new script, new data files)

Create `scripts/generate_pubchem_fp.py`.

**What it does:**
1. Reads `data/train.csv`, `data/val.csv`, `data/test.csv`.
2. Collects the set of unique `API_CID` values and, separately, the set of
   unique `Excipient_CID` values across all three files.
3. For each unique CID, fetches the **official PubChem CACTVS fingerprint**
   (the exact 881-bit fingerprint described in Table 2 of the DE-Interact
   paper — hierarchic element counts, ESSSR rings, atom pairs, atom
   neighborhoods, SMARTS patterns) using the `pubchempy` library:

   ```python
   import pubchempy as pcp
   compound = pcp.Compound.from_cid(int(cid))
   fp_string = compound.cactvs_fingerprint  # must be an 881-character string of '0'/'1'
   ```

   **Validate immediately after fetching**: `assert len(fp_string) == 881` and
   `assert set(fp_string) <= {"0", "1"}`. If this assertion fails on your
   installed `pubchempy` version, that means the library is returning a
   different encoding (e.g. still base64) — in that case decode it yourself
   per PubChem's fingerprint spec (skip the 32-bit length prefix, take the
   next 881 bits) until the assertion passes. Do not proceed past this
   validation with silently-wrong data.

4. **Rate limiting**: PubChem's PUG-REST API asks for no more than 5
   requests/second. Add `time.sleep(0.25)` between requests and wrap each
   fetch in a `try/except` with up to 3 retries (exponential backoff:
   0.5s, 1s, 2s) before giving up on a CID. If a CID ultimately fails after
   retries, log it clearly to stderr and continue — **do not** silently drop
   it or fabricate a zero vector; collect the list of failed CIDs and print
   it at the end so a human can investigate.
5. Cache as you go: write results incrementally (e.g. append to a temp
   file or checkpoint every N CIDs) so a network interruption doesn't lose
   progress on a run over thousands of CIDs. A simple approach: check if the
   output CSV already exists and already contains a given CID before
   re-fetching it (idempotent / resumable script).
6. Output two CSVs, mirroring the exact style of `data/api_descriptors.csv`
   (comma-separated, one row per unique CID, first column is the CID column
   name, followed by 881 bit columns named `fp_0000` through `fp_0880`,
   zero-padded 4 digits, values are integers `0`/`1`):

   - `data/api_pubchemfp.csv` — header: `API_CID,fp_0000,fp_0001,...,fp_0880`
   - `data/excipient_pubchemfp.csv` — header: `Excipient_CID,fp_0000,fp_0001,...,fp_0880`

**CLI**: no arguments required; running `python scripts/generate_pubchem_fp.py`
from the repo root should do the whole job. Print a final summary: total
unique API CIDs, total unique excipient CIDs, how many succeeded, how many
failed (with the failed CID list).

## Step 2 — Fingerprint lookup class (new file)

Create `src/pubchem_fp.py`. Mirror `src/descriptors.py`'s `DescriptorLookup`
class structure exactly, but simpler (no normalization — fingerprint bits are
already 0/1, do not mean/std-normalize them):

```python
class PubChemFPLookup:
    def __init__(self, api_csv_path: str, exc_csv_path: str):
        # load both CSVs into dicts keyed by str(cid) -> np.ndarray shape (881,), dtype float32
        ...

    def get_api(self, api_cid) -> np.ndarray:
        # raise ValueError(f"Missing API PubChemFP row for CID {api_cid}") if not found
        ...

    def get_exc(self, exc_cid) -> np.ndarray:
        # raise ValueError(f"Missing excipient PubChemFP row for CID {exc_cid}") if not found
        ...
```

Also add a module-level constant `PUBCHEM_FP_DIM = 881` in this file.

## Step 3 — Config additions

In `src/config.py`, add the following **new** fields to the existing
`Config` dataclass (do not remove or rename anything already there):

```python
    # ------------------------------------------------------------------ #
    # PubChemFP flat-vector encoder (Option B ablation)
    # ------------------------------------------------------------------ #
    pubchem_fp_dim: int = 881
    fp_proj_hidden_dim: int = 256      # mirrors the hidden size used in _projection() calls elsewhere
    api_pubchemfp_path: str = "data/api_pubchemfp.csv"
    excipient_pubchemfp_path: str = "data/excipient_pubchemfp.csv"
```

Reuse the existing `proj_dim`, `desc_proj_dim`, `num_descriptors`,
`use_descriptors`, `clf_hidden_dim`, `clf_hidden_dim_2`, `clf_dropout_1`,
`clf_dropout_2`, `proj_dropout`, `positive_prior`, loss/training/eval fields
as-is — the new model must read these from the same `Config` object so
hyperparameters are shared/comparable with the MoLFormer run wherever
architecturally meaningful.

## Step 4 — Dataset / collate additions

In `src/dataset.py`, **add** (do not modify existing functions) a new collate
factory and a new dataloader factory:

```python
def create_fp_collate_fn(fp_lookup, descriptor_lookup=None):
    """Collate function for the PubChemFP flat-vector model. No featurizer/model
    forward pass needed here — fingerprints are precomputed and looked up by CID."""

    def collate_fn(batch):
        exc_available = torch.tensor([item["exc_smiles_available"] for item in batch], dtype=torch.float32)
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

        api_fp = torch.tensor(
            np.stack([fp_lookup.get_api(item["api_cid"]) for item in batch]),
            dtype=torch.float32,
        )
        # NOTE: always look up the real excipient fingerprint by CID, exactly
        # like the existing collate_fn does for exc_desc — the *model*, not
        # the collate function, is responsible for swapping in the missing-
        # excipient placeholder based on exc_available. Do not zero this out
        # here.
        exc_fp = torch.tensor(
            np.stack([fp_lookup.get_exc(item["exc_cid"]) for item in batch]),
            dtype=torch.float32,
        )

        batch_dict = {
            "api_fp": api_fp,
            "exc_fp": exc_fp,
            "exc_available": exc_available,
            "labels": labels,
        }

        if descriptor_lookup is not None:
            batch_dict["api_desc"] = torch.tensor(
                [descriptor_lookup.get_api(item["api_cid"]) for item in batch],
                dtype=torch.float32,
            )
            batch_dict["exc_desc"] = torch.tensor(
                [descriptor_lookup.get_exc(item["exc_cid"]) for item in batch],
                dtype=torch.float32,
            )

        return batch_dict

    return collate_fn


def get_pubchemfp_dataloaders(config, fp_lookup):
    """Same structure/semantics as get_dataloaders(), but uses create_fp_collate_fn
    instead of create_collate_fn (no MolFormerFeaturizer involved)."""
    train_dataset = CompatibilityDataset(
        csv_path=config.get_train_csv_path(),
        is_train=True,
        modality_dropout_rate=config.modality_dropout_rate,
        smiles_augment_positive_class=config.smiles_augment_positive_class,
        smiles_augment_n_variants=config.smiles_augment_n_variants,
    )
    val_dataset = CompatibilityDataset(csv_path=config.get_val_csv_path(), is_train=False)
    test_dataset = CompatibilityDataset(csv_path=config.get_test_csv_path(), is_train=False)

    descriptor_lookup = create_descriptor_lookup(config)
    collate = create_fp_collate_fn(fp_lookup, descriptor_lookup)

    if config.use_balanced_sampler:
        train_sampler = create_balanced_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                   sampler=train_sampler, collate_fn=collate, drop_last=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                   shuffle=True, collate_fn=collate, drop_last=False)

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=collate, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                              shuffle=False, collate_fn=collate, drop_last=False)

    return train_loader, val_loader, test_loader
```

Note: `CompatibilityDataset.__getitem__` already returns `api_cid`, `exc_cid`,
`exc_smiles_available`, `label` — you do not need to modify
`CompatibilityDataset` at all. The `api_smi`/`exc_smi` fields it also returns
are simply unused by this new collate function, which is fine.

You will need `import numpy as np` at the top of `src/dataset.py` if it is
not already imported — check first before adding a duplicate import.

## Step 5 — New model (new file)

Create `src/model_fp.py`:

```python
"""PubChemFP flat-vector API/excipient compatibility model (no attention)."""

import math
import torch
import torch.nn as nn

from src.model import _projection  # reuse existing helper, do not redefine


class APIExcipientFPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        fp_dim = config.pubchem_fp_dim          # 881
        proj_dim = config.proj_dim              # same proj_dim as the MoLFormer model, for comparability
        hidden = config.fp_proj_hidden_dim

        self.api_proj = _projection(fp_dim, hidden, proj_dim, config.proj_dropout)
        self.exc_proj = _projection(fp_dim, hidden, proj_dim, config.proj_dropout)

        # Single learned placeholder vector for a missing excipient structure
        # (no token dimension here, unlike the MoLFormer model's per-token placeholders).
        self.exc_missing_placeholder = nn.Parameter(torch.randn(1, proj_dim) * 0.01)

        if config.use_descriptors:
            self.api_desc_proj = _projection(
                config.num_descriptors, 32, config.desc_proj_dim, config.desc_dropout
            )
            self.exc_desc_proj = _projection(
                config.num_descriptors, 32, config.desc_proj_dim, config.desc_dropout
            )

        dim_api = proj_dim
        dim_exc = proj_dim + 1
        if config.use_descriptors:
            dim_api = proj_dim + config.desc_proj_dim
            dim_exc = proj_dim + config.desc_proj_dim + 1
        self.dim_api = dim_api

        pair_dim = dim_api + dim_exc + dim_api + dim_api

        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, config.clf_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.clf_dropout_1),
            nn.Linear(config.clf_hidden_dim, config.clf_hidden_dim_2),
            nn.GELU(),
            nn.Dropout(config.clf_dropout_2),
            nn.Linear(config.clf_hidden_dim_2, 1),
        )

        final_linear_layer = self.classifier[-1]
        log_odds = math.log(config.positive_prior / (1.0 - config.positive_prior))
        torch.nn.init.constant_(final_linear_layer.bias, log_odds)

    def forward(self, batch):
        api_fp = batch["api_fp"]           # [B, 881]
        exc_fp = batch["exc_fp"]           # [B, 881]
        exc_avail = batch["exc_available"].unsqueeze(1)  # [B, 1]

        h_api_struct = self.api_proj(api_fp)   # [B, proj_dim]

        exc_struct_real = self.exc_proj(exc_fp)  # [B, proj_dim]
        missing_rows = (exc_avail == 0.0)        # [B, 1]
        h_exc_struct = torch.where(
            missing_rows,
            self.exc_missing_placeholder.expand(exc_struct_real.size(0), -1),
            exc_struct_real,
        )

        if self.config.use_descriptors:
            d_api = self.api_desc_proj(batch["api_desc"])
            d_exc = self.exc_desc_proj(batch["exc_desc"])
            h_api = torch.cat([h_api_struct, d_api], dim=-1)
            h_exc = torch.cat([h_exc_struct, d_exc, exc_avail], dim=-1)
        else:
            h_api = h_api_struct
            h_exc = torch.cat([h_exc_struct, exc_avail], dim=-1)

        h_exc_core = h_exc[:, :self.dim_api]
        interaction = h_api * h_exc_core
        difference = torch.abs(h_api - h_exc_core)

        pair_vec = torch.cat([h_api, h_exc, interaction, difference], dim=-1)
        logits = self.classifier(pair_vec)
        return logits.squeeze(1)
```

This is intentionally structurally parallel to `APIExcipientModel` in
`src/model.py` (same `pair_dim` formula, same classifier shape, same bias
initialization trick) so that any performance difference you measure is
attributable to the encoder (attention over MoLFormer tokens vs. flat FP MLP),
not to incidental architecture differences elsewhere.

## Step 6 — New entry-point script

Create `main_pubchemfp.py` at the repo root, cloned from `main.py`'s
structure but wired to the new components. Exact required behavior:

```python
"""Entry point for training/evaluating the PubChemFP flat-vector ablation model."""

import logging
import sys
import argparse
from src.runtime import configure_thread_limits, configure_torch_runtime

configure_thread_limits()

import torch
configure_torch_runtime(torch)

from src.config import Config
from src.utils import set_seed, count_parameters
from src.pubchem_fp import PubChemFPLookup
from src.dataset import get_pubchemfp_dataloaders
from src.model_fp import APIExcipientFPModel
from src.loss import AsymmetricLoss
from src.train import train_model
from src.evaluate import (
    tune_threshold, run_final_evaluation, print_run_summary, save_run_metrics,
    calculate_metrics_at_threshold, evaluate_model_full, auc, precision_recall_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate the PubChemFP ablation model.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/pubchemfp")
    parser.add_argument("--metrics-dir", default="metrics/pubchemfp")
    parser.add_argument("--descriptor-norm-stats-path", default=None)
    parser.add_argument("--api-pubchemfp-path", default=None)
    parser.add_argument("--excipient-pubchemfp-path", default=None)
    args = parser.parse_args()

    config = Config()
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.train_csv is not None:
        config.train_csv = args.train_csv
    if args.val_csv is not None:
        config.val_csv = args.val_csv
    if args.test_csv is not None:
        config.test_csv = args.test_csv
    config.checkpoint_dir = args.checkpoint_dir
    config.metrics_dir = args.metrics_dir
    if args.descriptor_norm_stats_path is not None:
        config.descriptor_norm_stats_path = args.descriptor_norm_stats_path
    if args.api_pubchemfp_path is not None:
        config.api_pubchemfp_path = args.api_pubchemfp_path
    if args.excipient_pubchemfp_path is not None:
        config.excipient_pubchemfp_path = args.excipient_pubchemfp_path

    config.positive_prior = config.compute_positive_prior()

    set_seed(config.seed)
    device = config.get_device()

    logging.info("=== API-Excipient Compatibility Prediction (PubChemFP ablation) ===")
    logging.info(f"Using device: {device}")

    fp_lookup = PubChemFPLookup(config.api_pubchemfp_path, config.excipient_pubchemfp_path)

    logging.info("Initializing datasets...")
    train_loader, val_loader, test_loader = get_pubchemfp_dataloaders(config, fp_lookup)
    logging.info(f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")

    model = APIExcipientFPModel(config).to(device)
    trainable, total = count_parameters(model)
    logging.info(f"Model initialized. Trainable params: {trainable:,} / Total params: {total:,}")

    criterion = AsymmetricLoss(
        gamma_neg=config.asl_gamma_neg,
        gamma_pos=config.asl_gamma_pos,
        clip=config.asl_clip,
    )

    best_model = train_model(config, model, train_loader, val_loader, criterion)

    best_threshold = tune_threshold(best_model, val_loader, criterion, device, step=config.threshold_step)

    _, val_loss, val_true, val_prob = evaluate_model_full(best_model, val_loader, criterion, device)
    val_metrics = calculate_metrics_at_threshold(val_true, val_prob, best_threshold)
    precision, recall, _ = precision_recall_curve(val_true, val_prob)
    val_metrics["PR-AUC"] = auc(recall, precision)
    val_metrics["Loss"] = val_loss

    test_metrics = run_final_evaluation(best_model, test_loader, criterion, device, best_threshold)

    best_epoch = getattr(best_model, "best_epoch", None)
    print_run_summary(val_metrics, test_metrics, best_threshold, best_epoch=best_epoch)
    save_run_metrics(val_metrics, test_metrics, best_threshold, config.metrics_dir, best_epoch=best_epoch)


if __name__ == "__main__":
    main()
```

Note the `--checkpoint-dir`/`--metrics-dir` defaults are already set to
`checkpoints/pubchemfp` / `metrics/pubchemfp` so this run's artifacts never
collide with the existing `checkpoints/molformer` / `metrics/molformer`
outputs.

## Step 7 — Dependencies

Add `pubchempy` to `requirements.txt` if it is not already present (check
first — do not add a duplicate line).

## Step 8 — Sanity checks before a full training run

Do these in order, and stop/report if any fails:

1. Run `python scripts/generate_pubchem_fp.py` to completion. Confirm
   `data/api_pubchemfp.csv` and `data/excipient_pubchemfp.csv` exist, each
   row has exactly 882 columns (1 CID column + 881 bit columns), and every
   `API_CID`/`Excipient_CID` that appears in `data/train.csv`,
   `data/val.csv`, `data/test.csv` has a corresponding row (no gaps).
2. Write a small standalone smoke test (you may add a new file
   `smoke_test_fp.py`, modeled after the existing `smoke_test.py` — inspect
   it first for the pattern) that: loads `Config`, loads `PubChemFPLookup`,
   builds one batch via `get_pubchemfp_dataloaders`, runs it through a freshly
   constructed `APIExcipientFPModel`, and asserts the output logits tensor
   has shape `[batch_size]` with no NaNs.
3. Only after the smoke test passes, run `python main_pubchemfp.py` for a
   real training run.

## Definition of done

- [ ] `scripts/generate_pubchem_fp.py` exists and produces valid, complete
      `data/api_pubchemfp.csv` and `data/excipient_pubchemfp.csv`.
- [ ] `src/pubchem_fp.py` exists with `PubChemFPLookup` implementing
      `get_api`/`get_exc` that raise on missing CIDs.
- [ ] `src/config.py` has the 4 new fields listed in Step 3, with all
      pre-existing fields untouched.
- [ ] `src/dataset.py` has the 2 new functions listed in Step 4 (
      `create_fp_collate_fn`, `get_pubchemfp_dataloaders`), with all
      pre-existing functions untouched.
- [ ] `src/model_fp.py` exists with `APIExcipientFPModel` as specified in
      Step 5, importing `_projection` from `src/model.py` rather than
      duplicating it.
- [ ] `main_pubchemfp.py` exists and runs end-to-end, writing metrics to
      `metrics/pubchemfp/run_metrics.json` and a checkpoint to
      `checkpoints/pubchemfp/best_model.pt`.
- [ ] Running `python main.py` (the original MoLFormer pipeline) still works
      exactly as before — no regressions.
- [ ] Final report to the user should compare
      `metrics/pubchemfp/run_metrics.json` against the existing
      `metrics/molformer/run_metrics.json` on PR-AUC, F1, MCC, Precision,
      Recall (the metrics already computed by `src/evaluate.py` for both
      runs, so no new metric code is needed).

## Explicitly out of scope for this task

- Do not implement Option A (fragment-as-attention-token) or Option C
  (ensemble) — only Option B.
- Do not attempt to locally approximate the PubChem CACTVS fingerprint
  algorithm with RDKit SMARTS matching — fetch the real fingerprint from
  PubChem via `pubchempy` as specified in Step 1. The CIDs are already known
  and present in the dataset, so this is a straightforward one-time fetch.
- Do not change `proj_dim`, `clf_hidden_dim`, `clf_hidden_dim_2`, or any
  other shared hyperparameter defaults in `Config` — keep them identical to
  the MoLFormer run so the comparison is fair. If you believe a different
  value is warranted for the FP model specifically, add it as a *new*,
  separately-named config field rather than repurposing an existing one.

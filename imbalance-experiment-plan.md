# Imbalance-Handling Experiment Plan — mlp-molformer-asl

## Purpose of this document

This repo (`mlp-molformer-asl`) trains an API–Excipient compatibility classifier
on a heavily imbalanced dataset (~9.7% positive / "incompatible" class, 344
positives out of 3,544 total pairs). The current pipeline already uses a
group-safe API-level split (`data/data_split.py`) and Asymmetric Loss (`src/loss.py`)
to address imbalance. This document specifies **five follow-up experiments**,
in a fixed order, meant to reduce training/evaluation variance and improve
minority-class performance further.

## Rules for the coding agent implementing this

1. **Implement exactly ONE numbered experiment per session, in the order given below.** Do not jump ahead, do not combine two experiments in one pass, even if it looks efficient.
2. **After implementing an experiment's code changes, STOP. Do not run training.** Do not invoke `main.py`, `cross_validate.py`, `sweep_asl.py`, or any script that trains a model. Do not run smoke tests that involve a training loop. Training and interpreting metrics is done manually by the user, in between agent sessions.
3. After implementing an experiment, give the user a short summary of exactly what changed (files + lines), and explicitly tell them: *"This experiment is implemented. Please train and evaluate it yourself now. Come back and tell me the results/ask me to proceed when ready."*
4. Do not modify or delete any existing checkpoints, metrics JSON files, or data CSVs. Every change must be additive/config-gated where possible, so previous behavior can still be reproduced by flipping a flag back.
5. Preserve the existing group-safe split logic (`data/data_split.py`) and the existing `AsymmetricLoss` implementation's math — none of these five experiments should touch how `train.csv`/`val.csv`/`test.csv` were generated, nor rewrite the ASL loss formula itself.
6. If a step requires a new config field, add it to `src/config.py` with a sensible default that **reproduces current behavior when unchanged** (i.e., new features should be opt-in via a flag, defaulting to the old behavior), unless the step explicitly says to replace old behavior outright (Experiment 1 does this intentionally).
7. Do not touch `sweep_asl.py`'s Mol2Vec-based imports until Experiment 3 — it currently references a stale `src.featurization.Mol2VecFeaturizer` and `mol2vec_dim` that no longer exist in `src/config.py`. Leave it broken until Experiment 3 explicitly asks you to port it.

---

## Experiment 1 — Classifier bias initialization (do this first)

### Why
`src/model.py`'s final classifier layer (`nn.Linear(config.clf_hidden_dim_2, 1)`)
currently uses PyTorch's default bias init (~0), which implicitly assumes a
~50/50 class prior. With true prior ≈ 9.7% positive, the model wastes its
first few epochs correcting this before it can learn real signal — plausibly
why `best_epoch=3` shows up repeatedly in `metrics/molformer/run_metrics.json`
and `cv_metrics.json`. This is a 2-line, zero-risk fix and must be validated
**before** any of the other four experiments, since it is a confound that
would otherwise make every later experiment's results noisier than they truly are.

### What to change
In `src/model.py`, inside `APIExcipientModel.__init__`, after the `self.classifier`
`nn.Sequential` is constructed:

- Add code that initializes the **bias of the final `nn.Linear` layer** (the
  one producing the single logit output, i.e. `nn.Linear(config.clf_hidden_dim_2, 1)`)
  to `math.log(prior / (1 - prior))`, where `prior` is the positive class rate
  in the training data.
- Add a new `Config` field in `src/config.py`, e.g. `positive_prior: float = 0.094`
  (use the actual train.csv positive rate — compute it as `1839/(1839+191)`... no,
  actually recompute exactly from `data/train.csv`'s `Outcome1` column: positives / total rows).
  Do not hardcode a rounded guess — read the true ratio and put that exact value in
  the config default with a comment noting it was computed from `data/train.csv`.
- Use `torch.nn.init.constant_` (or direct `.data.fill_()`) on the last Linear
  layer's `.bias` right after `nn.Sequential` construction, inside `__init__`.
- Do NOT change the weight initialization, only the bias of the final output layer.
- Leave every other layer's initialization untouched.

### What NOT to change
- Do not touch `AsymmetricLoss` in `src/loss.py`.
- Do not touch the DataLoader / sampling behavior.
- Do not change `config.lr`, `config.max_epochs`, or any other hyperparameter.

### Done when
- `src/model.py` initializes the final classifier bias to the log-odds of the
  true training-set positive rate.
- `src/config.py` has a documented `positive_prior` field with a comment showing
  how it was computed.
- No training run performed by the agent.

---

## Experiment 2 — Balanced batch sampling

### Why
`src/dataset.py`'s `get_dataloaders()` builds the train `DataLoader` with plain
`shuffle=True`. With batch_size=64 and a 9.4% positive rate, the *expected*
number of positives per batch is ~6, but with random shuffling alone,
batch-to-batch positive counts vary a lot — some batches may have 0–2
positives, others 10+. This is a likely contributor to the high fold-to-fold
variance seen in `metrics/molformer/cv_metrics.json` (Recall ranging from
0.45 to 0.76 across 5 folds). Using a weighted sampler makes every batch's
expected positive ratio consistent, without introducing any synthetic rows.

### What to change
In `src/dataset.py`:

- In `get_dataloaders()`, replace the **train** `DataLoader`'s `shuffle=True`
  with a `torch.utils.data.WeightedRandomSampler`:
  - Compute per-sample weights as `1.0 / class_count[label]` for each row in
    `train_dataset.df["Outcome1"]`.
  - Pass `sampler=sampler` to the train `DataLoader` instead of `shuffle=True`
    (a `DataLoader` cannot have both `shuffle` and `sampler` set).
  - `num_samples` for the sampler should equal `len(train_dataset)` (i.e., same
    epoch length as before — this only changes the composition of each batch,
    not how many gradient steps happen per epoch), and use `replacement=True`.
- Do the same for `get_dataloader_from_dataframe()` **only when `is_train=True`**,
  since this function is reused by `cross_validate.py` for per-fold training data —
  the CV harness should benefit from this too. Val/test dataloaders in both
  functions must remain untouched (`shuffle=False`, no sampler) — we only ever
  want balanced sampling on data the model trains on, never on data used for
  evaluation, since evaluation must reflect the true class distribution.
- Add a new `Config` field, e.g. `use_balanced_sampler: bool = True`, and gate
  the new sampler logic behind it, so it can be toggled off to reproduce old
  behavior if needed. When `False`, fall back exactly to the previous
  `shuffle=True` behavior.

### What NOT to change
- Do not change `batch_size`.
- Do not modify `CompatibilityDataset.__getitem__` or the collate function.
- Do not touch val/test DataLoader construction in either function.

### Done when
- Train DataLoaders (in both `get_dataloaders` and `get_dataloader_from_dataframe`
  when `is_train=True`) use a `WeightedRandomSampler` gated by
  `config.use_balanced_sampler`, defaulting to `True`.
- Val/test DataLoaders are unchanged.
- No training run performed by the agent.

---

## Experiment 3 — ASL hyperparameter re-sweep on the current pipeline

### Why
The existing `sweep_asl.py` was written against a previous encoder
(`src.featurization.Mol2VecFeaturizer`, `config.mol2vec_dim`) that no longer
exists — the project has since migrated to frozen MoLFormer
(`src/molformer_featurization.py`, `config.molformer_dim`). The ASL grid
documented in `src/loss.py`'s docstring (`gamma_neg ∈ {2,3,4}`,
`gamma_pos ∈ {0,1}`, `clip ∈ {0, 0.05, 0.1}`) has therefore never actually been
swept against the current architecture — the current defaults
(`gamma_neg=4.0, gamma_pos=1.0, clip=0.05`) may or may not still be optimal
now that Experiments 1 and 2 have changed the training dynamics. This must run
**after** 1 and 2, not before, so the sweep isn't tuning ASL to compensate for
problems that 1–2 already fixed.

### What to change
Rewrite `sweep_asl.py` to match the current pipeline:

- Replace `from src.featurization import Mol2VecFeaturizer` and its
  instantiation with `from src.molformer_featurization import MolFormerFeaturizer`,
  constructed the same way `main.py` does it
  (`MolFormerFeaturizer(model_path=config.molformer_model_path)`).
- Remove any reference to `mol2vec_dim` / `mol2vec_model_path`.
- Keep the existing grid (`gamma_neg ∈ {2,3,4}`, `gamma_pos ∈ {0,1}`,
  `clip ∈ {0, 0.05, 0.1}`) and the existing structure (loop over
  `itertools.product`, train each combo via `train_model`, record val PR-AUC).
- Each combo should reuse `get_dataloaders` from `src/dataset.py` (so it
  automatically inherits Experiment 2's balanced sampler and Experiment 1's
  bias init, since both live in shared code paths).
- Save all combo results (gamma_neg, gamma_pos, clip, val PR-AUC, val F1, val MCC)
  to a single JSON or CSV under `metrics/molformer/asl_sweep_results.<ext>`
  instead of only logging to stdout, so results persist across sessions.
- Do not change the grid values themselves, and do not change
  `AsymmetricLoss`'s math in `src/loss.py`.

### What NOT to change
- Do not modify `src/loss.py`.
- Do not change the model architecture.

### Done when
- `sweep_asl.py` runs against the current MoLFormer-based pipeline without
  import errors, uses the current `get_dataloaders`, and writes structured
  results to `metrics/molformer/asl_sweep_results.*`.
- No training run performed by the agent (the user will run `python sweep_asl.py`
  themselves).

---

## Experiment 4 — 5-fold CV as the standard evaluation harness (wire in, don't just run)

### Why
`cross_validate.py` already exists and is fully wired to `train_model`,
`AsymmetricLoss`, and `APIExcipientModel` — it just isn't the *default* way
results get judged. Given only ~76–77 positives in the single held-out val
split, single-split metrics are noisy (see the fold-5 collapse in
`cv_metrics.json`: PR-AUC 0.52 vs. fold 3's 0.70 — same code, different
group-split luck). From this point forward, every subsequent experiment
(including 5, below) should be judged by **mean ± std across the 5 CV folds**,
not a single train/val/test run. This experiment doesn't add a new modeling
idea — it upgrades the ruler used to measure experiments 1–3 and 5.

### What to change
- Confirm `cross_validate.py` picks up Experiment 1's bias init automatically
  (it constructs `APIExcipientModel(fold_config)`, so it should — verify by
  reading, not running).
- Confirm `cross_validate.py` picks up Experiment 2's balanced sampler
  automatically via `get_dataloader_from_dataframe(..., is_train=True, ...)`
  — verify by reading, not running.
- Add a small summary utility (a new function, e.g. `print_cv_summary()` in
  `src/evaluate.py`, or a short script `summarize_cv.py` at repo root) that
  reads `metrics/molformer/cv_metrics.json` and prints a clean table of
  mean ± std for PR-AUC, F1, MCC, Recall, Precision across folds, so the user
  can quickly compare CV runs across experiments without manually reading raw JSON.
- Do not change `n_splits` (must stay 5, as enforced by the existing
  `ValueError` check in `cross_validate_config`).
- Do not change how folds are grouped (`split_group` / `API_CID`) or how
  `StratifiedGroupKFold` is invoked.

### What NOT to change
- Do not modify the grouping/splitting logic in `cross_validate.py`.
- Do not change `save_cross_validation_metrics` in `src/evaluate.py` beyond
  what's needed to add the summary utility described above (which should be a
  new function, not a rewrite of the existing save logic).

### Done when
- A summary function/script exists that prints mean ± std across folds from
  `cv_metrics.json`.
- No training run performed by the agent (user will run
  `python cross_validate.py` themselves and then the new summary tool).

---

## Experiment 5 — SMILES randomization augmentation for the minority class

### Why
Because `src/molformer_featurization.py` caches MoLFormer embeddings **by
exact SMILES string** (not by CID), generating alternate, chemically valid,
non-canonical SMILES for the same molecule
(`Chem.MolToSmiles(mol, doRandom=True)` in RDKit) produces genuinely different
embeddings for a molecule the frozen encoder has never seen written that way —
all while remaining 100% valid chemistry. This is the correct domain-specific
analogue of image augmentation for SMILES-based models, and is explicitly
**not** the same thing as SMOTE (which interpolates between two different
molecules' embeddings and can land in a region of embedding space no real
molecule occupies — this is why `src/loss.py`'s docstring says "do not combine
with SMOTE", and that constraint must still be respected). This experiment
should run last, after 1–4 have given a stable, low-variance baseline to
measure it against, because it is the most invasive change (touches dataset
construction) and is easiest to evaluate cleanly once prior noise sources are
under control.

### What to change
In `src/dataset.py`:

- Add a new function, e.g. `generate_smiles_variants(smiles: str, n_variants: int) -> list[str]`,
  using `rdkit.Chem.MolFromSmiles` + `rdkit.Chem.MolToSmiles(mol, doRandom=True)`
  to produce up to `n_variants` distinct random SMILES strings for a given
  input SMILES (dedupe generated strings; if RDKit fails to parse the input,
  return `[smiles]` unchanged rather than raising).
- In `CompatibilityDataset.__getitem__`, when `self.is_train` is `True` **and**
  the row's `Outcome1 == 1`, replace `api_smi` and/or `exc_smi` with a randomly
  chosen variant from `generate_smiles_variants(...)` at read time (i.e., a
  different valid SMILES string may be returned on different epochs for the
  same row). Negative-class rows and all val/test rows must keep their exact
  original SMILES — this experiment must apply **only to minority-class
  training rows**.
- Add new `Config` fields, e.g. `smiles_augment_positive_class: bool = True` and
  `smiles_augment_n_variants: int = 5`, gating this behavior so it can be
  toggled off to reproduce current behavior exactly.
- Precompute/cache the variant lists once per unique SMILES string (e.g. a
  dict on the `CompatibilityDataset` instance built in `__init__` for all
  positive-class API/excipient SMILES in the given split), rather than calling
  RDKit fresh on every `__getitem__` call, for performance.
- This must NOT change `API_CID` / `Excipient_CID` used for descriptor lookup
  in `create_collate_fn` — `DescriptorLookup.get_api`/`get_exc` key off CID,
  not SMILES, so swapping the SMILES string for an augmented variant of the
  same molecule does not break descriptor lookup. Verify this stays true.

### What NOT to change
- Do not apply this to negative-class rows.
- Do not apply this to val/test datasets under any config setting.
- Do not touch `molformer_featurization.py`'s caching logic itself — it
  already handles arbitrary new SMILES strings correctly by design.

### Done when
- `CompatibilityDataset` can return augmented (but chemically valid) SMILES
  variants for positive-class training rows only, gated by
  `config.smiles_augment_positive_class`.
- Val/test behavior is provably unchanged.
- No training run performed by the agent.

---

## Session checklist (repeat for each experiment)

1. Read this document's section for the current experiment number only.
2. Implement the changes described.
3. Summarize the diff to the user in plain language.
4. Explicitly state: "Implementation for Experiment N is complete. I have not
   run any training. Please train and evaluate now, then come back with
   results or tell me to proceed to Experiment N+1."
5. Stop. Do not proceed to the next experiment number until the user
   explicitly says so in a new message.

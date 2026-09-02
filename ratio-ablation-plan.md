# Class-Ratio Ablation Series — mlp-molformer-asl

## Purpose

Following the 5-experiment imbalance-handling plan (bias init, balanced
sampler, ASL re-sweep, CV harness, SMILES augmentation — the last of which
was reverted after it hurt CV performance), this document specifies a
**data-ratio ablation**: retrain the exact same architecture and Experiment
1–4 settings on progressively less-imbalanced subsets of the data, to find
out whether the current ~9:1 class ratio is actually capping performance, or
whether the ceiling is more about total data volume / architecture capacity.

The curve to build, in order: **4:1 → 3:1 → 1:1**, always compared against
the current full-data (~9.3:1) CV baseline.

## A script already exists for this: `make_ratio_dataset.py`

This has been written and tested against the real `data/start_dataset.csv`
already (verified: correct ratios produced, zero API-group leakage across
train/val/test, descriptor norm stats regenerated correctly). It must be
placed at the repo root, **unmodified**, and used as-is:

- It reads only `data/start_dataset.csv` (plus the two descriptor CSVs) —
  read-only, never written to.
- It selects **all** positive rows, plus a priority- and diversity-aware
  sample of negatives at the requested ratio (prioritizing negatives whose
  API and/or Excipient also appear in a positive pair — these are the most
  informative contrastive examples — with a soft per-API cap so no single
  API dominates the selection).
- It re-runs the **same** Butina-clustering + `StratifiedGroupKFold`
  leakage-safe split logic used by `data/data_split.py`, so no API ever
  crosses train/val/test in these new subsets either.
- It writes everything to a **new** directory —
  `data/ratio_experiments/ratio_<N>to1/` — containing `subset_dataset.csv`,
  `train.csv`, `val.csv`, `test.csv`, `descriptor_norm_stats.json`, and
  `manifest.json` (composition stats for transparency).
- It **never** touches `data/train.csv`, `data/val.csv`, `data/test.csv`,
  `data/start_dataset.csv`, or `models/descriptor_norm_stats.json`.

## Rules for the coding agent (same as the imbalance-experiment plan)

1. Implement **one numbered step per session**, in order. Do not combine steps.
2. **After implementing a step's code/config changes, STOP. Do not run
   training, `main.py`, or `cross_validate.py` yourself.** Generating the
   ratio datasets via `make_ratio_dataset.py` (step 1) is a data-prep step,
   not training, and is fine to run. Training the model on that data is not.
3. After each step, summarize exactly what changed (files + lines) and tell
   the user: *"This step is complete. Please generate/train/evaluate this
   yourself now. Come back when ready for the next step."*
4. Never modify: `data/data_split.py`, `data/train.csv`, `data/val.csv`,
   `data/test.csv`, `data/start_dataset.csv`, `data/api_descriptors.csv`,
   `data/excipient_descriptors.csv`, `models/descriptor_norm_stats.json`,
   `src/model.py`, `src/loss.py`. This ablation is data-only by design — if
   the architecture also changes, you can no longer attribute a result to
   the ratio change.
5. All new config fields must default to reproducing current behavior
   exactly when unused, same as the imbalance-experiment plan.

---

## Step 1 — Generate the three ratio datasets

### What to do
Place `make_ratio_dataset.py` at the repo root exactly as provided, then run
(this is data generation, not training, so it's fine to execute):

```bash
python make_ratio_dataset.py --ratio 4
python make_ratio_dataset.py --ratio 3
python make_ratio_dataset.py --ratio 1
```

This produces:
- `data/ratio_experiments/ratio_4to1/`
- `data/ratio_experiments/ratio_3to1/`
- `data/ratio_experiments/ratio_1to1/`

each with `train.csv`, `val.csv`, `test.csv`, `descriptor_norm_stats.json`,
`manifest.json`.

### Order rationale (why 4:1 → 3:1 → 1:1, not the reverse)
4:1 is the mildest change from the current ~9.3:1 ratio — it preserves the
most negative-class diversity while still meaningfully reducing skew. 3:1 is
a bigger step. 1:1 is the extreme case (344 vs 344) and also the biggest cut
to total data volume (3,544 → 688 rows), so it's evaluated last, after the
milder points on the curve give you a trend to interpret it against. If 1:1
were run first in isolation, a bad result would be ambiguous — you wouldn't
know if it's "imbalance wasn't the problem" or "you just don't have enough
total data anymore."

### Done when
All three directories exist under `data/ratio_experiments/` with the files
listed above. No training performed.

---

## Step 2 — Wire `main.py` and `cross_validate.py` to point at these directories

### Why
`main.py` already supports `--data-dir`, `--checkpoint-dir`, and
`--metrics-dir` CLI overrides, but has **no way to override
`descriptor_norm_stats_path`** — and each ratio subset has its own
regenerated norm stats file (they must not share one, since the stats are
computed from each split's own `train.csv`). `cross_validate.py` currently
has **no CLI args at all** — it always uses `Config()` defaults pointing at
`data/`.

### What to change

In `src/config.py`: no change needed — `descriptor_norm_stats_path` already
exists as a field; it just isn't exposed on the CLI yet.

In `main.py`, add one more argument alongside the existing three:
```python
parser.add_argument("--descriptor-norm-stats-path", default=None,
                     help="Path to descriptor_norm_stats.json for this run.")
...
if args.descriptor_norm_stats_path is not None:
    config.descriptor_norm_stats_path = args.descriptor_norm_stats_path
```

In `cross_validate.py`, add the same four-argument CLI block `main.py` already
has (`--data-dir`, `--checkpoint-dir`, `--metrics-dir`,
`--descriptor-norm-stats-path`), applied to the `config = config or Config()`
line inside `cross_validate_config`, or by adding an `if __name__ ==
"__main__":` block that parses args and passes a modified `Config()` in —
whichever fits the existing function signature with the least disruption.
**Critically: when no args are passed, `python cross_validate.py` must
behave exactly as it does today (full dataset, default paths)** — this is
what preserves your original CV baseline as a valid comparison point.

### Done when
Both of the following run without touching original data (still not actually
run by the agent, just confirmed importable/argparse-valid):
```bash
python main.py --data-dir data/ratio_experiments/ratio_4to1 \
    --checkpoint-dir checkpoints/ratio_4to1 \
    --metrics-dir metrics/ratio_4to1 \
    --descriptor-norm-stats-path data/ratio_experiments/ratio_4to1/descriptor_norm_stats.json

python cross_validate.py --data-dir data/ratio_experiments/ratio_4to1 \
    --checkpoint-dir checkpoints/ratio_4to1_cv \
    --metrics-dir metrics/ratio_4to1_cv \
    --descriptor-norm-stats-path data/ratio_experiments/ratio_4to1/descriptor_norm_stats.json
```
No training performed by the agent.

---

## Step 3 — Architecture / config: what NOT to change, and why

**No architecture changes are required or recommended for this ablation.**
The entire point is to hold the model fixed and vary only the data, so any
difference in CV metrics can be attributed to the class ratio rather than a
confound. Specifically:

- Do not modify `src/model.py` or `src/loss.py`.
- Keep `config.use_balanced_sampler = True` (from Experiment 2) across all
  three ratio runs, for consistency with the full-data baseline — even
  though at 1:1 the balanced sampler will have almost nothing to correct
  (the data is already balanced), leaving it on everywhere keeps the
  comparison clean. Do not special-case it off for the 1:1 run.
- Keep the classifier bias init from Experiment 1 as-is — note that
  `positive_prior` (used for the log-odds bias init) is computed from
  `data/train.csv`'s positive rate today; when running against a ratio
  subset, this should reflect *that subset's* prior, not the original 9.4%,
  or the bias init will start the model assuming the wrong base rate for the
  data it's actually training on. Add a config field override path (e.g.
  read the prior from the ratio experiment's own `manifest.json`
  `actual_ratio`, or recompute it from that run's `train.csv` at startup)
  rather than hardcoding the original 0.094 value for every ratio run.
- Keep ASL hyperparameters at whatever Experiment 3 settled on
  (`clip=0`, plus whichever `gamma_neg`/`gamma_pos` pair the CV comparison
  favored) — do not re-sweep ASL per ratio subset in this pass; that would
  turn a 1-variable ablation into a 2-variable one. If the ratio ablation
  later suggests a promising ratio, a follow-up ASL re-sweep *on that
  specific subset* would be a reasonable next experiment, but it is out of
  scope here.

### One thing to be aware of (not a required code change)
`batch_size=64` is unchanged, but total training rows drop sharply across
this series (2,030 → 1,030 → 828 → 410). At 1:1 that's roughly 6–7 batches
per epoch. This is worth watching when you train, but is a training-loop
observation for you to react to manually, not something the agent should
"fix" automatically — leave `batch_size` untouched unless you explicitly
decide otherwise after seeing the training curves.

### Done when
No files under `src/model.py` or `src/loss.py` have been touched. The
positive-prior bias-init source has been made subset-aware (reads from the
active ratio experiment's own train.csv / manifest rather than a hardcoded
constant), if not already handled by an existing mechanism.

---

## Session checklist (repeat per step)

1. Read only the section for the current step number.
2. Implement the changes described (Step 1's `make_ratio_dataset.py` runs are
   the one exception to "don't execute anything" — they are data generation,
   not training).
3. Summarize the diff in plain language.
4. State explicitly: "Step N is complete. I have not trained anything.
   Please train/evaluate the relevant ratio subset(s) now via
   `cross_validate.py`, then come back with results or tell me to proceed to
   Step N+1."
5. Stop and wait for the user's next message before proceeding.

## How to judge the results once you have them

Compare **CV mean ± std PR-AUC/F1/MCC** (not single-split, not val alone —
see prior discussion) across four points:
`9.3:1 (current baseline) → 4:1 → 3:1 → 1:1`.

- If PR-AUC/F1/MCC climb steadily as the ratio flattens, and 1:1 is
  meaningfully better than baseline: the class ratio itself was a real
  bottleneck, and the milder ratios along the way tell you how much of the
  full 1:1 gain you can get without discarding as much negative-class data.
- If the curve is flat or 1:1 is no better (or worse) than baseline: the
  ceiling is more likely total signal / data volume / architecture capacity
  than the ratio per se — since you gave the model a "fair fight" and it
  didn't do much more with it.
- Watch the **std**, not just the mean — a ratio that raises mean PR-AUC but
  also blows up fold-to-fold std (as SMILES augmentation did) is not a clear
  win; it may just be trading one kind of instability for another on a much
  smaller dataset.

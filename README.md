# DATA 255 Homework 2 — Continuous-time ICU mortality prediction

Starter project for in-hospital mortality prediction on the PhysioNet
Challenge 2012 dataset. Four models are planned: a GRU baseline, an MLP
Neural ODE, a KAN Neural ODE, and a liquid time-constant model.

This repository does not contain trained weights or model results.
`METRICS.md` and `RUN_LOG.txt` stay empty of model results until you run the
later experiment code.

## Setup

From this directory (`ECHW2`):

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

Hyperparameters and paths are in `src/config.py`. The configured seed is 8893.
Do not duplicate them in the scripts.

## Part A: download and preprocess Set A

From `ECHW2`, download the public PhysioNet Challenge 2012 Set A files:

```text
python scripts/download_data.py
```

This downloads `set-a.tar.gz` and `Outcomes-a.txt` from the assignment URLs,
then safely extracts the patient files into `data/raw/set-a/`. Existing files
are reused. Pass `--force` only when you intentionally want to download and
extract them again.

Run deterministic Part A preprocessing:

```text
python -m src.data_preprocessing
```

The pipeline keeps exactly these variables, in this order:

```text
HR, NIDiasABP, NIMAP, NISysABP, Temp, RespRate, GCS,
Glucose, Lactate, Urine, Creatinine, Platelets
```

For each patient it groups duplicate timestamp-variable measurements by their
mean and builds `times`, `values`, `mask`, and `delta_t` arrays over the
0–48 hour window. It uses seed 8893 for a patient-level 70/15/15 split,
stratified by mortality when possible. Z-score statistics come only from
observed training values; the same statistics are applied to all splits and
missing cells remain zero.

The command prints a preprocessing summary and writes:

```text
data/processed/part_a_cache.pkl
data/processed/splits.json
data/processed/normalization_stats.json
data/processed/preprocessing_summary.json
```

Raw downloads and processed artifacts are gitignored. Do not edit
`splits.json` between experiments: it records the exact RecordIDs so all
later models use the same patients.

## Part B model smoke test

After installing `requirements.txt`, the future in-memory Part B check is:

```text
python scripts/smoke_test_part_b.py
```

It uses synthetic padded tensors only. It does not read the Part A cache,
train a model, compute metrics, or write output files. Part B provides the
48-bin GRU baseline and the MLP-parametrized Neural ODE; their training and
evaluation loops are intentionally not implemented yet.

## Part C KAN Neural ODE smoke test

Part C uses the package that provides `from kan import KAN`. Install the
official PyKAN package if it is not already available:

```text
pip install pykan
```

The future in-memory Part C check is:

```text
python scripts/smoke_test_part_c.py
```

The MLP and KAN Neural ODEs inherit the same masked-mean encoder, mortality
head, query validation, solver path, forward API, and final-risk method.
Only their latent vector fields differ.

KAN edge plots (`beta=100`), weak-edge pruning, and optional symbolic fitting
with `x`, `x^2`, `exp`, `sin`, `tanh`, and `abs` are explicit post-training
helpers on `KANNeuralODE`. They belong to Experiment 3 after a real model has
been trained. The smoke test does not call them and creates no plots or
symbolic results.

## Part D liquid-model smoke test

Part D uses the official `ncps` PyTorch API. Install it if needed:

```text
pip install ncps
```

Run the future in-memory check with:

```text
python scripts/smoke_test_part_d.py
```

LTC is the full ODE-based liquid time-constant model. CfC is its faster
closed-form continuous-time approximation. Both consume normalized values,
observation masks, and mean per-timestep `delta_t` timespans and return logits
without applying a sigmoid.

The default `per_sample_compat` mode loops over patients, passes only valid
timesteps to `ncps`, and repads trajectory logits. This avoids a known
compatibility problem in current `ncps` versions when `(B, T)` timespans are
used in one call with batch size greater than one, and it ensures padding
cannot change predictions. Native batched timing should be benchmarked later
on the RTX 4090 only after the installed `ncps` version has been verified; it
is explicit rather than an automatic fallback.

## Part E unified harness smoke test

Run the future in-memory harness check with:

```text
python scripts/smoke_test_part_e.py
```

The shared harness routes GRU, MLP Neural ODE, KAN Neural ODE, LTC, and CfC
through registered final-logit adapters. It provides one-epoch training,
AUROC/AUPRC/Brier evaluation, validation-AUPRC early stopping, parameter
counts, elapsed time, and best-effort peak-memory tracking. CPU memory is
reported as Python-tracked `tracemalloc` memory, not total process or system
RAM.

No harness function writes results by default. Checkpoints are written only
when a future caller explicitly supplies a checkpoint path. Actual model
training and output creation occur only when the later Experiment scripts are
run on the RTX 4090.

## Part F Experiments 1 and 2

Future RTX 4090 commands:

```text
python scripts/run_experiment_1.py --dry-run
python scripts/run_experiment_1.py --run
python scripts/run_experiment_1.py --run --include-cfc
python scripts/run_experiment_2.py --dry-run
python scripts/run_experiment_2.py --run
```

Experiment 1 trains the four-model headline comparison (GRU, MLP-ODE,
KAN-ODE, and LTC) with validation-AUPRC early stopping, then evaluates the
selected checkpoint once on the held-out test split. `--include-cfc` keeps
CfC in a separate variant result rather than adding it to the four-row
headline table.

Experiment 2 is zero-shot sampling-irregularity robustness. It loads the
Experiment 1 checkpoints, deterministically retains 100%, 75%, 50%, or 25%
of each patient's timestamps, recomputes `delta_t`, and evaluates without
retraining. The GRU performs its normal hourly binning only after timestamp
removal; continuous-time models retain the surviving true timestamps.

Both scripts require either `--dry-run` or the explicit `--run` flag.
Dry runs validate inputs without creating output directories or files.

The temporary-only orchestration smoke test is:

```text
python scripts/smoke_test_part_f12.py
```

## Part F Experiments 3 and 4

Future RTX 4090 commands:

```text
python scripts/run_experiment_3.py --dry-run
python scripts/run_experiment_3.py --run
python scripts/run_experiment_4.py --dry-run
python scripts/run_experiment_4.py --run
```

Experiment 3 loads the trained Experiment 1 KAN-ODE checkpoint. It creates a
held-out vector-field probe from test-set latent encoder representations and
`time_hours=48`, then uses PyKAN activation collection and `feature_score`
attribution. Inputs are labeled only as `latent_state_i` or `time_hours`;
they are not labeled as raw clinical variables. Edge plotting uses
`beta=100`. Pruning and symbolic fitting operate only on an isolated copy.
If the installed PyKAN lacks a required API, the script stops without
substituting a heuristic.

Experiment 4 measures actual trainable parameter counts for MLP-ODE and KAN
candidates at hidden dimensions 8, 20, and 40. It selects the closest KAN
candidate deterministically and records any nonzero parameter difference
rather than claiming an exact match.

The temporary-only orchestration smoke test is:

```text
python scripts/smoke_test_part_f34.py
```

## Part G and final validation

Part G and executive-summary starter documents are:

```text
docs/part_g_reflection_template.md
docs/executive_summary_template.md
```

After real experiments finish, write completed answers separately as
`docs/part_g_reflection.md` and `docs/executive_summary.md`. Templates are not
accepted as completed results.

Run the read-only validator with:

```text
python scripts/validate_submission.py --dry-run
python scripts/validate_submission.py --final
```

`--dry-run` checks source and template structure without requiring GPU
outputs. `--final` checks completed experiment artifacts and writing, reports
the submission as not ready when required items are absent, and never creates
or changes files.

## Deliberate deviations from the starter code

- **Neural ODE encoder sees time.** The MLP-ODE and KAN-ODE share one encoder
  whose per-timestep input is `[values, mask, delta_t/48, t/48]` (37 dims)
  instead of `[values, mask]`, so observation timing is not discarded by the
  masked mean. Encoder, solver, head, and loss stay identical between the two.
- **Time input to the vector field is `t/48`** in both `ODEFuncMLP` and
  `ODEFuncKAN`, keeping it inside the KAN spline grid.
- **LTC/CfC timespans are real inter-step gaps** (`diff` of observation times,
  clamp minimum as the first-step epsilon), still clamped to [1e-4, 48].
- **Early stopping monitors validation AUPRC** and gradients are clipped to
  `GRAD_CLIP_NORM` in the shared training step.
- **Experiment 1 headline has four rows** (GRU, MLP-ODE, KAN-ODE, LTC) and
  reports both train-only and train+validation seconds per epoch. CfC is never
  a headline row; `python scripts/time_ltc_vs_cfc.py --run` times LTC vs CfC
  for 3 epochs into `outputs/experiment_1/ltc_vs_cfc_timing.csv`.
- `scripts/run_experiment_1.py --run` also writes `outputs/experiment_1/train.log`.

## Protocol note: patience

GRU and MLP-ODE in Experiment 1 used patience 10; KAN, LTC and Experiments 2 and 4 use patience 4.
This is a protocol note, not a bug (see `docs/DECISIONS.md`).

## Time-saving choices (and which are deviations)

Deviations from a full-protocol run (each is recorded in the results tables):
- **Wall-clock cap** of 20 minutes per run (`--max-run-minutes`, `0` disables) in
  Experiments 1, 2 (retrain) and 4. A capped run stops after its current epoch,
  restores the best checkpoint, is tested as usual, and is flagged
  `time_capped` in every table and footnoted in the plots.
- **Smaller, capacity-matched KAN.** The Experiment 1 KAN-ODE uses the direct-search
  matched width and grid for h=20 (same trainable-parameter budget as the
  MLP-ODE, within about 1%), not a larger KAN. Experiment 1 KAN is
  capacity-matched to MLP-ODE.
- **Partial retraining in Experiment 2.** Default: GRU and MLP-ODE only, at 50% and 25%
  retained, max 25 epochs (`--retrain-models`, `--levels` for more).
- **Reduced epoch budget** (max 25) for Experiment 2 retraining and Experiment 4.
- **Reused h=20 runs.** Experiment 4 imports the h=20 MLP-ODE and KAN-ODE results
  from Experiment 1 when the configurations match (the MLP-ODE used patience 10;
  `--no-reuse` retrains them).

Not deviations (they do not change results):
- Eval batch size 256 for the batch-invariant models (GRU, LTC); the ODE models
  keep batch size 32 because their adaptive solver makes outputs batch-dependent.
- Probe subsample (at most 200 points) and reduced plot sizes in Experiment 3;
  KAN `symbolic_enabled=False` during training (the symbolic branch is exactly zero).

## Running the remaining experiments

`python -u scripts/run_all.py --dry-list` prints the plan; `python -u
scripts/run_all.py` runs it unattended (log: `outputs/run_all.log`, progress:
`outputs/STATUS.json`). Experiments can also be run individually with
`--dry-run` / `--run`; Experiment 4 takes `--family`, `--hidden-dim`,
`--aggregate`; Experiment 2 takes `--retrain` and `--retrain-models`; the KAN
spline range can be overridden with `--kan-grid-range LOW HIGH`.

## Layout

```text
src/config.py                 seed, paths, split, optimization, device
src/data_preprocessing.py     Set A parsing, validation, split, normalization
src/dataset.py                Part A cache dataset and padded collator
src/models/                   implemented Part B–D architectures
src/train.py                  shared adapters and one-epoch training
src/evaluate.py               AUROC, AUPRC, and Brier evaluation
scripts/                      command-line entry points
data/raw/                     PhysioNet files (gitignored)
data/processed/               tensor cache (gitignored)
outputs/checkpoints/          weights (gitignored)
```

## AI use

See `AI_USE.md`. Results in the report have to come from your own runs.

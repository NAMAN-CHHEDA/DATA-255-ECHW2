# Decisions

Choices made while bringing Experiments 2-4 up to the Experiment 1 standard and
then cutting runtime. Nothing here was run when it was written; results come
only from real runs.

## Protocol
- **Patience.** GRU and MLP-ODE in Experiment 1 used patience 10; KAN, LTC and
  Experiments 2 and 4 use patience 4 (min epochs 5, min delta 0, AUPRC,
  gradient clipping, seed 8893). This is a protocol note, not a bug.
- **Epoch budgets.** Experiment 1 KAN/LTC keep max 50 epochs; Experiment 2
  retraining and Experiment 4 use max 25 (`config.REDUCED_MAX_EPOCHS`).
- **Wall-clock cap.** `--max-run-minutes` (default 20, `0` disables) on
  Experiments 1, 2 (retrain) and 4. When a run exceeds it, it stops after the
  current epoch, restores the best checkpoint, tests as usual, and the result
  is marked `time_capped: true`. It is not an error. The flag appears in every
  results CSV/JSON; the Experiment 4 plot and the combined Experiment 2 plot
  carry a footnote and a dagger on capped points. Resuming a run that already
  exceeded the cap stops it immediately and tests.
- **One training routine.** `train_and_test_model` in `scripts/run_experiment_1.py`
  gained optional `model_factory`, `run_name`, `display_name`, `extra_config`,
  `first_epoch_timeout` and `max_run_minutes`; Experiments 2 (retrain) and 4
  call it, so loaders, adapters, loss, clipping, early stopping, NFE logging,
  checkpoint restore and resume are identical. `extra_config` is part of the
  resume compatibility check; the cap is not.
- **`KAN_GRID_RANGE`** (config, default `(-1, 1)`) with `--kan-grid-range LOW HIGH`
  on Experiments 1-4. It is part of the effective configuration, so a
  checkpoint trained with another range will not resume. Loading a trained KAN
  is unaffected (the grid is stored in the state dict).
- **`ODE_MAX_NUM_STEPS`** is deliberately not in the resume-compatibility
  configuration: it changes nothing unless the guard fires.

## Capacity-matched KAN in Experiment 1
- The Experiment 1 KAN-ODE now uses the same direct-search matched configuration
  as Experiment 4's h=20 KAN (`src/matching.py`: hidden width 4..64 x grid
  {3, 4, 5}, k=3, minimizing the trainable-parameter gap to the MLP-ODE). The
  chosen config, trainable parameters and gap are printed at the start of the
  KAN run and stored in the results row (`kan_hidden_width`, `kan_grid`,
  `parameter_gap`, `parameter_gap_pct`). Experiment 1 KAN is therefore
  capacity-matched to MLP-ODE, not the earlier larger default.
- Experiments 2 and 3 build the KAN through the same matched constructor.

## Eval batch size
- Validation, test and zero-shot passes use batch size 256 only for models
  whose outputs do not depend on batch composition (GRU, LTC/CfC). The ODE models
  (MLP-ODE, KAN-ODE) share one adaptive solver step across the batch, so their
  outputs change slightly with batch size; they keep 32 so evaluation stays
  identical to the saved Experiment 1 numbers and the Experiment 2 100% check.
  This is narrower than "256 everywhere" on purpose. LTC runs one patient at a
  time internally, so its batch size gives no speed-up; the saving is mainly
  small. `scripts/check_eval_batch_invariance.py --run` (not run) measures the
  logit difference between batch sizes 32 and 256.

## Experiment 2
- Zero-shot drops whole timestamps (all variables at that row), seeded per
  record, levels 75/50/25 retained, three drop seeds (8893, 8894, 8895).
  100% is deterministic and cross-checked against Experiment 1 (warning, not
  failure, if it differs by more than 1e-4). CfC is not part of Experiment 2.
- Retraining thins train, validation and test data with the same level and seed
  (drop seed 8893). **Default is partial:** GRU and MLP-ODE only, at 50% and 25%
  (`--retrain-models`, `--levels` add more). The retrained 100% point is the
  Experiment 1 result.
- KAN and LTC are skipped by default: each retrain is a full training run and
  the KAN would cost many hours on top of Experiment 4.

## Experiment 3
- Probe rows are `[h(t), t/48]` where `h(t)` is each held-out patient's ODE
  state at 12 query times drawn uniformly in [0, 48] h (seeded), subsampled
  (seeded) to at most 200 rows for speed.
- Plotting: one pykan plot call before pruning and one after, reduced figure
  size (`scale=0.35`), and pykan's per-edge scratch PNGs are written at low dpi
  and deleted after each call. All analysis runs on a CPU copy; each stage is
  wrapped and failures go to `errors.log`.
- `writeup.md` is a generated draft (150-300 words) filled from run facts. It
  must be checked against the figures before submission.

## Experiment 4
- Direct-search matching (above); counts are analytic and verified by
  instantiating the chosen model; frozen symbolic affine parameters are not counted.
- One JSON per run (`runs/<family>_h<dim>.json`), skipped if it exists.
- **Reuse:** if `outputs/experiment_1` has a finished row for `kan_neural_ode`
  (same width, grid, grid range and parameter count as the h=20 match) or
  `mlp_neural_ode` (same parameter count), the h=20 run is imported instead of
  retrained, with `imported_from_experiment_1: true` and a `patience_note`
  (the MLP-ODE used patience 10 and max 50 epochs). `--no-reuse` disables it.

## Driver (`scripts/run_all.py`)
- Experiment 1 KAN and LTC run as two separate invocations (`--models`) with
  patience 4, so a KAN failure cannot block LTC. Models that already have a
  finished headline row are never rerun or deleted. Experiment 1 KAN runs
  before Experiment 4 so the h=20 reuse can apply.
- KAN runaway guard: `--first-epoch-timeout 360` (exit code 3) or a
  `max_num_steps exceeded` error triggers one rerun with `--kan-grid-range -3 3`
  (10 h wall limit); the value is reused for later KAN steps and recorded in
  `outputs/STATUS.json`.
- `--dry-list` prints an estimated-minutes table: "upper" is the per-run cap
  plus overhead; "likely" assumes about 12 epochs (best epoch ~8 + patience 4)
  at assumed seconds per epoch (MLP-ODE measured ~17 s; others are guesses).

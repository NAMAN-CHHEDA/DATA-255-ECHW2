"""Configurable values for DATA 255 Homework 2.

Every path, split, and training hyperparameter used by the scripts lives
here. Nothing in this module is an experimental result. Defaults other than
the required seed have not been tuned.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 8893

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
RUN_LOG_PATH = PROJECT_ROOT / "RUN_LOG.txt"

# Public labeled PhysioNet Challenge 2012 Set A release. Set B and Set C are
# outside Part A and are not downloaded or read.
LABELED_SET_DIRNAME = "set-a"
OUTCOMES_FILENAME = "Outcomes-a.txt"
SET_A_ARCHIVE_FILENAME = "set-a.tar.gz"
SET_A_URL = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz"
OUTCOMES_A_URL = "https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt"

SET_A_DIR = RAW_DATA_DIR / LABELED_SET_DIRNAME
SET_A_ARCHIVE_PATH = RAW_DATA_DIR / SET_A_ARCHIVE_FILENAME
OUTCOMES_PATH = RAW_DATA_DIR / OUTCOMES_FILENAME

PROCESSED_FILENAME = "part_a_cache.pkl"
SPLITS_FILENAME = "splits.json"
NORMALIZATION_STATS_FILENAME = "normalization_stats.json"
PREPROCESSING_SUMMARY_FILENAME = "preprocessing_summary.json"

PART_A_CACHE_PATH = PROCESSED_DATA_DIR / PROCESSED_FILENAME
SPLITS_PATH = PROCESSED_DATA_DIR / SPLITS_FILENAME
NORMALIZATION_STATS_PATH = PROCESSED_DATA_DIR / NORMALIZATION_STATS_FILENAME
PREPROCESSING_SUMMARY_PATH = PROCESSED_DATA_DIR / PREPROCESSING_SUMMARY_FILENAME
EXPERIMENT_1_DIR = OUTPUTS_DIR / "experiment_1"
EXPERIMENT_2_DIR = OUTPUTS_DIR / "experiment_2"
EXPERIMENT_3_DIR = OUTPUTS_DIR / "experiment_3"
EXPERIMENT_4_DIR = OUTPUTS_DIR / "experiment_4"

# Stay-level fractions of the labeled set. Applied only after preprocessing
# exists. They must sum to 1.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_METRIC = "val_auprc"
EARLY_STOPPING_MODE = "max"
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 0

# Part B model defaults. These are implementation defaults, not tuned results.
N_VARS = 12
GRU_HIDDEN_SIZE = 32
ODE_HIDDEN_DIM = 20
ODE_WIDTH = 64
HIDDEN_DIM = GRU_HIDDEN_SIZE
DROPOUT = 0.1
ODE_SOLVER = "dopri5"
ODE_RTOL = 1e-3
ODE_ATOL = 1e-4
KAN_GRID = 5
KAN_SPLINE_ORDER = 3
KAN_HIDDEN_WIDTH_MULTIPLIER = 2
LIQUID_UNITS = 32
LIQUID_ODE_UNFOLDS = 6
LIQUID_TIMESPAN_MIN = 1e-4
LIQUID_TIMESPAN_MAX = 48.0
LIQUID_BATCH_MODE = "per_sample_compat"
PRIMARY_MODEL_KEYS = (
    "gru_baseline",
    "mlp_neural_ode",
    "kan_neural_ode",
    "ltc",
)
ROBUSTNESS_RETAINED_PCTS = (100, 75, 50, 25)
PARAMETER_EFFICIENCY_HIDDEN_DIMS = (8, 20, 40)
KAN_CANDIDATE_WIDTH_MULTIPLIERS = (1, 2, 3)
KAN_CANDIDATE_GRIDS = (3, 5, 7)

# "auto" uses CUDA when torch can see a GPU, otherwise CPU.
# Set DEVICE_PREFERENCE to "cpu" or "cuda" to force a device.
DEVICE_PREFERENCE = "auto"

MAX_HOURS = 48.0
LABEL_COLUMN = "In-hospital_death"


def validate_config() -> None:
    """Raise ValueError if a configured value cannot be used later."""
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1, got {total}."
        )
    if min(TRAIN_RATIO, VAL_RATIO, TEST_RATIO) <= 0:
        raise ValueError("Each split ratio must be positive.")
    if SEED < 0:
        raise ValueError("SEED must be non-negative.")
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1.")
    if LEARNING_RATE <= 0:
        raise ValueError("LEARNING_RATE must be positive.")
    if EPOCHS < 1:
        raise ValueError("EPOCHS must be at least 1.")
    if EARLY_STOPPING_PATIENCE < 1:
        raise ValueError("EARLY_STOPPING_PATIENCE must be at least 1.")
    if DEVICE_PREFERENCE not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "DEVICE_PREFERENCE must be 'auto', 'cpu', or 'cuda'."
        )
    if EARLY_STOPPING_MODE not in {"min", "max"}:
        raise ValueError("EARLY_STOPPING_MODE must be 'min' or 'max'.")
    if min(N_VARS, GRU_HIDDEN_SIZE, ODE_HIDDEN_DIM, ODE_WIDTH) < 1:
        raise ValueError("Part B model dimensions must be positive.")
    if ODE_RTOL <= 0 or ODE_ATOL <= 0:
        raise ValueError("ODE_RTOL and ODE_ATOL must be positive.")
    if min(KAN_GRID, KAN_SPLINE_ORDER, KAN_HIDDEN_WIDTH_MULTIPLIER) < 1:
        raise ValueError("KAN grid, spline order, and width multiplier must be positive.")
    if LIQUID_UNITS < 1 or LIQUID_ODE_UNFOLDS < 1:
        raise ValueError("Liquid units and ODE unfolds must be positive.")
    if not 0 < LIQUID_TIMESPAN_MIN <= LIQUID_TIMESPAN_MAX:
        raise ValueError(
            "Liquid timespan bounds must satisfy 0 < minimum <= maximum."
        )
    if LIQUID_BATCH_MODE not in {"per_sample_compat", "native_batched"}:
        raise ValueError(
            "LIQUID_BATCH_MODE must be 'per_sample_compat' or 'native_batched'."
        )
    if any(not 0 < retained <= 100 for retained in ROBUSTNESS_RETAINED_PCTS):
        raise ValueError("Robustness retained percentages must be in (0, 100].")
    if any(value < 1 for value in PARAMETER_EFFICIENCY_HIDDEN_DIMS):
        raise ValueError("Parameter-efficiency hidden dimensions must be positive.")
    if any(value < 1 for value in KAN_CANDIDATE_WIDTH_MULTIPLIERS):
        raise ValueError("KAN candidate width multipliers must be positive.")
    if any(value < 1 for value in KAN_CANDIDATE_GRIDS):
        raise ValueError("KAN candidate grid sizes must be positive.")


def resolve_device(preference: str | None = None) -> str:
    """Return the torch device string selected from ``preference``.

    ``None`` uses ``DEVICE_PREFERENCE``. This imports torch. It does not
    move any model parameters.
    """
    choice = DEVICE_PREFERENCE if preference is None else preference
    if choice not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device preference must be 'auto', 'cpu', or 'cuda'.")

    import torch

    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE_PREFERENCE is 'cuda', but torch.cuda.is_available() is false."
            )
        return "cuda"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def checkpoint_path(model_key: str) -> Path:
    """Path a future training run should use for this model and seed."""
    return CHECKPOINTS_DIR / f"{model_key}_seed{SEED}.pt"


def processed_path() -> Path:
    """Location of the Part A preprocessed pickle cache."""
    return PART_A_CACHE_PATH


def format_config(device: str) -> str:
    """Multiline summary of the active configuration. Contains no metrics."""
    lines = [
        "DATA 255 Homework 2 configuration",
        f"Project root: {PROJECT_ROOT}",
        f"Seed: {SEED}",
        f"Device preference: {DEVICE_PREFERENCE}",
        f"Resolved device: {device}",
        f"Raw data: {RAW_DATA_DIR}",
        f"Processed cache: {processed_path()}",
        f"Checkpoints: {CHECKPOINTS_DIR}",
        f"Figures: {FIGURES_DIR}",
        f"Tables: {TABLES_DIR}",
        (
            "Split ratios (train / validation / test): "
            f"{TRAIN_RATIO:.2f} / {VAL_RATIO:.2f} / {TEST_RATIO:.2f}"
        ),
        f"Batch size: {BATCH_SIZE}",
        f"Learning rate: {LEARNING_RATE}",
        f"Weight decay: {WEIGHT_DECAY}",
        f"Epochs: {EPOCHS}",
        f"Early stopping patience: {EARLY_STOPPING_PATIENCE}",
        f"Early stopping metric: {EARLY_STOPPING_METRIC} ({EARLY_STOPPING_MODE})",
        f"Gradient clip norm: {GRAD_CLIP_NORM}",
        f"Number of Part B variables: {N_VARS}",
        f"GRU hidden size: {GRU_HIDDEN_SIZE}",
        f"ODE hidden dimension: {ODE_HIDDEN_DIM}",
        f"ODE width: {ODE_WIDTH}",
        f"Dropout: {DROPOUT}",
        f"ODE solver name: {ODE_SOLVER}",
        f"ODE rtol / atol: {ODE_RTOL} / {ODE_ATOL}",
        f"KAN grid / spline order: {KAN_GRID} / {KAN_SPLINE_ORDER}",
        (
            "KAN hidden width formula: "
            f"{KAN_HIDDEN_WIDTH_MULTIPLIER} * hidden_dim + 1"
        ),
        f"Liquid units / ODE unfolds: {LIQUID_UNITS} / {LIQUID_ODE_UNFOLDS}",
        (
            "Liquid timespan min / max: "
            f"{LIQUID_TIMESPAN_MIN} / {LIQUID_TIMESPAN_MAX}"
        ),
        f"Liquid batch mode: {LIQUID_BATCH_MODE}",
        f"Max hours: {MAX_HOURS}",
    ]
    return "\n".join(lines)

"""Experiment 2: zero-shot robustness to timestamp removal."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment_1 import (  # noqa: E402
    PRIMARY_MODEL_NAMES,
    VARIANT_MODEL_NAME,
    build_model,
    collate_for_experiment,
    configuration_snapshot,
    planned_model_keys,
    validate_part_a_cache,
)

ROBUSTNESS_FIELDS = [
    "model",
    "retained_pct",
    "auroc",
    "auprc",
    "brier_score",
    "sample_count",
]


def build_parser() -> argparse.ArgumentParser:
    """Build a CLI that requires explicit permission for evaluation output."""
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 2: load Experiment 1 checkpoints and evaluate "
            "zero-shot robustness after deterministic timestamp removal."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate cache/checkpoints without creating files.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Explicitly execute zero-shot evaluation and create outputs.",
    )
    parser.add_argument(
        "--include-cfc",
        action="store_true",
        help="Evaluate CfC separately and include its line only when requested.",
    )
    return parser


def _record_seed(seed: int, record_id: str, retained_pct: int) -> int:
    """Derive a stable RNG seed independent of Python's randomized hash."""
    payload = f"{seed}:{record_id}:{retained_pct}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def drop_record_timestamps(
    record: dict[str, Any],
    retained_pct: int,
    seed: int,
) -> dict[str, Any]:
    """Deterministically retain whole timestamps and recompute ``delta_t``."""
    from src.utils import recompute_delta_t

    if not 0 < retained_pct <= 100:
        raise ValueError("retained_pct must be in (0, 100].")
    record_id = str(record["record_id"])
    times = np.asarray(record["times"])
    values = np.asarray(record["values"])
    mask = np.asarray(record["mask"])
    if times.ndim != 1 or len(times) == 0:
        raise ValueError(f"Record {record_id} has no timestamps.")
    if values.shape[:1] != times.shape or mask.shape != values.shape:
        raise ValueError(f"Record {record_id} has inconsistent tensor shapes.")

    timestamp_count = len(times)
    retain_count = max(
        1,
        min(
            timestamp_count,
            math.ceil(timestamp_count * retained_pct / 100.0),
        ),
    )
    if retain_count == timestamp_count:
        retained_indices = np.arange(timestamp_count)
    else:
        rng = np.random.default_rng(
            _record_seed(seed, record_id, retained_pct)
        )
        retained_indices = np.sort(
            rng.choice(timestamp_count, size=retain_count, replace=False)
        )

    retained_times = times[retained_indices].copy()
    retained_values = values[retained_indices].copy()
    retained_mask = mask[retained_indices].copy()
    if len(retained_times) > 1 and np.any(np.diff(retained_times) <= 0):
        raise ValueError(f"Record {record_id} retained times are not sorted.")

    times_tensor = torch.as_tensor(retained_times, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.as_tensor(retained_mask, dtype=torch.float32).unsqueeze(0)
    valid = torch.ones((1, len(retained_times)), dtype=torch.bool)
    retained_delta_t = recompute_delta_t(
        times_tensor,
        mask_tensor,
        valid,
    )[0].cpu().numpy()

    return {
        "record_id": record_id,
        "times": retained_times,
        "values": retained_values,
        "mask": retained_mask,
        "delta_t": retained_delta_t,
        "label": int(record["label"]),
    }


def checkpoint_paths(
    experiment_1_dir: Path,
    include_cfc: bool,
) -> dict[str, Path]:
    """Return the required Experiment 1 checkpoint paths."""
    return {
        model_key: experiment_1_dir / "checkpoints" / f"{model_key}_best.pt"
        for model_key in planned_model_keys(include_cfc)
    }


def validate_dry_run(
    cache_path: Path,
    experiment_1_dir: Path,
    output_dir: Path,
    include_cfc: bool = False,
    validate_state_dicts: bool = True,
) -> dict[str, Any]:
    """Validate the zero-shot plan without creating output directories."""
    from src.config import ROBUSTNESS_RETAINED_PCTS, validate_config

    validate_config()
    payload = validate_part_a_cache(cache_path)
    paths = checkpoint_paths(experiment_1_dir, include_cfc)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Experiment 1 checkpoints are missing: " + ", ".join(missing)
        )
    if validate_state_dicts:
        for model_key, path in paths.items():
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            model = build_model(model_key)
            model.load_state_dict(state_dict)

    return {
        "models": planned_model_keys(include_cfc),
        "retained_percentages": list(ROBUSTNESS_RETAINED_PCTS),
        "test_patients": len(payload["records"]["test"]),
        "checkpoints": {key: str(path) for key, path in paths.items()},
        "output_dir": str(output_dir),
    }


def build_robustness_loader(
    records: list[dict[str, Any]],
    retained_pct: int,
) -> Any:
    """Build an unshuffled loader over deterministically thinned records."""
    from torch.utils.data import DataLoader

    from src.config import BATCH_SIZE, NUM_WORKERS, SEED
    from src.dataset import ICUMortalityDataset
    from src.utils import seed_worker

    dropped_records = [
        drop_record_timestamps(record, retained_pct, SEED)
        for record in records
    ]
    dataset = ICUMortalityDataset(dropped_records, split="test")
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_for_experiment,
        worker_init_fn=seed_worker,
    )


def evaluate_checkpoint(
    model_key: str,
    checkpoint_path: Path,
    records_by_retention: dict[int, Any],
    device: str,
) -> list[dict[str, Any]]:
    """Load one best checkpoint and evaluate every drop level without training."""
    from src.evaluate import evaluate
    from src.train import get_forward_adapter

    model = build_model(model_key).to(device)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    forward_fn = get_forward_adapter(model_key)
    display_name = (
        VARIANT_MODEL_NAME
        if model_key == "cfc"
        else PRIMARY_MODEL_NAMES[model_key]
    )

    rows: list[dict[str, Any]] = []
    for retained_pct, loader in records_by_retention.items():
        metrics = evaluate(model, loader, forward_fn, device)
        rows.append(
            {
                "model": display_name,
                "retained_pct": retained_pct,
                "auroc": float(metrics["auroc"]),
                "auprc": float(metrics["auprc"]),
                "brier_score": float(metrics["brier_score"]),
                "sample_count": int(metrics["sample_count"]),
            }
        )
    return rows


def save_robustness_plot(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Save AUROC versus retained timestamps for the requested models."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    model_order = list(PRIMARY_MODEL_NAMES.values())
    if any(row["model"] == VARIANT_MODEL_NAME for row in rows):
        model_order.append(VARIANT_MODEL_NAME)
    for model_name in model_order:
        model_rows = sorted(
            (row for row in rows if row["model"] == model_name),
            key=lambda row: row["retained_pct"],
        )
        if not model_rows:
            continue
        axis.plot(
            [row["retained_pct"] for row in model_rows],
            [row["auroc"] for row in model_rows],
            marker="o",
            label=model_name,
        )
    axis.set_xlabel("% observations retained")
    axis.set_ylabel("AUROC")
    axis.set_xticks([25, 50, 75, 100])
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_experiment(include_cfc: bool = False) -> list[dict[str, Any]]:
    """Execute zero-shot robustness evaluation and write defined outputs."""
    from src.config import (
        EXPERIMENT_1_DIR,
        EXPERIMENT_2_DIR,
        PART_A_CACHE_PATH,
        ROBUSTNESS_RETAINED_PCTS,
        SEED,
        resolve_device,
    )
    from src.utils import package_versions, write_csv, write_json

    validate_dry_run(
        PART_A_CACHE_PATH,
        EXPERIMENT_1_DIR,
        EXPERIMENT_2_DIR,
        include_cfc=include_cfc,
    )
    payload = validate_part_a_cache(PART_A_CACHE_PATH)
    test_records = payload["records"]["test"]
    loaders = {
        retained_pct: build_robustness_loader(test_records, retained_pct)
        for retained_pct in ROBUSTNESS_RETAINED_PCTS
    }
    device = resolve_device()
    paths = checkpoint_paths(EXPERIMENT_1_DIR, include_cfc)
    rows: list[dict[str, Any]] = []
    for model_key in planned_model_keys(include_cfc):
        rows.extend(
            evaluate_checkpoint(
                model_key,
                paths[model_key],
                loaders,
                device,
            )
        )

    primary_rows = [row for row in rows if row["model"] != VARIANT_MODEL_NAME]
    write_csv(
        EXPERIMENT_2_DIR / "robustness_results.csv",
        primary_rows,
        ROBUSTNESS_FIELDS,
    )
    if include_cfc:
        cfc_rows = [row for row in rows if row["model"] == VARIANT_MODEL_NAME]
        write_csv(
            EXPERIMENT_2_DIR / "cfc_variant_robustness_results.csv",
            cfc_rows,
            ROBUSTNESS_FIELDS,
        )
    save_robustness_plot(rows, EXPERIMENT_2_DIR / "robustness_auroc.png")

    metadata = {
        "experiment": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "configuration": configuration_snapshot(),
        "package_versions": package_versions(
            [
                "torch",
                "numpy",
                "scikit-learn",
                "torchdiffeq",
                "pykan",
                "ncps",
                "matplotlib",
            ]
        ),
        "device": device,
        "models": planned_model_keys(include_cfc),
        "retained_percentages": list(ROBUSTNESS_RETAINED_PCTS),
        "zero_shot_no_retraining": True,
        "source_checkpoints": {key: str(path) for key, path in paths.items()},
    }
    write_json(EXPERIMENT_2_DIR / "run_metadata.json", metadata)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Validate a plan or explicitly execute Experiment 2."""
    args = build_parser().parse_args(argv)
    from src.config import (
        EXPERIMENT_1_DIR,
        EXPERIMENT_2_DIR,
        PART_A_CACHE_PATH,
    )

    if args.dry_run:
        plan = validate_dry_run(
            PART_A_CACHE_PATH,
            EXPERIMENT_1_DIR,
            EXPERIMENT_2_DIR,
            include_cfc=args.include_cfc,
        )
        print("Experiment 2 dry run")
        print(f"Part A cache: {PART_A_CACHE_PATH}")
        print(f"Validated checkpoints: {', '.join(plan['models'])}")
        print(
            "Retained timestamp levels: "
            + ", ".join(str(value) for value in plan["retained_percentages"])
        )
        print(f"Planned output directory: {EXPERIMENT_2_DIR}")
        print("No directories or files were created.")
        return 0

    run_experiment(include_cfc=args.include_cfc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

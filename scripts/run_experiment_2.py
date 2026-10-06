"""Experiment 2: robustness to timestamp removal.

``--run``     zero-shot: Experiment 1 best checkpoints evaluated on test data
              with whole timestamps randomly dropped (seeded per record, three
              drop seeds, no re-binning).
``--retrain`` retrain each model from scratch on thinned train/val/test data at
              each level with the Experiment 1 harness (patience 4, max 25 epochs, wall-clock
              cap per run; default GRU and MLP-ODE at 50% and 25%), one JSON
              per run, skipped if it already exists.
Both modes refresh the figures; the combined figure shows zero-shot (solid)
and retrained (dashed) curves in the same color per model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment_1 import (  # noqa: E402
    PRIMARY_MODEL_NAMES,
    TrainingSettings,
    apply_kan_grid_range,
    build_model,
    configuration_snapshot,
    load_model_state,
    resolve_max_run_minutes,
    tee_output_to_log,
    train_and_test_model,
    validate_part_a_cache,
)

COLORS = {
    "gru_baseline": "#009E73",
    "mlp_neural_ode": "#0072B2",
    "kan_neural_ode": "#D55E00",
    "ltc": "#CC79A7",
}
MARKERS = {
    "gru_baseline": "o",
    "mlp_neural_ode": "s",
    "kan_neural_ode": "^",
    "ltc": "D",
}
RAW_FIELDS = ["model", "retained_pct", "seed", "auroc", "auprc", "brier_score", "sample_count"]
SUMMARY_FIELDS = [
    "model",
    "retained_pct",
    "n_seeds",
    "auroc_mean",
    "auroc_std",
    "auprc_mean",
    "auprc_std",
    "brier_mean",
    "brier_std",
]
CONSISTENCY_TOLERANCE = 1e-4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate inputs; write nothing.")
    mode.add_argument("--run", action="store_true", help="Zero-shot evaluation.")
    mode.add_argument("--retrain", action="store_true", help="Retrain on thinned data.")
    models = list(PRIMARY_MODEL_NAMES)
    parser.add_argument("--models", nargs="+", choices=models, default=models, help="Zero-shot models.")
    parser.add_argument("--retrain-models", nargs="+", choices=models, default=["gru_baseline", "mlp_neural_ode"], help="Models to retrain (default GRU and MLP-ODE; add kan_neural_ode or ltc on demand).")
    parser.add_argument("--levels", nargs="+", type=int, default=[50, 25], help="Retained percentages to retrain at (default 50 25).")
    parser.add_argument("--max-run-minutes", type=float, default=None, help="Wall-clock cap per retrain run (default 20; 0 disables).")
    parser.add_argument("--seeds", type=int, default=3, help="Number of drop seeds for zero-shot (default 3).")
    parser.add_argument("--resume", action="store_true", help="Resume unfinished retrain runs from latest.pt.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: outputs/experiment_2.")
    parser.add_argument("--kan-grid-range", nargs=2, type=float, metavar=("LOW", "HIGH"), default=None)
    return parser


def _record_seed(seed: int, record_id: str, retained_pct: int) -> int:
    """Stable RNG seed independent of Python's randomized hash."""
    digest = hashlib.sha256(f"{seed}:{record_id}:{retained_pct}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def drop_record_timestamps(
    record: dict[str, Any],
    retained_pct: int,
    seed: int,
) -> dict[str, Any]:
    """Keep ``ceil(pct%)`` whole timestamps (all variables at those rows)."""
    if not 0 < retained_pct <= 100:
        raise ValueError("retained_pct must be in (0, 100].")
    record_id = str(record["record_id"])
    times = np.asarray(record["times"])
    values = np.asarray(record["values"])
    mask = np.asarray(record["mask"])
    if times.ndim != 1 or len(times) == 0:
        raise ValueError(f"Record {record_id} has no timestamps.")
    count = len(times)
    keep = max(1, min(count, math.ceil(count * retained_pct / 100.0)))
    if keep == count:
        indices = np.arange(count)
    else:
        rng = np.random.default_rng(_record_seed(seed, record_id, retained_pct))
        indices = np.sort(rng.choice(count, size=keep, replace=False))
    return {
        "record_id": record_id,
        "times": times[indices].copy(),
        "values": values[indices].copy(),
        "mask": mask[indices].copy(),
        "label": int(record["label"]),
    }


def thin_records(records: list[dict[str, Any]], retained_pct: int, seed: int) -> list[dict[str, Any]]:
    return [drop_record_timestamps(record, retained_pct, seed) for record in records]


def checkpoint_paths(experiment_1_dir: Path, models: list[str]) -> dict[str, Path]:
    return {key: experiment_1_dir / "checkpoints" / key / "best.pt" for key in models}


def make_eval_loader(dataset: Any, model_key: str, device: str) -> Any:
    from torch.utils.data import DataLoader

    from src.config import NUM_WORKERS, eval_batch_size
    from src.dataset import materialized_collator

    return DataLoader(
        dataset,
        batch_size=eval_batch_size(model_key),
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=materialized_collator(model_key),
        pin_memory=device == "cuda",
    )


def load_experiment_1_model(model_key: str, path: Path, device: str) -> Any:
    model = build_model(model_key).to(device)
    model.load_state_dict(load_model_state(path, map_location=device))
    if model_key == "kan_neural_ode":
        from src.models.kan_neural_ode import sync_kan_device

        sync_kan_device(model.ode_func.network)
    return model


def validate_dry_run(
    cache_path: Path,
    experiment_1_dir: Path,
    models: list[str],
) -> dict[str, Any]:
    from src.config import ROBUSTNESS_RETAINED_PCTS, validate_config

    validate_config()
    payload = validate_part_a_cache(cache_path)
    paths = checkpoint_paths(experiment_1_dir, models)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Experiment 1 checkpoints are missing: " + ", ".join(missing))
    for model_key, path in paths.items():
        build_model(model_key).load_state_dict(load_model_state(path, map_location="cpu"))
    return {
        "models": models,
        "retained_percentages": list(ROBUSTNESS_RETAINED_PCTS),
        "test_patients": len(payload["records"]["test"]),
        "checkpoints": {key: str(path) for key, path in paths.items()},
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return float(array.mean()), std


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    keys = sorted({(row["model"], row["retained_pct"]) for row in rows}, key=lambda k: (list(PRIMARY_MODEL_NAMES.values()).index(k[0]), -k[1]))
    for model, pct in keys:
        subset = [r for r in rows if r["model"] == model and r["retained_pct"] == pct]
        entry: dict[str, Any] = {"model": model, "retained_pct": pct, "n_seeds": len(subset)}
        for name, field in (("auroc", "auroc"), ("auprc", "auprc"), ("brier", "brier_score")):
            mean, std = _mean_std([float(r[field]) for r in subset])
            entry[f"{name}_mean"] = mean
            entry[f"{name}_std"] = std
        summary.append(entry)
    return summary


def check_against_experiment_1(
    rows: list[dict[str, Any]],
    experiment_1_dir: Path,
) -> list[dict[str, Any]]:
    """The 100% row must reproduce the Experiment 1 test metrics (1e-4)."""
    path = experiment_1_dir / "headline_results.json"
    if not path.is_file():
        print("WARNING: headline_results.json missing; cannot cross-check 100% row.", flush=True)
        return []
    headline = {r.get("model_key"): r for r in json.loads(path.read_text(encoding="utf-8"))}
    checks: list[dict[str, Any]] = []
    for model_key, name in PRIMARY_MODEL_NAMES.items():
        full = [r for r in rows if r["model"] == name and r["retained_pct"] == 100]
        if not full or model_key not in headline:
            continue
        for metric, field in (("auroc", "auroc"), ("auprc", "auprc"), ("brier_score", "brier_score")):
            reference = float(headline[model_key][field])
            observed = float(full[0][metric])
            difference = abs(reference - observed)
            ok = difference <= CONSISTENCY_TOLERANCE
            checks.append({"model": name, "metric": metric, "experiment_1": reference, "experiment_2": observed, "abs_difference": difference, "ok": ok})
            if not ok:
                print(
                    f"WARNING: {name} 100% {metric} {observed:.6f} differs from "
                    f"Experiment 1 {reference:.6f} by {difference:.2e} "
                    f"(> {CONSISTENCY_TOLERANCE:g}).",
                    flush=True,
                )
    return checks


def run_zero_shot(models: list[str], n_seeds: int, output_dir: Path) -> list[dict[str, Any]]:
    from src.config import (
        EXPERIMENT_1_DIR,
        PART_A_CACHE_PATH,
        ROBUSTNESS_RETAINED_PCTS,
        SEED,
        resolve_device,
    )
    from src.dataset import materialize_records
    from src.evaluate import evaluate
    from src.train import get_forward_adapter
    from src.utils import package_versions, set_seed, write_csv, write_json

    validate_dry_run(PART_A_CACHE_PATH, EXPERIMENT_1_DIR, models)
    payload = validate_part_a_cache(PART_A_CACHE_PATH)
    test_records = payload["records"]["test"]
    device = resolve_device()
    seeds = [SEED + offset for offset in range(n_seeds)]
    levels = [pct for pct in ROBUSTNESS_RETAINED_PCTS if pct != 100]

    datasets: dict[tuple[int, int | None], Any] = {
        (100, None): materialize_records(test_records, "test")
    }
    for pct in levels:
        for seed in seeds:
            datasets[(pct, seed)] = materialize_records(
                thin_records(test_records, pct, seed), "test"
            )
    print(f"Prepared {len(datasets)} test variants.", flush=True)

    paths = checkpoint_paths(EXPERIMENT_1_DIR, models)
    rows: list[dict[str, Any]] = []
    for model_key in models:
        set_seed(SEED)
        model = load_experiment_1_model(model_key, paths[model_key], device)
        forward_fn = get_forward_adapter(model_key)
        for (pct, seed), dataset in datasets.items():
            metrics = evaluate(model, make_eval_loader(dataset, model_key, device), forward_fn, device)
            rows.append(
                {
                    "model": PRIMARY_MODEL_NAMES[model_key],
                    "retained_pct": pct,
                    "seed": "" if seed is None else seed,
                    "auroc": float(metrics["auroc"]),
                    "auprc": float(metrics["auprc"]),
                    "brier_score": float(metrics["brier_score"]),
                    "sample_count": int(metrics["sample_count"]),
                }
            )
            print(
                f"{model_key} retained={pct}% seed={seed} "
                f"AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f}",
                flush=True,
            )
        del model

    checks = check_against_experiment_1(rows, EXPERIMENT_1_DIR)
    summary = summarize(rows)
    write_csv(output_dir / "robustness_results.csv", rows, RAW_FIELDS)
    write_csv(output_dir / "robustness_summary.csv", summary, SUMMARY_FIELDS)
    write_json(
        output_dir / "run_metadata.json",
        {
            "experiment": 2,
            "mode": "zero_shot",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "drop_seeds": seeds,
            "retained_percentages": [100, *levels],
            "models": models,
            "device": device,
            "configuration": configuration_snapshot(),
            "package_versions": package_versions(["torch", "numpy", "scikit-learn", "torchdiffeq", "pykan", "ncps", "matplotlib"]),
            "zero_shot_no_retraining": True,
            "experiment_1_consistency": checks,
            "source_checkpoints": {k: str(v) for k, v in paths.items()},
        },
    )
    return summary


def retrain_settings() -> TrainingSettings:
    from src.config import REDUCED_MAX_EPOCHS

    return TrainingSettings(
        max_epochs=REDUCED_MAX_EPOCHS,
        min_epochs=5,
        patience=4,
        min_delta=0.0,
        checkpoint_every=5,
        log_every_epoch=True,
    )


def run_retrain(
    models: list[str],
    levels: list[int],
    resume: bool,
    output_dir: Path,
    max_run_minutes: float | None,
) -> None:
    from src.config import (
        PART_A_CACHE_PATH,
        SEED,
        resolve_device,
        validate_config,
    )
    from src.dataset import materialize_records
    from src.utils import write_json

    validate_config()
    if any(not 0 < pct < 100 for pct in levels):
        raise ValueError("--levels must be percentages in (0, 100).")
    payload = validate_part_a_cache(PART_A_CACHE_PATH)
    device = resolve_device()
    runs_dir = output_dir / "retrain" / "runs"
    splits_cache: dict[int, dict[str, Any]] = {}

    for model_key in models:
        for pct in levels:
            run_name = f"{model_key}_retain{pct}"
            result_path = runs_dir / f"{run_name}.json"
            if result_path.is_file():
                print(f"Skip {run_name}: {result_path} already exists.", flush=True)
                continue
            if pct not in splits_cache:
                splits_cache[pct] = {
                    split: materialize_records(
                        thin_records(payload["records"][split], pct, SEED), split
                    )
                    for split in ("train", "validation", "test")
                }
            row = train_and_test_model(
                model_key=model_key,
                device=device,
                checkpoint_dir=output_dir / "retrain" / "checkpoints" / run_name,
                settings=retrain_settings(),
                resume=resume,
                materialized_splits=splits_cache[pct],
                run_name=run_name,
                extra_config={"EXP2_RETAINED_PCT": pct, "EXP2_DROP_SEED": SEED},
                max_run_minutes=max_run_minutes,
            )
            row["retained_pct"] = pct
            row["drop_seed"] = SEED
            row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(result_path, row)
            print(f"Run saved: {result_path}", flush=True)


def _read_summary(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "robustness_summary.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {**row, "retained_pct": int(row["retained_pct"]), "auroc_mean": float(row["auroc_mean"]), "auroc_std": float(row["auroc_std"])}
            for row in csv.DictReader(handle)
        ]


def _read_retrain(output_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "retrain" / "runs").glob("*.json"))
    ]


def refresh_figures(output_dir: Path) -> None:
    """Zero-shot figure with error bars, plus the combined figure if retrains exist."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.config import EXPERIMENT_1_DIR

    summary = _read_summary(output_dir)
    retrain = _read_retrain(output_dir)
    capped_models = {
        r["model_key"].rsplit("_retain", 1)[0] for r in retrain if r.get("time_capped")
    }
    headline: dict[str, Any] = {}
    headline_path = EXPERIMENT_1_DIR / "headline_results.json"
    if headline_path.is_file():
        headline = {r.get("model_key"): r for r in json.loads(headline_path.read_text(encoding="utf-8"))}

    def draw(include_retrain: bool, path: Path) -> None:
        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        for model_key, name in PRIMARY_MODEL_NAMES.items():
            color, marker = COLORS[model_key], MARKERS[model_key]
            zero = sorted((r for r in summary if r["model"] == name), key=lambda r: r["retained_pct"])
            if zero:
                axis.errorbar(
                    [r["retained_pct"] for r in zero],
                    [r["auroc_mean"] for r in zero],
                    yerr=[r["auroc_std"] for r in zero],
                    color=color, marker=marker, linestyle="-", capsize=3,
                    label=f"{name} (zero-shot)" if include_retrain else name,
                )
            if include_retrain:
                points = {
                    int(r["retained_pct"]): float(r["auroc"])
                    for r in retrain
                    if r["model_key"] == f"{model_key}_retain{r['retained_pct']}"
                }
                if points and model_key in headline:
                    points[100] = float(headline[model_key]["auroc"])
                if points:
                    xs = sorted(points)
                    axis.plot(
                        xs, [points[x] for x in xs], color=color, marker=marker,
                        linestyle="--", mfc="none",
                        label=f"{name} (retrained)" + ("†" if model_key in capped_models else ""),
                    )
        axis.set_xlabel("% observation timestamps retained")
        axis.set_ylabel("Test AUROC")
        axis.set_xticks([25, 50, 75, 100])
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        if include_retrain and capped_models:
            figure.subplots_adjust(bottom=0.17)
            figure.text(
                0.01, 0.01,
                "† at least one retrained run was stopped by the wall-clock cap "
                "(best checkpoint tested). The 100% retrained point is the "
                "Experiment 1 result.",
                fontsize=7, va="bottom",
            )
        figure.savefig(path, dpi=150)
        plt.close(figure)

    if summary:
        draw(False, output_dir / "robustness_auroc.png")
    if summary and retrain:
        draw(True, output_dir / "robustness_zero_shot_vs_retrained.png")
    if retrain:
        from src.utils import write_csv

        fields = ["model", "retained_pct", "auroc", "auprc", "brier_score", "sample_count", "trainable_parameters", "best_epoch", "epochs_completed", "time_capped", "max_run_minutes", "patience", "max_epochs", "mean_train_seconds_per_epoch", "mean_train_val_seconds_per_epoch"]
        write_csv(output_dir / "retrain_results.csv", [{f: r.get(f) for f in fields} for r in retrain], fields)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from src.config import EXPERIMENT_1_DIR, EXPERIMENT_2_DIR, PART_A_CACHE_PATH

    apply_kan_grid_range(args.kan_grid_range)
    output_dir = args.output_dir or EXPERIMENT_2_DIR

    if args.dry_run:
        plan = validate_dry_run(PART_A_CACHE_PATH, EXPERIMENT_1_DIR, args.models)
        print("Experiment 2 dry run", flush=True)
        print(f"Validated checkpoints: {', '.join(plan['models'])}", flush=True)
        print(f"Retained levels: {plan['retained_percentages']} x {args.seeds} drop seeds", flush=True)
        print(f"Retrain models: {', '.join(args.retrain_models)} at levels {args.levels} (cap {resolve_max_run_minutes(args.max_run_minutes)} min)", flush=True)
        print(f"Output directory: {output_dir} (nothing written)", flush=True)
        return 0

    with tee_output_to_log(output_dir / "train.log"):
        if args.run:
            run_zero_shot(args.models, args.seeds, output_dir)
        else:
            run_retrain(
                args.retrain_models,
                args.levels,
                args.resume,
                output_dir,
                resolve_max_run_minutes(args.max_run_minutes),
            )
        refresh_figures(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

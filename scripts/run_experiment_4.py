"""Experiment 4: MLP-ODE versus KAN-ODE parameter efficiency.

Every trained run goes through the Experiment 1 training routine
(``train_and_test_model``): materialized loaders, AUPRC early stopping,
gradient clipping, NFE logging, seed 8893, checkpoint restore, one test
evaluation. Each run writes its own JSON, so a disconnect loses at most one
run and finished runs are skipped on relaunch. The h=20 runs are imported from
Experiment 1 when an identical finished result exists (disable: --no-reuse).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment_1 import (  # noqa: E402
    TrainingSettings,
    apply_kan_grid_range,
    configuration_snapshot,
    materialize_experiment_datasets,
    resolve_max_run_minutes,
    tee_output_to_log,
    train_and_test_model,
    validate_part_a_cache,
)
from src.matching import (  # noqa: E402
    build_kan,
    format_match,
    match_kan_to_mlp,
)

PROTOCOL_PATIENCE = 4
FAMILIES = ("mlp", "kan")
DISPLAY = {"mlp": "MLP-ODE", "kan": "KAN-ODE"}
ADAPTER_KEY = {"mlp": "mlp_neural_ode", "kan": "kan_neural_ode"}

RESULT_FIELDS = [
    "family",
    "hidden_dim",
    "kan_hidden_width",
    "kan_grid",
    "kan_spline_order",
    "kan_grid_range",
    "trainable_parameters",
    "mlp_target_parameters",
    "parameter_gap",
    "parameter_gap_pct",
    "test_auroc",
    "test_auprc",
    "test_brier_score",
    "sample_count",
    "best_validation_auprc",
    "best_epoch",
    "epochs_completed",
    "time_capped",
    "max_run_minutes",
    "mean_train_seconds_per_epoch",
    "mean_train_val_seconds_per_epoch",
    "patience",
    "max_epochs",
    "imported_from_experiment_1",
    "patience_note",
    "run_name",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 4: MLP-ODE vs KAN-ODE at matched capacities."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print the plan; write nothing.")
    mode.add_argument("--match", action="store_true", help="Write matching.json only.")
    mode.add_argument("--run", action="store_true", help="Train (or import) the selected run(s).")
    mode.add_argument("--aggregate", action="store_true", help="Write CSV and plot from runs/*.json.")
    parser.add_argument("--family", choices=FAMILIES, default=None, help="With --run: only this family (default both).")
    parser.add_argument("--hidden-dim", type=int, default=None, help="With --run: only this MLP hidden dim (default 8, 20, 40).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: outputs/experiment_4.")
    parser.add_argument("--resume", action="store_true", help="Resume an unfinished run from latest.pt.")
    parser.add_argument("--no-reuse", action="store_true", help="Do not import h=20 results from Experiment 1; retrain them.")
    parser.add_argument("--max-run-minutes", type=float, default=None, help="Wall-clock cap per run (default 20; 0 disables).")
    parser.add_argument("--kan-grid-range", nargs=2, type=float, metavar=("LOW", "HIGH"), default=None)
    parser.add_argument("--first-epoch-timeout", type=float, default=None, help="Abort with exit code 3 if a first epoch exceeds this many seconds.")
    return parser


def protocol_settings() -> TrainingSettings:
    """Experiment 4 protocol: AUPRC, patience 4, min 5, max 25 epochs."""
    from src.config import REDUCED_MAX_EPOCHS

    return TrainingSettings(
        max_epochs=REDUCED_MAX_EPOCHS,
        min_epochs=5,
        patience=PROTOCOL_PATIENCE,
        min_delta=0.0,
        checkpoint_every=5,
        log_every_epoch=True,
    )


def compute_matching() -> dict[str, Any]:
    """Direct search (src/matching.py) for every MLP hidden size."""
    from src.config import PARAMETER_EFFICIENCY_HIDDEN_DIMS

    entries = []
    for hidden_dim in PARAMETER_EFFICIENCY_HIDDEN_DIMS:
        entry = match_kan_to_mlp(hidden_dim)
        print(format_match(entry), flush=True)
        entries.append(entry)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "counts_trainable_parameters_only": True,
        "entries": entries,
    }


def load_or_compute_matching(output_dir: Path, write: bool) -> dict[str, Any]:
    from src.utils import write_json

    path = output_dir / "matching.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    matching = compute_matching()
    if write:
        write_json(path, matching)
    return matching


def _entry(matching: dict[str, Any], hidden_dim: int) -> dict[str, Any]:
    for entry in matching["entries"]:
        if entry["hidden_dim"] == hidden_dim:
            return entry
    raise KeyError(f"No matching entry for hidden_dim={hidden_dim}.")


def make_result(
    family: str,
    hidden_dim: int,
    entry: dict[str, Any],
    row: dict[str, Any],
    imported: bool,
    patience_note: str | None,
) -> dict[str, Any]:
    from src import config

    target = entry["mlp_parameters"]
    chosen = entry["chosen"]
    gap = row["trainable_parameters"] - target
    return {
        "family": DISPLAY[family],
        "hidden_dim": hidden_dim,
        "kan_hidden_width": chosen["hidden_width"] if family == "kan" else None,
        "kan_grid": chosen["grid"] if family == "kan" else None,
        "kan_spline_order": entry["kan_spline_order"] if family == "kan" else None,
        "kan_grid_range": list(config.KAN_GRID_RANGE) if family == "kan" else None,
        "trainable_parameters": row["trainable_parameters"],
        "mlp_target_parameters": target,
        "parameter_gap": gap,
        "parameter_gap_pct": 100.0 * gap / target,
        "test_auroc": row["auroc"],
        "test_auprc": row["auprc"],
        "test_brier_score": row["brier_score"],
        "sample_count": row["sample_count"],
        "best_validation_auprc": row["best_validation_auprc"],
        "best_epoch": row["best_epoch"],
        "epochs_completed": row["epochs_completed"],
        "time_capped": bool(row.get("time_capped", False)),
        "max_run_minutes": row.get("max_run_minutes"),
        "mean_train_seconds_per_epoch": row["mean_train_seconds_per_epoch"],
        "mean_train_val_seconds_per_epoch": row["mean_train_val_seconds_per_epoch"],
        "patience": row["patience"],
        "max_epochs": row["max_epochs"],
        "imported_from_experiment_1": imported,
        "patience_note": patience_note,
        "run_name": f"{family}_h{hidden_dim}",
        "configuration": configuration_snapshot(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def try_reuse(
    family: str,
    hidden_dim: int,
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    """Return an Experiment 1 result row if it is the same h=20 configuration."""
    from src import config

    if hidden_dim != config.ODE_HIDDEN_DIM:
        return None
    path = config.EXPERIMENT_1_DIR / "headline_results.json"
    if not path.is_file():
        return None
    key = ADAPTER_KEY[family]
    row = next(
        (r for r in json.loads(path.read_text(encoding="utf-8")) if r.get("model_key") == key),
        None,
    )
    if row is None:
        return None
    if family == "mlp":
        if row["trainable_parameters"] != entry["mlp_parameters"]:
            return None
    else:
        chosen = entry["chosen"]
        same_range = [float(v) for v in (row.get("kan_grid_range") or [])] == [
            float(v) for v in config.KAN_GRID_RANGE
        ]
        if (
            row.get("kan_hidden_width") != chosen["hidden_width"]
            or row.get("kan_grid") != chosen["grid"]
            or row["trainable_parameters"] != chosen["parameters"]
            or not same_range
        ):
            return None
    return row


def run_one(
    family: str,
    hidden_dim: int,
    output_dir: Path,
    matching: dict[str, Any],
    materialized: dict[str, Any] | None,
    device: str,
    resume: bool,
    first_epoch_timeout: float | None,
    max_run_minutes: float | None,
    reuse: bool,
) -> dict[str, Any] | None:
    from src import config
    from src.config import ODE_WIDTH
    from src.models.mlp_neural_ode import MLPNeuralODE
    from src.utils import write_json

    run_name = f"{family}_h{hidden_dim}"
    result_path = output_dir / "runs" / f"{run_name}.json"
    if result_path.is_file():
        print(f"Skip {run_name}: {result_path} already exists.", flush=True)
        return None

    entry = _entry(matching, hidden_dim)
    if reuse:
        row = try_reuse(family, hidden_dim, entry)
        if row is not None:
            note = (
                f"Imported from Experiment 1; trained with patience {row['patience']} "
                f"and max_epochs {row['max_epochs']} (Experiment 4 default: patience "
                f"{PROTOCOL_PATIENCE}, max_epochs {config.REDUCED_MAX_EPOCHS})."
            )
            result = make_result(family, hidden_dim, entry, row, True, note)
            write_json(result_path, result)
            print(f"Imported {run_name} from Experiment 1: {result_path}", flush=True)
            return result

    assert materialized is not None
    extra_config: dict[str, Any] = {"EXP4_FAMILY": family, "EXP4_HIDDEN_DIM": hidden_dim}
    if family == "mlp":
        def factory() -> Any:
            return MLPNeuralODE(hidden_dim=hidden_dim, width=ODE_WIDTH)
    else:
        width, grid = entry["chosen"]["hidden_width"], entry["chosen"]["grid"]
        extra_config.update({"EXP4_KAN_WIDTH": width, "EXP4_KAN_GRID": grid})

        def factory() -> Any:
            return build_kan(hidden_dim, width, grid)

    row = train_and_test_model(
        model_key=ADAPTER_KEY[family],
        device=device,
        checkpoint_dir=output_dir / "checkpoints" / run_name,
        settings=protocol_settings(),
        resume=resume,
        materialized_splits=materialized,
        model_factory=factory,
        run_name=run_name,
        display_name=DISPLAY[family],
        extra_config=extra_config,
        first_epoch_timeout=first_epoch_timeout,
        max_run_minutes=max_run_minutes,
    )
    result = make_result(family, hidden_dim, entry, row, False, None)
    write_json(result_path, result)
    print(f"Run saved: {result_path}", flush=True)
    return result


def aggregate(output_dir: Path) -> list[dict[str, Any]]:
    """Write the CSV and AUROC-vs-parameters plot from every finished run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils import write_csv

    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "runs").glob("*.json"))
    ]
    if not rows:
        raise FileNotFoundError(f"No run JSON files in {output_dir / 'runs'}.")
    rows.sort(key=lambda row: (row["family"], row["hidden_dim"]))
    write_csv(
        output_dir / "parameter_efficiency_results.csv",
        [{field: row.get(field) for field in RESULT_FIELDS} for row in rows],
        RESULT_FIELDS,
    )
    styles = {
        "MLP-ODE": {"marker": "o", "linestyle": "-", "color": "#0072B2"},
        "KAN-ODE": {"marker": "s", "linestyle": "--", "color": "#D55E00"},
    }
    figure, axis = plt.subplots(figsize=(8, 5.4))
    any_capped = any(row.get("time_capped") for row in rows)
    any_imported = any(row.get("imported_from_experiment_1") for row in rows)
    for family, style in styles.items():
        family_rows = sorted(
            (row for row in rows if row["family"] == family),
            key=lambda row: row["trainable_parameters"],
        )
        if not family_rows:
            continue
        axis.plot(
            [row["trainable_parameters"] for row in family_rows],
            [row["test_auroc"] for row in family_rows],
            label=family,
            **style,
        )
        for row in family_rows:
            tags = ("†" if row.get("time_capped") else "") + (
                "‡" if row.get("imported_from_experiment_1") else ""
            )
            axis.annotate(
                f"h={row['hidden_dim']}{tags}",
                (row["trainable_parameters"], row["test_auroc"]),
                textcoords="offset points",
                xytext=(4, 6),
                fontsize=8,
            )
    axis.set_xscale("log")
    axis.set_xlabel("Trainable parameters (log scale)")
    axis.set_ylabel("Test AUROC")
    axis.set_title("MLP-ODE vs KAN-ODE: AUROC vs parameter count")
    axis.grid(True, alpha=0.3)
    axis.legend()
    footnotes = []
    if any_capped:
        footnotes.append("† stopped by the wall-clock cap (best checkpoint tested)")
    if any_imported:
        footnotes.append("‡ imported from Experiment 1 (different patience / max epochs)")
    figure.tight_layout()
    if footnotes:
        figure.subplots_adjust(bottom=0.2)
        figure.text(0.01, 0.01, "\n".join(footnotes), fontsize=7, va="bottom")
    figure.savefig(output_dir / "parameter_efficiency_auroc_vs_params.png", dpi=150)
    plt.close(figure)
    print(f"Aggregated {len(rows)} runs into {output_dir}.", flush=True)
    return rows


def planned_runs(args: argparse.Namespace) -> list[tuple[str, int]]:
    from src.config import PARAMETER_EFFICIENCY_HIDDEN_DIMS

    families = (args.family,) if args.family else FAMILIES
    dims = (
        (args.hidden_dim,)
        if args.hidden_dim is not None
        else PARAMETER_EFFICIENCY_HIDDEN_DIMS
    )
    return [(family, dim) for family in families for dim in dims]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from src.config import (
        EXPERIMENT_4_DIR,
        PARAMETER_EFFICIENCY_HIDDEN_DIMS,
        PART_A_CACHE_PATH,
        resolve_device,
        validate_config,
    )

    apply_kan_grid_range(args.kan_grid_range)
    validate_config()
    output_dir = args.output_dir or EXPERIMENT_4_DIR
    if args.hidden_dim is not None and args.hidden_dim not in PARAMETER_EFFICIENCY_HIDDEN_DIMS:
        raise SystemExit(
            f"--hidden-dim must be one of {list(PARAMETER_EFFICIENCY_HIDDEN_DIMS)}."
        )

    if args.dry_run:
        validate_part_a_cache(PART_A_CACHE_PATH)
        print("Experiment 4 dry run", flush=True)
        matching = compute_matching()
        for family, dim in planned_runs(args):
            reuse = (
                not args.no_reuse
                and try_reuse(family, dim, _entry(matching, dim)) is not None
            )
            print(f"Planned run: {family}_h{dim}{' (import from Experiment 1)' if reuse else ''}", flush=True)
        print(f"Output directory: {output_dir} (nothing written)", flush=True)
        return 0

    if args.match:
        from src.utils import write_json

        write_json(output_dir / "matching.json", compute_matching())
        return 0

    if args.aggregate:
        with tee_output_to_log(output_dir / "train.log"):
            aggregate(output_dir)
        return 0

    with tee_output_to_log(output_dir / "train.log"):
        validate_part_a_cache(PART_A_CACHE_PATH)
        matching = load_or_compute_matching(output_dir, write=True)
        device = resolve_device()
        runs = planned_runs(args)
        needs_training = any(
            not (output_dir / "runs" / f"{family}_h{dim}.json").is_file()
            and (
                args.no_reuse
                or try_reuse(family, dim, _entry(matching, dim)) is None
            )
            for family, dim in runs
        )
        materialized = None
        if needs_training:
            materialized, seconds = materialize_experiment_datasets(PART_A_CACHE_PATH)
            print(f"Materialized datasets in {seconds:.2f}s.", flush=True)
        try:
            for family, dim in runs:
                run_one(
                    family,
                    dim,
                    output_dir,
                    matching,
                    materialized,
                    device,
                    args.resume,
                    args.first_epoch_timeout,
                    resolve_max_run_minutes(args.max_run_minutes),
                    not args.no_reuse,
                )
        except TimeoutError as error:
            print(str(error), flush=True)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Experiment 3: post-training interpretation of the KAN-ODE vector field.

Probes the vector field on latent states the solver actually visits, at times
spread uniformly over [0, 48] h. Every stage is wrapped so a failure (for
example symbolic fitting) is logged to ``errors.log`` and the remaining
artifacts are still produced.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.metadata
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment_1 import (  # noqa: E402
    apply_kan_grid_range,
    build_model,
    configuration_snapshot,
    load_model_state,
    tee_output_to_log,
    validate_part_a_cache,
)

N_PROBE_TIMES = 12
MAX_PROBE_POINTS = 200
SCRATCH_DPI = 60  # pykan writes one scratch PNG per edge; keep them small
TOP_EDGES_PER_DIM = 4
WRITEUP_WORDS = (150, 300)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate inputs; write nothing.")
    mode.add_argument("--run", action="store_true", help="Create Experiment 3 artifacts.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: outputs/experiment_3.")
    parser.add_argument("--kan-grid-range", nargs=2, type=float, metavar=("LOW", "HIGH"), default=None)
    return parser


def kan_input_labels(hidden_dim: int) -> list[str]:
    """Labels for the KAN inputs ``[latent state..., t/48]``."""
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be positive.")
    return [*[f"latent_state_{index}" for index in range(hidden_dim)], "t/48"]


def require_pykan_capabilities(network: Any) -> str:
    try:
        version = importlib.metadata.version("pykan")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("PyKAN is not installed; Experiment 3 requires it.") from exc
    required = ["get_act", "attribute", "plot", "prune", "auto_symbolic", "symbolic_formula"]
    missing = [m for m in required if not callable(getattr(network, m, None))]
    if not hasattr(type(network), "feature_score"):
        missing.append("feature_score")
    if missing:
        raise RuntimeError(f"PyKAN {version} lacks: " + ", ".join(missing))
    return version


def validate_dry_run(cache_path: Path, checkpoint_path: Path, output_dir: Path) -> dict[str, Any]:
    from src.config import ODE_HIDDEN_DIM, validate_config

    validate_config()
    payload = validate_part_a_cache(cache_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Experiment 1 KAN checkpoint is absent: {checkpoint_path}")
    model = build_model("kan_neural_ode")
    model.load_state_dict(load_model_state(checkpoint_path, map_location="cpu"))
    version = require_pykan_capabilities(model.ode_func.network)
    return {
        "checkpoint": str(checkpoint_path),
        "test_patients": len(payload["records"]["test"]),
        "input_labels": kan_input_labels(ODE_HIDDEN_DIM),
        "pykan_version": version,
        "output_dir": str(output_dir),
    }


def build_probe_set(model: Any, device: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Probe rows ``[h(t), t/48]`` with ``t`` uniform in [0, 48] h.

    ``h(t)`` is the ODE state of each held-out patient at the query times, so
    the probe covers states the vector field is actually evaluated on. A seeded
    subsample of at most ``MAX_PROBE_POINTS`` rows is kept. Returns the CPU
    probe matrix, the sorted query times, and the number of candidate rows.
    """
    from torch.utils.data import DataLoader

    from src.config import BATCH_SIZE, MAX_HOURS, NUM_WORKERS, PART_A_CACHE_PATH, SEED
    from src.dataset import materialize_records, materialized_collator

    payload = validate_part_a_cache(PART_A_CACHE_PATH)
    dataset = materialize_records(payload["records"]["test"], "test")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=materialized_collator("kan_neural_ode"),
    )
    generator = torch.Generator().manual_seed(SEED)
    query_times = torch.sort(torch.rand(N_PROBE_TIMES, generator=generator) * MAX_HOURS).values
    query_times = query_times.clamp_min(1e-3)
    if not bool(torch.all(query_times[1:] > query_times[:-1])):
        raise RuntimeError("Sampled probe times are not strictly increasing.")

    rows: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            states = model.latent_trajectory(
                batch["values"],
                batch["mask"],
                batch["valid_time_mask"],
                query_times.to(device),
                times=batch["times"],
                delta_t=batch["delta_t"],
            )
            steps, size, _ = states.shape
            time_column = (query_times.to(device) / MAX_HOURS).view(steps, 1, 1).expand(steps, size, 1)
            rows.append(torch.cat([states, time_column], dim=-1).reshape(steps * size, -1).cpu())
    probe = torch.cat(rows, dim=0)
    if not torch.isfinite(probe).all():
        raise RuntimeError("Probe set contains non-finite latent states.")
    candidates = int(probe.shape[0])
    if candidates > MAX_PROBE_POINTS:
        keep = torch.randperm(candidates, generator=generator)[:MAX_PROBE_POINTS]
        probe = probe[keep.sort().values]
    return probe, query_times, candidates


def _log_error(output_dir: Path, stage: str) -> None:
    text = traceback.format_exc()
    print(f"STAGE FAILED ({stage}); continuing.\n{text}", flush=True)
    with (output_dir / "errors.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now(timezone.utc).isoformat()}] {stage}\n{text}\n")


def _active_edges(network: Any) -> int:
    return int(sum(float(layer.mask.sum()) for layer in network.act_fun))


@contextlib.contextmanager
def _low_dpi_scratch_pngs():
    """Cap the dpi of pykan's per-edge scratch PNGs (it hard-codes 400)."""
    import matplotlib.pyplot as plt

    original = plt.savefig

    def savefig(*args: Any, **kwargs: Any) -> Any:
        kwargs["dpi"] = min(int(kwargs.get("dpi", SCRATCH_DPI)), SCRATCH_DPI)
        return original(*args, **kwargs)

    plt.savefig = savefig
    try:
        yield
    finally:
        plt.savefig = original


def save_edge_plot(
    network: Any,
    labels: list[str],
    output_dir: Path,
    name: str,
    title: str,
) -> Path:
    """Save pykan's network plot (all edges, beta=100) to ``name``.png."""
    import matplotlib.pyplot as plt

    scratch = output_dir / "kan_edge_functions" / f"_{name}_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    out_vars = [f"dh_{i}/dt" for i in range(network.width_out[-1])]
    with _low_dpi_scratch_pngs():  # one pykan plot call, reduced size
        network.plot(
            folder=str(scratch),
            beta=100,
            metric="backward",
            scale=0.35,
            in_vars=labels if network.width_in[0] == len(labels) else None,
            out_vars=out_vars,
            title=title,
        )
    path = output_dir / "kan_edge_functions" / f"{name}.png"
    plt.gcf().savefig(path, dpi=100, bbox_inches="tight")
    plt.close("all")
    shutil.rmtree(scratch, ignore_errors=True)  # pykan's per-edge scratch PNGs
    return path


def save_univariate_plot(
    network: Any,
    probe: torch.Tensor,
    top_indices: list[int],
    labels: list[str],
    output_dir: Path,
) -> Path:
    """Plot the first-layer edge functions of the two top-ranked inputs."""
    import matplotlib.pyplot as plt

    layer = network.act_fun[0]
    k = int(network.k)
    low, high = float(layer.grid[0, k]), float(layer.grid[0, -k - 1])
    margin = 0.25 * (high - low)
    grid_x = torch.linspace(low - margin, high + margin, 200)
    baseline = probe.median(dim=0).values
    figure, axes = plt.subplots(1, len(top_indices), figsize=(6.5 * len(top_indices), 4.5))
    axes = [axes] if len(top_indices) == 1 else list(axes)
    for axis, index in zip(axes, top_indices):
        inputs = baseline.repeat(len(grid_x), 1)
        inputs[:, index] = grid_x
        with torch.no_grad():
            _, _, postacts, _ = layer(inputs)  # (200, out, in)
        curves = postacts[:, :, index]
        spread = curves.max(dim=0).values - curves.min(dim=0).values
        for node in torch.topk(spread, min(TOP_EDGES_PER_DIM, spread.numel())).indices.tolist():
            axis.plot(grid_x, curves[:, node], label=f"edge to hidden node {node}")
        axis.axvspan(low, high, color="grey", alpha=0.12, label="spline grid")
        density = axis.twinx()
        density.hist(probe[:, index].numpy(), bins=40, color="grey", alpha=0.25)
        density.set_yticks([])
        axis.set_title(f"{labels[index]} (others at probe median)")
        axis.set_xlabel("input value")
        axis.set_ylabel("edge output")
        axis.legend(fontsize=7)
    figure.tight_layout()
    path = output_dir / "top2_univariate_functions.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def build_writeup(facts: dict[str, Any]) -> str:
    d1, d2 = facts["top_labels"]
    s1, s2 = facts["top_scores"]
    if facts["symbolic_ok"]:
        symbolic = (
            "Symbolic fitting with the library x, x^2, exp, sin, tanh and abs "
            "produced closed forms (saved in symbolic_formulas.txt)."
        )
    else:
        symbolic = (
            "Symbolic fitting failed or was incomplete (see errors.log), so "
            "no closed-form claim is made."
        )
    grid_note = (
        f"{facts['outside_grid_pct']:.1f}% of probed states fell outside the "
        f"spline grid [{facts['grid_low']:.2f}, {facts['grid_high']:.2f}], "
        "where splines fade out and the SiLU base branch dominates, so edge "
        "shapes there should be read cautiously."
    )
    return f"""# Experiment 3 write-up (draft)

The KAN-ODE vector field takes the {facts['hidden_dim']} latent state coordinates plus scaled time t/48. It never sees raw ICU variables: the coordinates are outputs of a learned encoder, so a name like latent_state_7 has no clinical meaning by itself. Attribution therefore shows which latent directions drive the dynamics, not which vital sign matters.

I probed {facts['n_probe']:,} states (a seeded subsample of {facts['n_candidates']:,} held-out patient-time pairs, with {facts['n_times']} times drawn uniformly from [0, 48] h). The two highest-scoring inputs were {d1} (score {s1:.3g}) and {d2} (score {s2:.3g}). {grid_note} Before pruning the network had {facts['edges_before']} active edges; after pruning, {facts['edges_after']} remained. {symbolic}

The MLP-ODE has no comparable view: its tanh layers mix all inputs, so there is no per-edge function to plot. The KAN's advantage is exactly these one-dimensional curves, but only in latent coordinates. The LTC/CfC model is interpretable differently: each neuron carries a learned, input-dependent time constant that shows how quickly units respond to irregular gaps, yet it also acts on latent units. Neither model maps directly to raw variables, so clinical claims would need an observation-level attribution.

Check the saved figures and edit any sentence they do not support before submitting.
"""


def count_words(markdown: str) -> int:
    body = [line for line in markdown.splitlines() if not line.startswith("#")]
    return len(" ".join(body).split())


def run_experiment(output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")

    from src.config import (
        EXPERIMENT_1_DIR,
        MAX_HOURS,
        PART_A_CACHE_PATH,
        SEED,
        resolve_device,
    )
    from src.models.kan_neural_ode import SYMBOLIC_LIBRARY, sync_kan_device
    from src.utils import package_versions, set_seed, write_json, write_text

    checkpoint_path = EXPERIMENT_1_DIR / "checkpoints" / "kan_neural_ode" / "best.pt"
    validate_dry_run(PART_A_CACHE_PATH, checkpoint_path, output_dir)
    set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()

    model = build_model("kan_neural_ode").to(device)
    model.load_state_dict(load_model_state(checkpoint_path, map_location=device))
    model.enable_interpretability_mode()
    model.ode_func.network.to(device)
    sync_kan_device(model.ode_func.network)

    labels = kan_input_labels(model.hidden_dim)
    probe, query_times, n_candidates = build_probe_set(model, device)
    print(f"Probe set: {tuple(probe.shape)} rows from t in [0, {MAX_HOURS:g}] h.", flush=True)

    # All analysis runs on a CPU copy so plotting and pruning never mix devices.
    network = copy.deepcopy(model.ode_func.network).to("cpu")
    sync_kan_device(network)
    network.ckpt_path = str(output_dir / "pykan_ckpt")  # contain pykan side effects
    k = int(network.k)
    grid_low = float(network.act_fun[0].grid[0, k])
    grid_high = float(network.act_fun[0].grid[0, -k - 1])
    outside = float(((probe[:, :-1] < grid_low) | (probe[:, :-1] > grid_high)).float().mean()) * 100.0

    status: dict[str, str] = {}
    facts: dict[str, Any] = {
        "hidden_dim": model.hidden_dim,
        "n_probe": int(probe.shape[0]),
        "n_candidates": n_candidates,
        "n_times": N_PROBE_TIMES,
        "outside_grid_pct": outside,
        "grid_low": grid_low,
        "grid_high": grid_high,
        "edges_before": _active_edges(network),
        "edges_after": None,
        "symbolic_ok": False,
        "top_labels": ["?", "?"],
        "top_scores": [float("nan")] * 2,
    }

    scores: list[float] = []
    top_indices: list[int] = []
    try:
        network.get_act(probe)
        network.attribute(plot=False)
        scores = [float(v) for v in network.feature_score.detach().reshape(-1).cpu().tolist()]
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        top_indices = ranked[:2]
        facts["top_labels"] = [labels[i] for i in top_indices]
        facts["top_scores"] = [scores[i] for i in top_indices]
        write_text(
            output_dir / "top_two_input_dimensions.csv",
            "rank,input_index,input_label,attribution_value\n"
            + "".join(f"{r},{i},{labels[i]},{scores[i]}\n" for r, i in enumerate(top_indices, 1)),
        )
        status["attribution"] = "ok"
    except Exception:  # noqa: BLE001
        _log_error(output_dir, "attribution")
        status["attribution"] = "failed"

    try:
        path = save_edge_plot(network, labels, output_dir, "before_prune", "KAN edges before pruning")
        status["plot_before_prune"] = f"ok: {path}"
    except Exception:  # noqa: BLE001
        _log_error(output_dir, "plot_before_prune")
        status["plot_before_prune"] = "failed"

    if top_indices:
        try:
            path = save_univariate_plot(network, probe, top_indices, labels, output_dir)
            status["univariate_plot"] = f"ok: {path}"
        except Exception:  # noqa: BLE001
            _log_error(output_dir, "univariate_plot")
            status["univariate_plot"] = "failed"

    pruned = network
    try:
        pruned = network.prune()
        pruned.symbolic_enabled = True
        pruned.save_act = True
        pruned.get_act(probe)
        facts["edges_after"] = _active_edges(pruned)
        status["prune"] = f"ok: width {[list(w) for w in pruned.width]}"
    except Exception:  # noqa: BLE001
        _log_error(output_dir, "prune")
        status["prune"] = "failed"
        facts["edges_after"] = facts["edges_before"]

    try:
        path = save_edge_plot(pruned, labels, output_dir, "after_prune", "KAN edges after pruning")
        status["plot_after_prune"] = f"ok: {path}"
    except Exception:  # noqa: BLE001
        _log_error(output_dir, "plot_after_prune")
        status["plot_after_prune"] = "failed"

    try:
        pruned.auto_symbolic(lib=SYMBOLIC_LIBRARY, verbose=0)
        formulas = pruned.symbolic_formula(var=labels)
        expressions = formulas[0] if isinstance(formulas, (tuple, list)) else formulas
        write_text(
            output_dir / "symbolic_formulas.txt",
            "\n".join(f"dh_{i}/dt = {expression}" for i, expression in enumerate(expressions)) + "\n",
        )
        facts["symbolic_ok"] = True
        status["symbolic"] = "ok"
    except Exception:  # noqa: BLE001
        _log_error(output_dir, "symbolic")
        status["symbolic"] = "failed"

    writeup = build_writeup(facts)
    words = count_words(writeup)
    write_text(output_dir / "writeup.md", writeup)
    if not WRITEUP_WORDS[0] <= words <= WRITEUP_WORDS[1]:
        print(f"WARNING: writeup.md has {words} words (target {WRITEUP_WORDS}).", flush=True)

    metadata = {
        "experiment": 3,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "checkpoint": str(checkpoint_path),
        "probe": "h(t) of held-out test patients at times uniform in [0, 48] h, plus t/48",
        "probe_times_hours": [float(t) for t in query_times],
        "probe_count": int(probe.shape[0]),
        "feature_scores": scores,
        "input_labels": labels,
        "selected_input_labels": facts["top_labels"],
        "outside_grid_pct": outside,
        "grid_range": [grid_low, grid_high],
        "edges_before_prune": facts["edges_before"],
        "edges_after_prune": facts["edges_after"],
        "stage_status": status,
        "writeup_word_count": words,
        "symbolic_library": SYMBOLIC_LIBRARY,
        "configuration": configuration_snapshot(),
        "package_versions": package_versions(["torch", "numpy", "pykan", "sympy", "matplotlib"]),
        "device": device,
    }
    write_json(output_dir / "interpretability_metadata.json", metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from src.config import EXPERIMENT_1_DIR, EXPERIMENT_3_DIR, PART_A_CACHE_PATH

    apply_kan_grid_range(args.kan_grid_range)
    output_dir = args.output_dir or EXPERIMENT_3_DIR
    checkpoint_path = EXPERIMENT_1_DIR / "checkpoints" / "kan_neural_ode" / "best.pt"

    if args.dry_run:
        plan = validate_dry_run(PART_A_CACHE_PATH, checkpoint_path, output_dir)
        print("Experiment 3 dry run", flush=True)
        print(f"Checkpoint: {plan['checkpoint']}", flush=True)
        print(f"PyKAN version: {plan['pykan_version']}", flush=True)
        print(f"Output directory: {output_dir} (nothing written)", flush=True)
        return 0

    with tee_output_to_log(output_dir / "train.log"):
        run_experiment(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Unattended driver for the remaining experiments (plain Python, no prompts).

Runs each step as a subprocess, logs everything to ``outputs/run_all.log`` and
records progress in ``outputs/STATUS.json`` (step, start/end, exit code).
A failed step is marked FAILED and the driver continues with the steps that do
not depend on it. Finished results are never deleted or overwritten, and
GRU / MLP-ODE are never rerun in Experiment 1.

    python -u scripts/run_all.py --dry-list     # print the plan only
    python -u scripts/run_all.py                # run everything
    python -u scripts/run_all.py --only exp4    # a step name or group
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LOG_PATH = OUT / "run_all.log"
STATUS_PATH = OUT / "STATUS.json"
EXP1 = OUT / "experiment_1"
EXP4 = OUT / "experiment_4"
EXP2 = OUT / "experiment_2"
PY = sys.executable

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import DEFAULT_MAX_RUN_MINUTES, REDUCED_MAX_EPOCHS  # noqa: E402

PATIENCE = 4  # KAN, LTC, Experiment 2 retrain and Experiment 4 (GRU/MLP-ODE used 10)
CAP_MIN = DEFAULT_MAX_RUN_MINUTES  # per-run wall-clock cap used as the upper bound
# --dry-list "likely" estimates: ~(best epoch 8 + patience) epochs per run at the
# assumed seconds per epoch below (MLP-ODE measured ~17 s; the others are guesses).
LIKELY_EPOCHS = 8 + PATIENCE
ASSUMED_SEC_PER_EPOCH = {"gru": 0.3, "mlp": 17.0, "kan": 60.0, "ltc": 60.0}
OVERHEAD_MIN = 1.5  # data materialization + test pass + checkpoint restore

HEADLINE_KEYS = ("gru_baseline", "mlp_neural_ode", "kan_neural_ode", "ltc")
RUNAWAY_MARKERS = ("max_num_steps exceeded", "FIRST_EPOCH_TIMEOUT")
KAN_FIRST_EPOCH_TIMEOUT_S = 360
FALLBACK_GRID_RANGE = ["-3", "3"]
FALLBACK_WALL_HOURS = 10.0
MLP_DIMS = (8, 20, 40)

# Set when the KAN runaway guard fires; later KAN steps inherit it.
state: dict[str, list[str]] = {"grid_range_args": []}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


@dataclass
class Step:
    name: str
    group: str
    run: Callable[["Step"], tuple[str, str]]  # returns (status, note)
    describe: Callable[[], list[str]]
    status: str = "PENDING"
    start: str | None = None
    end: str | None = None
    exit_code: int | None = None
    notes: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)


def write_status(steps: list[Step]) -> None:
    payload = {
        "updated_utc": now(),
        "steps": [
            {
                "name": s.name,
                "status": s.status,
                "start": s.start,
                "end": s.end,
                "exit_code": s.exit_code,
                "commands": s.commands,
                "notes": s.notes,
            }
            for s in steps
        ],
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_name(STATUS_PATH.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS_PATH)


def headline_keys() -> set[str]:
    path = EXP1 / "headline_results.json"
    if not path.is_file():
        return set()
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return {str(r.get("model_key")) for r in rows}
    except (OSError, ValueError):
        return set()


def safe_rmtree(path: Path) -> None:
    """Delete an unfinished checkpoint directory (never anything outside outputs/)."""
    resolved = path.resolve()
    if OUT.resolve() not in resolved.parents:
        raise RuntimeError(f"Refusing to delete outside outputs/: {resolved}")
    if path.exists():
        shutil.rmtree(path)
        log(f"Deleted unfinished checkpoint directory: {path}")


def run_command(
    step: Step,
    command: list[str],
    timeout_seconds: float | None = None,
) -> tuple[int, list[str]]:
    """Run a subprocess, mirror its output to the log, return (code, marker hits)."""
    step.commands.append(" ".join(command))
    log(f"RUN [{step.name}] {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    lines: "queue.Queue[str | None]" = queue.Queue()

    def reader() -> None:
        assert process.stdout is not None
        for text in process.stdout:
            lines.put(text)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    hits: list[str] = []
    started = time.monotonic()
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        while True:
            try:
                item = lines.get(timeout=1.0)
            except queue.Empty:
                item = ""
            if item is None:
                break
            if item:
                handle.write(item)
                handle.flush()
                for marker in RUNAWAY_MARKERS:
                    if marker in item and marker not in hits:
                        hits.append(marker)
            if timeout_seconds and time.monotonic() - started > timeout_seconds:
                log(f"TIMEOUT [{step.name}] after {timeout_seconds / 3600:.1f} h; terminating.")
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
    code = process.wait()
    log(f"EXIT [{step.name}] code={code} markers={hits}")
    return code, hits


# ---------------------------------------------------------------- Experiment 1
def exp1_command(model: str, extra: list[str]) -> list[str]:
    return [
        PY, "-u", "scripts/run_experiment_1.py", "--run", "--models", model,
        "--max-epochs", "50", "--min-epochs", "5", "--patience", str(PATIENCE),
        "--min-delta", "0", "--checkpoint-every", "5", "--log-every-epoch",
        *extra,
    ]


def make_exp1_step(model: str) -> Step:
    is_kan = model == "kan_neural_ode"
    checkpoint_dir = EXP1 / "checkpoints" / model

    def guard_args() -> list[str]:
        if not is_kan:
            return []
        return ["--first-epoch-timeout", str(KAN_FIRST_EPOCH_TIMEOUT_S), *state["grid_range_args"]]

    def run(step: Step) -> tuple[str, str]:
        if model in headline_keys():
            return "OK", f"{model} already has a finished result; not rerun."
        if checkpoint_dir.exists():
            safe_rmtree(checkpoint_dir)  # unfinished: no row exists for it
        code, hits = run_command(step, exp1_command(model, guard_args()))
        step.exit_code = code
        if code == 0:
            return "OK", ""
        if is_kan and hits:
            note = f"KAN runaway guard fired ({hits}); rerunning once with grid_range {FALLBACK_GRID_RANGE}."
            log(note)
            step.notes.append(note)
            safe_rmtree(checkpoint_dir)
            state["grid_range_args"] = ["--kan-grid-range", *FALLBACK_GRID_RANGE]
            code, _ = run_command(
                step,
                exp1_command(model, state["grid_range_args"]),
                timeout_seconds=FALLBACK_WALL_HOURS * 3600,
            )
            step.exit_code = code
            return ("OK", "completed with grid_range [-3, 3]") if code == 0 else ("FAILED", "fallback run failed")
        if (checkpoint_dir / "latest.pt").is_file():
            step.notes.append("retrying once with --resume")
            code, _ = run_command(step, exp1_command(model, [*guard_args(), "--resume"]))
            step.exit_code = code
            return ("OK", "") if code == 0 else ("FAILED", "failed again after --resume")
        return "FAILED", "no latest.pt to resume from"

    def describe() -> list[str]:
        pending = model not in headline_keys()
        lines = [f"# runs only if {model} has no finished headline row (currently pending: {pending})"]
        if checkpoint_dir.exists() and pending:
            lines.append(f"# would first delete unfinished {checkpoint_dir}")
        lines.append(" ".join(exp1_command(model, guard_args())))
        if is_kan:
            lines.append(f"# on max_num_steps error or first epoch > {KAN_FIRST_EPOCH_TIMEOUT_S}s: rerun once with --kan-grid-range {' '.join(FALLBACK_GRID_RANGE)}")
        return lines

    return Step(f"exp1_{'kan' if is_kan else 'ltc'}", "exp1", run, describe)


def run_estimate(kind: str, max_epochs: int = 50) -> tuple[float, float]:
    """(upper bound, likely) minutes for one capped training run."""
    likely = min(LIKELY_EPOCHS, max_epochs) * ASSUMED_SEC_PER_EPOCH[kind] / 60.0 + OVERHEAD_MIN
    return CAP_MIN + OVERHEAD_MIN, min(likely, CAP_MIN + OVERHEAD_MIN)


def exp1_verify_step() -> Step:
    def run(step: Step) -> tuple[str, str]:
        path = EXP1 / "headline_results.json"
        rows = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        keys = [r.get("model_key") for r in rows]
        ok = sorted(keys) == sorted(HEADLINE_KEYS) and len(keys) == 4
        names = [r.get("model") for r in rows]
        return ("OK", f"four rows: {names}") if ok else ("FAILED", f"headline has {keys}, expected {list(HEADLINE_KEYS)}")

    return Step("exp1_verify", "exp1", run, lambda: ["# verify headline_results.json has exactly GRU, MLP-ODE, KAN-ODE, LTC"])


# ------------------------------------------------------------------- the rest
def simple_step(name: str, group: str, command: list[str], precondition: Callable[[], str | None] | None = None) -> Step:
    def run(step: Step) -> tuple[str, str]:
        if precondition is not None:
            reason = precondition()
            if reason:
                return "SKIPPED", reason
        code, _ = run_command(step, command)
        step.exit_code = code
        return ("OK", "") if code == 0 else ("FAILED", f"exit code {code}")

    return Step(name, group, run, lambda: [" ".join(command)])


def available_models() -> list[str]:
    keys = headline_keys()
    return [m for m in HEADLINE_KEYS if m in keys and (EXP1 / "checkpoints" / m / "best.pt").is_file()]


def exp2_zero_shot_step() -> Step:
    def command() -> list[str]:
        return [PY, "-u", "scripts/run_experiment_2.py", "--run", "--seeds", "3", "--models", *available_models()]

    def run(step: Step) -> tuple[str, str]:
        models = available_models()
        if not models:
            return "SKIPPED", "no finished Experiment 1 models"
        if len(models) < 4:
            step.notes.append(f"only evaluating {models} (others have no finished result)")
        code, _ = run_command(step, command())
        step.exit_code = code
        return ("OK", "") if code == 0 else ("FAILED", f"exit code {code}")

    return Step("exp2_zero_shot", "exp2", run, lambda: [" ".join(command()) + "   # --models = those with finished results"])


def exp3_step() -> Step:
    command = [PY, "-u", "scripts/run_experiment_3.py", "--run"]

    def run(step: Step) -> tuple[str, str]:
        if "kan_neural_ode" not in headline_keys():
            return "SKIPPED", "depends on a finished KAN-ODE (Experiment 1)"
        code, _ = run_command(step, command)
        step.exit_code = code
        return ("OK", "") if code == 0 else ("FAILED", f"exit code {code}")

    return Step("exp3", "exp3", run, lambda: [" ".join(command) + "   # needs finished KAN-ODE"])


def exp4_run_step(family: str, hidden: int) -> Step:
    name = f"{family}_h{hidden}"
    checkpoint_dir = EXP4 / "checkpoints" / name
    result_path = EXP4 / "runs" / f"{name}.json"
    is_kan = family == "kan"

    def command(extra: list[str] | None = None) -> list[str]:
        base = [PY, "-u", "scripts/run_experiment_4.py", "--run", "--family", family, "--hidden-dim", str(hidden)]
        if is_kan:
            base += ["--first-epoch-timeout", str(KAN_FIRST_EPOCH_TIMEOUT_S), *state["grid_range_args"]]
        return base + (extra or [])

    def run(step: Step) -> tuple[str, str]:
        if result_path.is_file():
            return "OK", "result already exists; skipped"
        code, hits = run_command(step, command())
        step.exit_code = code
        if code == 0:
            return "OK", ""
        if is_kan and hits:
            note = f"KAN runaway guard fired ({hits}); rerunning once with grid_range {FALLBACK_GRID_RANGE}."
            log(note)
            step.notes.append(note)
            safe_rmtree(checkpoint_dir)
            state["grid_range_args"] = ["--kan-grid-range", *FALLBACK_GRID_RANGE]
            fallback = [PY, "-u", "scripts/run_experiment_4.py", "--run", "--family", family, "--hidden-dim", str(hidden), *state["grid_range_args"]]
            code, _ = run_command(step, fallback, timeout_seconds=FALLBACK_WALL_HOURS * 3600)
            step.exit_code = code
            return ("OK", "completed with grid_range [-3, 3]") if code == 0 else ("FAILED", "fallback run failed")
        if (checkpoint_dir / "latest.pt").is_file():
            step.notes.append("retrying once with --resume")
            code, _ = run_command(step, command(["--resume"]))
            step.exit_code = code
            return ("OK", "") if code == 0 else ("FAILED", "failed again after --resume")
        return "FAILED", f"exit code {code}; no latest.pt to resume from"

    return Step(f"exp4_{name}", "exp4", run, lambda: [" ".join(command())])


def exp4_aggregate_step() -> Step:
    return simple_step(
        "exp4_aggregate", "exp4",
        [PY, "-u", "scripts/run_experiment_4.py", "--aggregate"],
        precondition=lambda: None if list((EXP4 / "runs").glob("*.json")) else "no Experiment 4 runs finished",
    )


def retrain_step(model: str) -> Step:
    command = [PY, "-u", "scripts/run_experiment_2.py", "--retrain", "--retrain-models", model]

    def run(step: Step) -> tuple[str, str]:
        code, _ = run_command(step, command)
        step.exit_code = code
        if code == 0:
            return "OK", ""
        step.notes.append("retrying once with --resume")
        code, _ = run_command(step, [*command, "--resume"])
        step.exit_code = code
        return ("OK", "") if code == 0 else ("FAILED", "failed again after --resume")

    return Step(f"exp2_retrain_{model}", "retrain", run, lambda: [" ".join(command) + "   # levels 50 and 25 (defaults); KAN and LTC skipped (see docs/DECISIONS.md)"])


def build_steps() -> list[Step]:
    steps = [
        make_exp1_step("kan_neural_ode"),
        make_exp1_step("ltc"),
        exp1_verify_step(),
        simple_step("time_ltc_vs_cfc", "time", [PY, "-u", "scripts/time_ltc_vs_cfc.py", "--run", "--epochs", "3"]),
        exp2_zero_shot_step(),
        exp3_step(),
    ]
    steps += [exp4_run_step("mlp", h) for h in MLP_DIMS]
    steps += [exp4_run_step("kan", h) for h in MLP_DIMS]
    steps.append(exp4_aggregate_step())
    steps += [retrain_step(m) for m in ("gru_baseline", "mlp_neural_ode")]
    return steps


def estimate(step: Step) -> tuple[float, float, str]:
    """(upper, likely, note) minutes for the --dry-list column."""
    name = step.name
    if name == "exp1_kan":
        return (*run_estimate("kan"), "")
    if name == "exp1_ltc":
        return (*run_estimate("ltc"), "")
    if name == "exp1_verify":
        return 0.0, 0.0, ""
    if name == "time_ltc_vs_cfc":
        return 10.0, 5.0, "3 epochs x 2 cell types"
    if name == "exp2_zero_shot":
        return 10.0, 4.0, "4 models x 10 test variants"
    if name == "exp3":
        return 15.0, 8.0, "plots dominate"
    if name.startswith("exp4_mlp") or name.startswith("exp4_kan"):
        kind = "mlp" if name.startswith("exp4_mlp") else "kan"
        upper, likely = run_estimate(kind, REDUCED_MAX_EPOCHS)
        note = "~0 if the Exp 1 result is reused" if name.endswith("_h20") else ""
        return upper, likely, note
    if name == "exp4_aggregate":
        return 1.0, 0.5, ""
    if name.startswith("exp2_retrain_"):
        kind = "gru" if "gru" in name else "mlp"
        upper, likely = run_estimate(kind, REDUCED_MAX_EPOCHS)
        return 2 * upper, 2 * likely, "levels 50 and 25"
    return 0.0, 0.0, ""


def select(steps: list[Step], only: str | None) -> list[Step]:
    if not only:
        return steps
    wanted = {token.strip() for token in only.split(",") if token.strip()}
    chosen = [s for s in steps if s.name in wanted or s.group in wanted]
    if not chosen:
        names = sorted({s.name for s in steps} | {s.group for s in steps})
        raise SystemExit(f"--only {only!r} matched nothing. Valid: {', '.join(names)}")
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default=None, help="Step name or group (exp1, time, exp2, exp3, exp4, retrain); comma-separated allowed.")
    parser.add_argument("--dry-list", action="store_true", help="Print planned commands; run nothing.")
    args = parser.parse_args(argv)

    steps = select(build_steps(), args.only)
    if args.dry_list:
        total_upper = total_likely = 0.0
        print(
            f"Estimates: upper = per-run wall-clock cap ({CAP_MIN:g} min) + {OVERHEAD_MIN} min "
            f"overhead; likely = ~{LIKELY_EPOCHS} epochs (patience {PATIENCE}) at assumed s/epoch "
            f"{ASSUMED_SEC_PER_EPOCH}. Sequential, so times add."
        )
        print(f"{'step':<22}{'upper min':>10}{'likely min':>12}  note")
        for step in steps:
            upper, likely, note = estimate(step)
            total_upper += upper
            total_likely += likely
            print(f"{step.name:<22}{upper:>10.1f}{likely:>12.1f}  {note}")
        print(f"{'TOTAL':<22}{total_upper:>10.1f}{total_likely:>12.1f}  (~{total_upper / 60:.1f} h upper, ~{total_likely / 60:.1f} h likely)")
        print()
        for step in steps:
            print(f"== {step.name} ({step.group})")
            for line in step.describe():
                print(f"   {line}")
        return 0

    log(f"run_all start; steps: {[s.name for s in steps]}")
    write_status(steps)
    for step in steps:
        step.status, step.start = "RUNNING", now()
        write_status(steps)
        try:
            step.status, note = step.run(step)
            if note:
                step.notes.append(note)
        except Exception:  # noqa: BLE001
            step.status = "FAILED"
            trace = traceback.format_exc()
            step.notes.append("driver exception; see run_all.log")
            log(f"EXCEPTION [{step.name}]\n{trace}")
        step.end = now()
        log(f"STEP {step.name}: {step.status} {step.notes[-1] if step.notes else ''}")
        write_status(steps)

    failed = [s.name for s in steps if s.status == "FAILED"]
    log(f"run_all finished; failed steps: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

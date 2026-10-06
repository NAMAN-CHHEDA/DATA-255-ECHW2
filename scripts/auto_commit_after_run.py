"""Wait for scripts/run_all.py to finish, then commit a safe subset of the results.

Plain Python, subprocess-based, no prompts. It never touches main: it creates
and pushes a new branch ``results/run-<YYYYMMDD-HHMM>`` (no force, no retry loop;
if the push fails the local commit is kept). Everything is logged to
``outputs/auto_commit.log`` (never staged).

Stages only:
  * changed/untracked files under src/, scripts/, docs/ plus README.md,
    requirements.txt, .gitignore;
  * (with ``git add -f``) small result files under outputs/ (see RESULT_RULES).
Never stages *.pt, *.pkl, data/, checkpoints, credentials, or anything over 5 MB.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LOG_PATH = OUT / "auto_commit.log"
STATUS_PATH = OUT / "STATUS.json"
MAX_BYTES = 5 * 1024 * 1024
POLL_SECONDS = 30

CODE_DIRS = ("src/", "scripts/", "docs/")
CODE_FILES = ("README.md", "requirements.txt", ".gitignore")
BLOCKED_SUFFIXES = (".pt", ".pth", ".pkl", ".tar.gz", ".pem", ".key")
BLOCKED_PARTS = {"data", "checkpoints", "pykan_ckpt", "__pycache__", ".venv", "venv"}
BLOCKED_NAMES = ("credentials", "secret", "id_rsa", "token")

# (directory, glob patterns, recursive)
RESULT_RULES = [
    ("outputs/experiment_1", ["headline_results.*", "run_metadata.json", "ltc_vs_cfc_timing.*", "train.log"], False),
    ("outputs/experiment_2", ["*.csv", "*.json", "*.png"], True),
    ("outputs/experiment_3", ["*.png", "*.md", "*.json", "errors.log", "*.csv", "symbolic_formulas.txt"], True),
    ("outputs/experiment_4", ["*.csv", "*.json", "*.png", "train.log"], True),
]
RESULT_FILES = ["outputs/STATUS.json", "outputs/run_all.log"]


def log(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(command: list[str], input_text: str | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    log("$ " + " ".join(command))
    result = subprocess.run(
        command, cwd=ROOT, input=input_text, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    for stream in (result.stdout, result.stderr):
        for line in stream.strip().splitlines():
            log("  " + line)
    return result


def find_run_all_pids() -> list[int]:
    script = (
        "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_all\\.py' "
        "-and $_.CommandLine -notmatch 'auto_commit_after_run' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True, text=True, timeout=60,
    )
    return [int(token) for token in result.stdout.split() if token.isdigit()]


def pid_alive(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=60
    )
    return str(pid) in result.stdout


def status_complete() -> bool:
    try:
        steps = json.loads(STATUS_PATH.read_text(encoding="utf-8"))["steps"]
    except (OSError, ValueError, KeyError):
        return False
    return bool(steps) and all(step.get("end") for step in steps)


def wait_for_run(pids: list[int], max_wait_hours: float) -> None:
    deadline = time.monotonic() + max_wait_hours * 3600
    if pids:
        log(f"Waiting for run_all.py PID(s) {pids} to exit (poll {POLL_SECONDS}s).")
        while any(pid_alive(pid) for pid in pids):
            if time.monotonic() > deadline:
                log("Max wait reached; proceeding anyway.")
                return
            time.sleep(POLL_SECONDS)
        log("run_all.py process exited.")
        return
    log("No run_all.py process found; waiting until STATUS.json shows every step with an end time.")
    while not status_complete():
        if time.monotonic() > deadline:
            log("Max wait reached; proceeding anyway.")
            return
        time.sleep(POLL_SECONDS)
    log("STATUS.json shows all steps finished.")


def blocked(path: Path) -> str | None:
    """Reason a file must never be staged, else None."""
    relative = path.relative_to(ROOT) if path.is_absolute() else path
    name = relative.name.lower()
    if name.endswith(BLOCKED_SUFFIXES):
        return "blocked suffix"
    if name.startswith(".env") or any(token in name for token in BLOCKED_NAMES):
        return "credential-like name"
    if BLOCKED_PARTS & set(relative.parts):
        return "blocked directory"
    full = ROOT / relative
    if full.is_file() and full.stat().st_size > MAX_BYTES:
        return f"over 5 MB ({full.stat().st_size / 1e6:.1f} MB)"
    return None


def changed_code_paths() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "-uall"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    paths: list[str] = []
    entries = result.stdout.split("\0")
    skip_next = False
    for entry in entries:
        if skip_next:  # rename: second path
            skip_next = False
            continue
        if len(entry) < 4:
            continue
        code, path = entry[:2], entry[3:]
        if "R" in code or "C" in code:
            skip_next = True
        if path.startswith(CODE_DIRS) or path in CODE_FILES:
            paths.append(path)
    return paths


def result_paths() -> list[str]:
    found: list[str] = []
    for directory, patterns, recursive in RESULT_RULES:
        base = ROOT / directory
        if not base.is_dir():
            continue
        for pattern in patterns:
            matches = base.rglob(pattern) if recursive else base.glob(pattern)
            found += [p.relative_to(ROOT).as_posix() for p in matches if p.is_file()]
    found += [p for p in RESULT_FILES if (ROOT / p).is_file()]
    return sorted(set(found))


def summarize_status() -> tuple[int, int, str]:
    try:
        steps = json.loads(STATUS_PATH.read_text(encoding="utf-8"))["steps"]
    except (OSError, ValueError, KeyError):
        return 0, 0, "(STATUS.json unavailable)"
    ok = sum(1 for s in steps if s.get("status") == "OK")
    failed = sum(1 for s in steps if s.get("status") == "FAILED")
    body = "\n".join(
        f"- {s.get('name')}: {s.get('status')}"
        + (f" ({s['notes'][-1]})" if s.get("notes") else "")
        for s in steps
    )
    return ok, failed, body


def commit_and_push(no_push: bool) -> None:
    run(["git", "remote", "-v"])
    original_branch = run(["git", "branch", "--show-current"]).stdout.strip()
    run(["git", "status"])

    code_paths = changed_code_paths()
    skipped: list[str] = []
    to_add_plain: list[str] = []
    for path in code_paths:
        reason = blocked(Path(path))
        if reason:
            skipped.append(f"{path} ({reason})")
        else:
            to_add_plain.append(path)
    to_add_forced: list[str] = []
    for path in result_paths():
        reason = blocked(Path(path))
        if reason:
            skipped.append(f"{path} ({reason})")
        else:
            to_add_forced.append(path)
    # Large result files that the rules above would have matched are reported too.
    for directory, patterns, recursive in RESULT_RULES:
        base = ROOT / directory
        if base.is_dir():
            for path in base.rglob("*") if recursive else base.glob("*"):
                if path.is_file() and path.stat().st_size > MAX_BYTES:
                    note = f"{path.relative_to(ROOT).as_posix()} (over 5 MB)"
                    if note not in skipped:
                        skipped.append(note)

    log(f"Skipped files ({len(skipped)}):")
    for item in skipped:
        log(f"  SKIPPED {item}")
    log(f"Staging {len(to_add_plain)} code/doc paths and {len(to_add_forced)} result files.")

    if not to_add_plain and not to_add_forced:
        log("Nothing to stage; no commit made.")
        return

    branch = "results/run-" + datetime.now().strftime("%Y%m%d-%H%M")
    result = run(["git", "switch", "-c", branch])
    if result.returncode != 0:
        log("Could not create the results branch; aborting without committing.")
        return
    log(f"Now on new branch {branch} (was {original_branch}); the working tree is unchanged.")

    if to_add_plain:
        run(["git", "add", "--pathspec-from-file=-", "--pathspec-file-nul"], input_text="\0".join(to_add_plain) + "\0")
    if to_add_forced:
        run(["git", "add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul"], input_text="\0".join(to_add_forced) + "\0")

    staged_files = run(["git", "diff", "--cached", "--name-only"]).stdout.split()
    bad = [f for f in staged_files if blocked(Path(f)) and f not in to_add_plain and f not in to_add_forced]
    if bad:
        log(f"Pre-staged files outside the allowed set will not be committed: {bad}")

    ok, failed, body = summarize_status()
    message = (
        f"ECHW2 run results: {ok} steps OK, {failed} FAILED\n\n{body}\n\n"
        "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>\n"
    )
    allowed = sorted(set(to_add_plain) | set(to_add_forced))
    # --only commits just these paths even if other things were staged earlier.
    commit = run(["git", "commit", "-F", "-", "--only", "--", *allowed], input_text=message)
    if commit.returncode != 0:
        log("Commit failed or had nothing to commit; see output above.")
        return
    run(["git", "log", "-1", "--stat", "--format=%H %s"])

    if no_push:
        log("--no-push given; local commit only.")
        return
    remotes = run(["git", "remote"]).stdout.split()
    if "origin" not in remotes:
        log("No 'origin' remote; keeping the local commit only.")
        return
    push = run(["git", "push", "-u", "origin", branch], timeout=180)
    if push.returncode != 0:
        log(f"Push failed (kept local commit on {branch}); not retrying.")
    else:
        log(f"Pushed {branch} to origin.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pid", type=int, action="append", default=None, help="PID of the run_all.py process to wait for (repeatable).")
    parser.add_argument("--max-wait-hours", type=float, default=72.0)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    log("auto_commit_after_run started")
    log(f"branch at start: {subprocess.run(['git', 'branch', '--show-current'], cwd=ROOT, capture_output=True, text=True).stdout.strip()}")
    pids = args.pid if args.pid else find_run_all_pids()
    wait_for_run(pids, args.max_wait_hours)
    try:
        commit_and_push(args.no_push)
    except Exception as error:  # noqa: BLE001
        import traceback

        log("auto_commit failed: " + repr(error) + "\n" + traceback.format_exc())
        return 1
    log("auto_commit_after_run finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

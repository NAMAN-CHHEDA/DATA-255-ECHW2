"""Read-only structure and final-artifact validator.

This script only reads paths and file contents. It never creates, modifies,
renames, or deletes files.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CheckResult:
    """One validation result and whether it is required in final mode."""

    name: str
    status: str
    detail: str
    required: bool = True


def build_parser() -> argparse.ArgumentParser:
    """Build mutually exclusive read-only validation modes."""
    parser = argparse.ArgumentParser(
        description="Read-only DATA 255 Homework 2 submission validator."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Check source/template structure only. Real experimental outputs "
            "are not required."
        ),
    )
    mode.add_argument(
        "--final",
        action="store_true",
        help="Require completed GPU-run artifacts and final written documents.",
    )
    return parser


def _result(
    name: str,
    passed: bool,
    detail: str,
    required: bool = True,
) -> CheckResult:
    """Construct a required or optional result."""
    if passed:
        status = "PASS"
    elif required:
        status = "MISSING"
    else:
        status = "OPTIONAL"
    return CheckResult(name, status, detail, required)


def _nonempty(path: Path) -> bool:
    """Return whether ``path`` is a nonempty regular file."""
    return path.is_file() and path.stat().st_size > 0


def _valid_json(path: Path) -> bool:
    """Return whether a nonempty path contains valid JSON."""
    if not _nonempty(path):
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return True


def _csv_has_data(path: Path) -> bool:
    """Return whether a CSV has a header and at least one data row."""
    if not _nonempty(path):
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return False
    return len(rows) >= 2 and bool(rows[0])


def check_python_syntax() -> CheckResult:
    """Parse every project source/script file without importing or executing it."""
    python_files = sorted(
        [
            *PROJECT_ROOT.joinpath("src").rglob("*.py"),
            *PROJECT_ROOT.joinpath("scripts").rglob("*.py"),
        ]
    )
    if not python_files:
        return _result("Runnable Python code", False, "No Python files found.")
    failures: list[str] = []
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            failures.append(f"{path.relative_to(PROJECT_ROOT)}: {exc}")
    return _result(
        "Runnable Python code",
        not failures,
        (
            f"Parsed {len(python_files)} Python files."
            if not failures
            else "; ".join(failures)
        ),
    )


def check_readme() -> CheckResult:
    """Check for README and the documented validation commands."""
    path = PROJECT_ROOT / "README.md"
    if not _nonempty(path):
        return _result("README", False, "README.md is absent or empty.")
    text = path.read_text(encoding="utf-8")
    commands = [
        "python scripts/validate_submission.py --dry-run",
        "python scripts/validate_submission.py --final",
    ]
    missing = [command for command in commands if command not in text]
    return _result(
        "README",
        not missing,
        "Required validator commands are documented."
        if not missing
        else "Missing commands: " + ", ".join(missing),
    )


def check_part_g_template() -> CheckResult:
    """Check that all six required reflection prompts are present."""
    path = PROJECT_ROOT / "docs" / "part_g_reflection_template.md"
    if not _nonempty(path):
        return _result(
            "Part G reflection template",
            False,
            "docs/part_g_reflection_template.md is absent or empty.",
        )
    text = path.read_text(encoding="utf-8")
    required_fragments = [
        "## 1. Why is feeding Δt",
        "## 2. Why can deeper or wider KANs",
        "## 3. How does LTC/CfC forgetfulness",
        "## 4. Based on Experiment 4",
        "## 5. Which model should be deployed",
        "## 6. Did the biologically or mathematically principled motivation",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in text]
    reminders = text.count("**Target:** 100–250 words.")
    placeholders = text.count("**My result-based argument**")
    passed = not missing and reminders == 6 and placeholders == 6
    return _result(
        "Part G reflection template",
        passed,
        (
            "Six prompts, word targets, and argument placeholders are present."
            if passed
            else (
                f"Missing heading fragments={missing}; target reminders={reminders}; "
                f"argument placeholders={placeholders}."
            )
        ),
    )


def check_executive_template() -> CheckResult:
    """Check one-page executive-summary template structure."""
    path = PROJECT_ROOT / "docs" / "executive_summary_template.md"
    if not _nonempty(path):
        return _result(
            "Executive summary template",
            False,
            "docs/executive_summary_template.md is absent or empty.",
        )
    text = path.read_text(encoding="utf-8")
    headings = [
        "## Objective",
        "## Dataset and Protocol",
        "## Headline Results",
        "## Robustness Finding",
        "## Parameter-Efficiency Finding",
        "## Interpretability Finding",
        "## Deployment Recommendation",
        "## Limitations",
    ]
    missing = [heading for heading in headings if heading not in text]
    passed = not missing and "One Page Maximum" in text
    return _result(
        "Executive summary template",
        passed,
        "One-page marker and all required sections are present."
        if passed
        else "Missing sections/marker: " + ", ".join(missing),
    )


def check_regeneration_pipeline() -> CheckResult:
    """Accept either the Part A cache or its complete regeneration pipeline."""
    cache = PROJECT_ROOT / "data" / "processed" / "part_a_cache.pkl"
    pipeline = [
        PROJECT_ROOT / "scripts" / "download_data.py",
        PROJECT_ROOT / "src" / "data_preprocessing.py",
    ]
    cache_ready = _nonempty(cache)
    pipeline_ready = all(_nonempty(path) for path in pipeline)
    return _result(
        "Part A cache or regeneration pipeline",
        cache_ready or pipeline_ready,
        (
            f"Cache present: {cache.relative_to(PROJECT_ROOT)}."
            if cache_ready
            else (
                "Cache absent; download and preprocessing pipeline is present."
                if pipeline_ready
                else "Neither a cache nor the complete regeneration pipeline exists."
            )
        ),
    )


def _contains_placeholder(text: str) -> bool:
    """Detect template instructions that must not remain in completed writing."""
    markers = [
        "[WRITE",
        "[INSERT",
        "[STATE",
        "[DESCRIBE",
        "[CITE",
        "[EXPLAIN",
        "[COMPARE",
        "[REFERENCE",
        "[RECOMMEND",
        "[LIST",
        "PLACEHOLDER",
        "TODO",
    ]
    upper = text.upper()
    return any(marker in upper for marker in markers)


def check_completed_reflection() -> CheckResult:
    """Require six completed 100–250 word responses, not the template."""
    path = PROJECT_ROOT / "docs" / "part_g_reflection.md"
    if not _nonempty(path):
        return _result(
            "Completed Part G reflection",
            False,
            "Create docs/part_g_reflection.md after completing the GPU runs.",
        )
    text = path.read_text(encoding="utf-8")
    if _contains_placeholder(text):
        return _result(
            "Completed Part G reflection",
            False,
            "The final reflection still contains template placeholders.",
        )
    matches = list(re.finditer(r"(?m)^##\s+([1-6])\.\s+.+$", text))
    if len(matches) != 6:
        return _result(
            "Completed Part G reflection",
            False,
            f"Expected six numbered response headings; found {len(matches)}.",
        )
    invalid_counts: list[str] = []
    for index, match in enumerate(matches):
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():section_end]
        word_count = len(re.findall(r"\b[\wΔ-]+\b", body))
        if not 100 <= word_count <= 250:
            invalid_counts.append(f"Q{match.group(1)}={word_count} words")
    return _result(
        "Completed Part G reflection",
        not invalid_counts,
        "Six completed responses are each within 100–250 words."
        if not invalid_counts
        else "Out-of-range responses: " + ", ".join(invalid_counts),
    )


def check_completed_executive_summary() -> CheckResult:
    """Require a completed non-template executive summary of plausible page length."""
    path = PROJECT_ROOT / "docs" / "executive_summary.md"
    if not _nonempty(path):
        return _result(
            "Completed one-page executive summary",
            False,
            "Create docs/executive_summary.md after completing the GPU runs.",
        )
    text = path.read_text(encoding="utf-8")
    if _contains_placeholder(text):
        return _result(
            "Completed one-page executive summary",
            False,
            "The final executive summary still contains template placeholders.",
        )
    word_count = len(re.findall(r"\b[\wΔ-]+\b", text))
    passed = 100 <= word_count <= 700
    return _result(
        "Completed one-page executive summary",
        passed,
        (
            f"Completed summary has {word_count} words; confirm page length manually."
            if passed
            else f"Summary has {word_count} words; expected approximately 100–700."
        ),
    )


def check_experiment_3_artifacts() -> CheckResult:
    """Check required KAN figures, metadata, and generated reflection template."""
    root = PROJECT_ROOT / "outputs" / "experiment_3"
    single_figure = root / "kan_edge_functions.png"
    figure_directory = root / "kan_edge_functions"
    has_figure = _nonempty(single_figure) or (
        figure_directory.is_dir()
        and any(_nonempty(path) for path in figure_directory.rglob("*.png"))
    )
    metadata = root / "interpretability_metadata.json"
    reflection = root / "reflection_e3_template.md"
    passed = has_figure and _valid_json(metadata) and _nonempty(reflection)
    return _result(
        "Experiment 3 KAN figures, metadata, and reflection template",
        passed,
        (
            "KAN figure artifact, valid metadata, and reflection template found."
            if passed
            else "One or more Experiment 3 required artifacts are absent."
        ),
    )


def dry_run_checks() -> list[CheckResult]:
    """Return source/template checks that do not require GPU outputs."""
    return [
        check_python_syntax(),
        check_readme(),
        check_part_g_template(),
        check_executive_template(),
        check_regeneration_pipeline(),
    ]


def final_checks() -> list[CheckResult]:
    """Return all required and explicitly optional final-submission checks."""
    checks = dry_run_checks()
    checks.extend(
        [
            _result(
                "Experiment 1 headline table",
                _csv_has_data(
                    PROJECT_ROOT
                    / "outputs"
                    / "experiment_1"
                    / "headline_results.csv"
                ),
                "outputs/experiment_1/headline_results.csv with data rows.",
            ),
            _result(
                "Experiment 2 robustness CSV and AUROC figure",
                _csv_has_data(
                    PROJECT_ROOT
                    / "outputs"
                    / "experiment_2"
                    / "robustness_results.csv"
                )
                and _nonempty(
                    PROJECT_ROOT
                    / "outputs"
                    / "experiment_2"
                    / "robustness_auroc.png"
                ),
                "Required Experiment 2 CSV and PNG must both be nonempty.",
            ),
            check_experiment_3_artifacts(),
            _result(
                "Experiment 3 attribution and symbolic supplements",
                _csv_has_data(
                    PROJECT_ROOT
                    / "outputs"
                    / "experiment_3"
                    / "top_two_input_dimensions.csv"
                )
                and _nonempty(
                    PROJECT_ROOT
                    / "outputs"
                    / "experiment_3"
                    / "symbolic_formulas.txt"
                ),
                "Optional final check for attribution CSV and symbolic formulas.",
                required=False,
            ),
            _result(
                "Experiment 4 parameter-efficiency table and figure",
                _csv_has_data(
                    PROJECT_ROOT
                    / "outputs"
                    / "experiment_4"
                    / "parameter_efficiency_results.csv"
                )
                and _nonempty(
                    PROJECT_ROOT
                    / "outputs"
                    / "experiment_4"
                    / "parameter_efficiency_auroc_vs_params.png"
                ),
                "Required Experiment 4 CSV and PNG must both be nonempty.",
            ),
            check_completed_reflection(),
            check_completed_executive_summary(),
            _result(
                "Optional CfC comparison output",
                _csv_has_data(
                    PROJECT_ROOT
                    / "outputs"
                    / "experiment_1"
                    / "cfc_variant_results.csv"
                ),
                "CfC is separate from the required four-model headline table.",
                required=False,
            ),
        ]
    )
    return checks


def print_results(results: list[CheckResult]) -> None:
    """Print one PASS, MISSING, or OPTIONAL line per item."""
    for result in results:
        print(f"{result.status}: {result.name} — {result.detail}")


def main(argv: list[str] | None = None) -> int:
    """Run read-only validation and fail only for final required omissions."""
    args = build_parser().parse_args(argv)
    if args.dry_run:
        results = dry_run_checks()
        print("Dry-run validation: real experimental outputs are not required.")
        print_results(results)
        return 0

    results = final_checks()
    print_results(results)
    missing_required = [
        result
        for result in results
        if result.required and result.status == "MISSING"
    ]
    if missing_required:
        print(
            "Submission is not ready: "
            f"{len(missing_required)} required item(s) are missing."
        )
        return 1
    print("Submission validation passed: all required final artifacts are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

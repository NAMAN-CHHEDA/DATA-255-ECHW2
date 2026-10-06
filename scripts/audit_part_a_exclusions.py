"""Audit raw Set A records omitted from the existing Part A cache.

This diagnostic is intentionally read-only: it loads the existing outcomes and
cache, then applies the real Part A patient parser to raw records absent from
the cache. It does not invoke the preprocessing entry point or write artifacts.
"""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep this read-only diagnostic from creating __pycache__ files.
sys.dont_write_bytecode = True

import src.data_preprocessing as preprocessing  # noqa: E402
from src.config import (  # noqa: E402
    MAX_HOURS,
    OUTCOMES_PATH,
    PART_A_CACHE_PATH,
    SET_A_DIR,
)


def _load_cache() -> dict[str, Any]:
    """Load the existing Part A cache without changing it."""
    if not PART_A_CACHE_PATH.is_file():
        raise FileNotFoundError(f"Part A cache not found: {PART_A_CACHE_PATH}")
    with PART_A_CACHE_PATH.open("rb") as handle:
        cache = pickle.load(handle)
    if not isinstance(cache, dict) or not isinstance(cache.get("records"), dict):
        raise ValueError("Part A cache does not contain the expected records mapping.")
    return cache


def _cached_record_ids(cache: dict[str, Any]) -> set[str]:
    """Extract and validate RecordIDs from every cached split."""
    record_ids: list[str] = []
    for split_records in cache["records"].values():
        if not isinstance(split_records, list):
            raise ValueError("Each cache records split must be a list.")
        for patient in split_records:
            try:
                record_ids.append(
                    preprocessing.canonical_record_id(patient["record_id"])
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Cache contains a record without a valid RecordID.") from exc

    unique_ids = set(record_ids)
    if len(unique_ids) != len(record_ids):
        raise ValueError("Part A cache contains duplicate RecordIDs.")
    return unique_ids


def _raw_patient_paths() -> dict[str, Path]:
    """Map canonical filename RecordIDs to all raw Set A patient files."""
    if not SET_A_DIR.is_dir():
        raise FileNotFoundError(f"Set A directory not found: {SET_A_DIR}")

    paths_by_id: dict[str, Path] = {}
    for patient_path in sorted(SET_A_DIR.glob("*.txt")):
        record_id = preprocessing.canonical_record_id(patient_path.stem)
        if record_id in paths_by_id:
            raise ValueError(f"Duplicate raw patient RecordID {record_id}.")
        paths_by_id[record_id] = patient_path
    return paths_by_id


def _reason_when_no_patient(patient_path: Path) -> str:
    """Distinguish the two ways the real parser can return ``None``."""
    has_selected_variable = False
    with patient_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["Parameter"].strip() not in preprocessing.SELECTED_VARIABLES:
                continue
            has_selected_variable = True
            time_hours = preprocessing.parse_time_hours(row["Time"])
            if 0.0 <= time_hours <= MAX_HOURS:
                raise RuntimeError(
                    f"{patient_path} has an in-window selected observation, but "
                    "the preprocessing parser returned no patient."
                )

    if not has_selected_variable:
        return "no selected variable"
    return "no selected observation within 0-48 hours"


def _is_validation_error(message: str) -> bool:
    """Recognize errors raised by the existing ``validate_patient`` contract."""
    validation_fragments = (
        "has no valid binary label",
        "must have a non-empty 1D times array",
        "has a time outside [0, 48]",
        "times are not strictly ascending",
        " shape is ",
        "mask contains a value other than 0 or 1",
        "contains a non-finite tensor value",
        "contains a negative delta_t",
        "has nonzero values where mask is 0",
    )
    return any(fragment in message for fragment in validation_fragments)


def _exclusion_reason(patient_path: Path, outcomes: dict[str, int]) -> str:
    """Apply the real patient parser and classify why the record was excluded."""
    try:
        patient = preprocessing.read_patient_file(patient_path, outcomes)
    except (csv.Error, UnicodeError) as exc:
        return f"parsing issue: {exc}"
    except ValueError as exc:
        message = str(exc)
        if "has no mortality label" in message:
            return "missing outcome label"
        if _is_validation_error(message):
            return f"validation failure: {message}"
        return f"parsing issue: {message}"

    if patient is None:
        return _reason_when_no_patient(patient_path)
    return "cache mismatch: record passes current preprocessing"


def main() -> int:
    """Print the raw/cache comparison and the reason for every exclusion."""
    raw_paths = _raw_patient_paths()
    cache = _load_cache()
    cached_ids = _cached_record_ids(cache)
    outcomes = preprocessing.load_outcomes(OUTCOMES_PATH)

    raw_ids = set(raw_paths)
    cached_without_raw = sorted(cached_ids - raw_ids)
    if cached_without_raw:
        raise ValueError(
            "Cache contains RecordIDs absent from raw Set A: "
            + ", ".join(cached_without_raw)
        )

    excluded_ids = sorted(raw_ids - cached_ids, key=lambda value: (int(value), value))
    reasons = [
        (record_id, _exclusion_reason(raw_paths[record_id], outcomes))
        for record_id in excluded_ids
    ]

    print(f"Total raw records: {len(raw_ids)}")
    print(f"Total cached usable records: {len(cached_ids)}")
    print(f"Total excluded records: {len(excluded_ids)}")
    print()
    print("RecordID | exclusion reason")
    print("---------|-----------------")
    for record_id, reason in reasons:
        print(f"{record_id} | {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

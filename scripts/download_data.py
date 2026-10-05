"""Download and extract the public PhysioNet Challenge 2012 Set A files."""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download PhysioNet Challenge 2012 Set A and Outcomes-a.txt, "
            "then safely extract patient files into data/raw/set-a/."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download files again and replace extracted Set A text files.",
    )
    return parser


def download_file(url: str, destination: Path, force: bool = False) -> None:
    """Download ``url`` atomically unless ``destination`` already exists."""
    if destination.is_file() and not force:
        print(f"Already present: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DATA255-ECHW2/1.0"},
    )

    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Saved: {destination}")


def _member_relative_path(member_name: str) -> Path:
    """Return a safe path below ``set-a`` for one archive member."""
    normalized = member_name.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if parts and parts[0].lower() == "set-a":
        parts = parts[1:]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe archive member path: {member_name!r}")
    relative = Path(*parts)
    if relative.is_absolute():
        raise ValueError(f"Unsafe absolute archive member path: {member_name!r}")
    return relative


def extract_set_a(archive_path: Path, set_a_dir: Path, force: bool = False) -> int:
    """Safely extract regular ``.txt`` patient files into ``set_a_dir``."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Set A archive not found: {archive_path}")

    set_a_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            relative = _member_relative_path(member.name)
            if relative.suffix.lower() != ".txt":
                continue

            destination = set_a_dir / relative
            destination_resolved = destination.resolve()
            set_a_resolved = set_a_dir.resolve()
            if set_a_resolved not in destination_resolved.parents:
                raise ValueError(f"Archive member escapes Set A directory: {member.name!r}")
            if destination.exists() and not force:
                continue

            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Could not read archive member: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f"{destination.name}.part")
            with source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            temporary.replace(destination)
            extracted += 1
    return extracted


def main(argv: list[str] | None = None) -> int:
    """Download Outcomes A and download/extract the Set A archive."""
    args = build_parser().parse_args(argv)
    from src.config import (
        OUTCOMES_A_URL,
        OUTCOMES_PATH,
        SET_A_ARCHIVE_PATH,
        SET_A_DIR,
        SET_A_URL,
    )

    download_file(SET_A_URL, SET_A_ARCHIVE_PATH, force=args.force)
    download_file(OUTCOMES_A_URL, OUTCOMES_PATH, force=args.force)
    extracted = extract_set_a(SET_A_ARCHIVE_PATH, SET_A_DIR, force=args.force)

    patient_files = sorted(SET_A_DIR.glob("*.txt"))
    if not patient_files:
        raise RuntimeError(f"No patient text files were extracted into {SET_A_DIR}")

    print(f"Extracted or replaced {extracted} patient files.")
    print(f"Set A patient files available: {len(patient_files)}")
    print(f"Set A directory: {SET_A_DIR}")
    print(f"Outcomes file: {OUTCOMES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

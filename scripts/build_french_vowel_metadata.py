"""CLI entry point for building the French vowel metadata CSV.

Usage (from the project root)::

    uv run python scripts/build_french_vowel_metadata.py --config config/config.yaml

The script discovers all .TextGrid / .textgrid files under the input_folder
specified in config.yaml, extracts vowel-tier intervals, applies label
corrections, and writes:

    outputs/french_vowels_metadata.csv   — one row per vowel interval
    outputs/skipped_labels.csv           — files / labels that could not be processed
"""
import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the package importable when run directly (e.g. without `uv sync`).
# When the package is installed via `uv sync`, this insert is a no-op because
# Python will find `french_ids` in site-packages first.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from french_ids.build_metadata_csv import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build French vowel metadata CSV from TextGrid files. "
            "Acoustic feature columns are included but left empty (NaN) "
            "until feature extraction is enabled in config.yaml."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        metavar="PATH",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    run(args.config)


if __name__ == "__main__":
    main()

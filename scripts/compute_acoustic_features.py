"""CLI entry point for computing acoustic features from metadata CSV.

Usage (from the project root)::

    uv run python scripts/compute_acoustic_features.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from french_ids.config import load_config  # noqa: E402
from french_ids.praat_features import AcousticFeatureExtractor  # noqa: E402


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute acoustic features for the existing French vowel metadata CSV. "
            "Formant ceilings are estimated first per speakerid × vowel group, "
            "then merged back before final formants and pitch are extracted."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        metavar="PATH",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--recompute-ceilings",
        action="store_true",
        help="Ignore any cached formant_ceilings.csv and recompute ceilings",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override paths.metadata_csv from the config",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override paths.output_csv from the config",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    log_level = logging.INFO if bool(config.get("logging", {}).get("verbose", True)) else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s | %(name)s | %(message)s")

    logger.info("Loading acoustic feature pipeline config from %s", args.config)
    extractor = AcousticFeatureExtractor(
        config,
        config_path=args.config,
        recompute_ceilings=args.recompute_ceilings,
        metadata_csv=args.metadata_csv,
        output_csv=args.output_csv,
    )
    logger.info("Starting acoustic feature extraction")
    extractor.compute_all_features()
    extractor.save_outputs()
    logger.info("Acoustic feature extraction finished")


if __name__ == "__main__":
    main()
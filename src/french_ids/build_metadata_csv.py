"""Main pipeline: TextGrid files -> french_vowels_metadata.csv.

Usage (from the project root)::

    uv run python scripts/build_french_vowel_metadata.py --config config/config.yaml
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import load_config
from .filename_parser import parse_filename
from .label_cleaning import apply_label_corrections, parse_label
from .praat_features import ACOUSTIC_COLUMNS
from .textgrid_reader import find_textgrid_files, find_vowel_tier, read_textgrid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
METADATA_COLUMNS: list[str] = [
    "speakerid",
    "session",
    "activity",
    "time",
    "word",
    "vowel",
    "register",
    "start_sec",
    "duration_sec",
    "duration_ms",
]

ALL_COLUMNS: list[str] = METADATA_COLUMNS + ACOUSTIC_COLUMNS


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def build_csv(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Extract vowel metadata from all TextGrid files described in *config*.

    Args:
        config: Parsed config dict.  ``config["paths"]["input_folder"]`` must
                already be an absolute, existing path when this function is
                called (resolved by :func:`run` beforehand).

    Returns:
        A three-tuple ``(metadata_df, skipped_df, stats)`` where *stats* is a
        dict of summary counters.
    """
    input_folder = Path(config["paths"]["input_folder"])
    min_duration_ms: float = config["filters"]["min_duration_ms"]
    remove_registers: set[str] = set(config["filters"]["remove_registers"])

    # ── Discover files ──────────────────────────────────────────────────────
    tg_files = find_textgrid_files(input_folder)
    n_files_found = len(tg_files)
    logger.info("Found %d TextGrid file(s) in %s", n_files_found, input_folder)

    rows: list[dict] = []
    skipped: list[dict] = []

    n_processed = 0
    n_skipped_files = 0
    n_intervals_read = 0
    n_removed_short = 0
    n_skipped_parse = 0

    # ── Process each file ───────────────────────────────────────────────────
    for idx, tg_path in enumerate(tg_files):
        logger.info("[%d/%d] %s", idx + 1, n_files_found, tg_path.name)

        # Parse filename metadata
        file_meta = parse_filename(tg_path.stem)
        if file_meta is None:
            logger.warning("Cannot parse filename '%s' — skipping", tg_path.stem)
            skipped.append({
                "file": str(tg_path),
                "raw_label": tg_path.stem,
                "reason": (
                    "Filename does not match expected pattern "
                    "speakerid_session_activity_time"
                ),
            })
            n_skipped_files += 1
            continue

        # Read TextGrid
        tg = read_textgrid(tg_path)
        if tg is None:
            skipped.append({
                "file": str(tg_path),
                "raw_label": "",
                "reason": "Failed to read TextGrid file",
            })
            n_skipped_files += 1
            continue

        # Find vowel tier
        tier = find_vowel_tier(tg)
        if tier is None:
            available = [t.name for t in tg.tiers]
            skipped.append({
                "file": str(tg_path),
                "raw_label": "",
                "reason": (
                    f"No 'vowel' tier found. "
                    f"Available tiers: {available}"
                ),
            })
            n_skipped_files += 1
            continue

        n_processed += 1

        # ── Process intervals ────────────────────────────────────────────
        for interval in tier.intervals:
            raw_label = interval.text.strip()

            # Skip empty labels (silences / unlabelled regions)
            if not raw_label:
                continue

            n_intervals_read += 1

            start_sec = round(float(interval.xmin), 2)
            raw_duration_sec = float(interval.xmax) - float(interval.xmin)
            duration_sec = round(raw_duration_sec, 2)
            duration_ms = round(raw_duration_sec * 1000, 2)

            # Duration filter
            if duration_ms < min_duration_ms:
                n_removed_short += 1
                continue

            # Label corrections then parsing
            corrected_label = apply_label_corrections(raw_label)
            parsed = parse_label(corrected_label)

            if parsed is None:
                n_skipped_parse += 1
                skipped.append({
                    "file": str(tg_path),
                    "raw_label": raw_label,
                    "reason": f"Cannot parse label '{corrected_label}'",
                })
                continue

            vowel, word, register = parsed

            row: dict = {
                **file_meta,
                "word": word,
                "vowel": vowel,
                "register": register,
                "start_sec": start_sec,
                "duration_sec": duration_sec,
                "duration_ms": duration_ms,
            }
            # Acoustic columns — all NaN for now
            for col in ACOUSTIC_COLUMNS:
                row[col] = float("nan")

            rows.append(row)

    # ── Build DataFrame ──────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=ALL_COLUMNS) if rows else pd.DataFrame(columns=ALL_COLUMNS)

    # ── Remove unwanted registers (e.g. IDS(chant)) ─────────────────────────
    mask_remove = df["register"].isin(remove_registers)
    n_removed_chant = int(mask_remove.sum())
    df = df[~mask_remove].reset_index(drop=True)

    # ── Skipped DataFrame ────────────────────────────────────────────────────
    skipped_df = (
        pd.DataFrame(skipped, columns=["file", "raw_label", "reason"])
        if skipped
        else pd.DataFrame(columns=["file", "raw_label", "reason"])
    )

    stats = {
        "n_files_found":    n_files_found,
        "n_processed":      n_processed,
        "n_skipped_files":  n_skipped_files,
        "n_intervals_read": n_intervals_read,
        "n_removed_short":  n_removed_short,
        "n_removed_chant":  n_removed_chant,
        "n_skipped_parse":  n_skipped_parse,
        "final_rows":       len(df),
    }

    return df, skipped_df, stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config_path: Path) -> None:
    """Load config, validate paths, run extraction, and save outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    config = load_config(config_path)

    # ── Resolve input_folder (relative paths resolved from cwd) ─────────────
    input_folder = Path(config["paths"]["input_folder"])
    if not input_folder.is_absolute():
        input_folder = Path.cwd() / input_folder
    config["paths"]["input_folder"] = str(input_folder)

    if not input_folder.exists():
        logger.error("input_folder does not exist: %s", input_folder)
        raise SystemExit(1)

    # ── Resolve output paths (relative to cwd) ───────────────────────────────
    output_csv = Path(config["paths"]["output_csv"])
    if not output_csv.is_absolute():
        output_csv = Path.cwd() / output_csv

    skipped_csv = Path(config["paths"]["skipped_labels_csv"])
    if not skipped_csv.is_absolute():
        skipped_csv = Path.cwd() / skipped_csv

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    skipped_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Run pipeline ─────────────────────────────────────────────────────────
    df, skipped_df, stats = build_csv(config)

    # ── Save outputs ─────────────────────────────────────────────────────────
    df.to_csv(output_csv, index=False)
    logger.info("Saved metadata CSV -> %s  (%d rows)", output_csv, stats["final_rows"])

    if not skipped_df.empty:
        skipped_df.to_csv(skipped_csv, index=False)
        logger.info(
            "Saved skipped labels CSV -> %s  (%d rows)",
            skipped_csv, len(skipped_df),
        )
    else:
        logger.info("No skipped labels.")

    # ── Summary ───────────────────────────────────────────────────────────────
    min_ms = config["filters"]["min_duration_ms"]
    print("\n=== Pipeline Summary ===")
    print(f"  TextGrid files found          : {stats['n_files_found']}")
    print(f"  Files successfully processed  : {stats['n_processed']}")
    print(f"  Files skipped                 : {stats['n_skipped_files']}")
    print(f"  Intervals read (non-empty)    : {stats['n_intervals_read']}")
    print(f"  Removed (duration < {min_ms} ms)    : {stats['n_removed_short']}")
    print(f"  Removed (IDS(chant))          : {stats['n_removed_chant']}")
    print(f"  Skipped (parse errors)        : {stats['n_skipped_parse']}")
    print(f"  Final rows                    : {stats['final_rows']}")
    print(f"  Output CSV                    : {output_csv}")
    print(f"  Skipped labels CSV            : {skipped_csv}")
    print("========================\n")

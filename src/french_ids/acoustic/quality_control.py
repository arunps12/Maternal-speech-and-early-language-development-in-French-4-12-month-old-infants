from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .utils import safe_float

REQUIRED_METADATA_COLUMNS: list[str] = [
    "speakerid",
    "session",
    "activity",
    "time",
    "vowel",
    "start_sec",
    "duration_sec",
]

REQUIRED_CEILING_COLUMNS: list[str] = ["speakerid", "vowel", "formant_ceiling", "optimizer_version"]

VALID_ERROR_LABELS: set[str] = {
    "audio_file_missing",
    "segment_too_short",
    "invalid_time_interval",
    "empty_segment",
    "pitch_not_detected",
    "formant_extraction_failed",
    "ceiling_optimization_failed",
    "parselmouth_error",
    "unknown_error",
}


def validate_required_metadata_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {', '.join(missing)}")


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def validate_cached_ceiling_columns(df: pd.DataFrame, expected_version: int) -> None:
    validate_required_columns(df, REQUIRED_CEILING_COLUMNS)
    versions = set(df["optimizer_version"].dropna().astype(int).tolist())
    if versions != {expected_version}:
        raise ValueError(
            f"Cached ceiling optimizer version mismatch: expected {expected_version}, found {sorted(versions)}"
        )


def validate_acoustic_config(config: Mapping[str, object], *, base_dir: Path) -> None:
    if "paths" not in config:
        raise ValueError("config must define paths")

    paths = config["paths"]
    if not isinstance(paths, Mapping):
        raise ValueError("config.paths must be a mapping")

    required_path_keys = [
        "input_folder",
        "metadata_csv",
        "output_csv",
        "formant_ceiling_csv",
        "log_file",
    ]
    missing_path_keys = [key for key in required_path_keys if key not in paths]
    if missing_path_keys:
        raise ValueError(f"config.paths missing keys: {', '.join(missing_path_keys)}")

    input_folder = Path(paths["input_folder"])
    if not input_folder.is_absolute():
        input_folder = (base_dir / input_folder).resolve()
    if not input_folder.exists():
        raise FileNotFoundError(f"input_folder does not exist: {input_folder}")

    for section_name in ("pitch", "formant_ceiling", "formants", "error_handling", "logging"):
        if section_name not in config:
            raise ValueError(f"config missing section: {section_name}")

    pitch = config["pitch"]
    formant_ceiling = config["formant_ceiling"]
    formants = config["formants"]
    error_handling = config["error_handling"]

    if not isinstance(pitch, Mapping) or not isinstance(formant_ceiling, Mapping) or not isinstance(formants, Mapping):
        raise ValueError("pitch, formant_ceiling, and formants config sections must be mappings")
    if not isinstance(error_handling, Mapping):
        raise ValueError("error_handling config section must be a mapping")

    numeric_checks = [
        (pitch.get("pitch_floor_hz"), "pitch.pitch_floor_hz"),
        (pitch.get("pitch_ceiling_hz"), "pitch.pitch_ceiling_hz"),
        (formant_ceiling.get("fallback_ceiling_hz"), "formant_ceiling.fallback_ceiling_hz"),
        (formant_ceiling.get("min_tokens_per_group"), "formant_ceiling.min_tokens_per_group"),
        (formants.get("fallback_ceiling_hz"), "formants.fallback_ceiling_hz"),
    ]
    for value, label in numeric_checks:
        if safe_float(value) <= 0:
            raise ValueError(f"Invalid config value for {label}: {value}")

    status_column = error_handling.get("status_column")
    error_column = error_handling.get("error_column")
    if not status_column or not error_column:
        raise ValueError("error_handling.status_column and error_handling.error_column are required")


def validate_row_interval(row: Mapping[str, object], min_duration_ms: float) -> str | None:
    start_sec = safe_float(row.get("start_sec"))
    duration_sec = safe_float(row.get("duration_sec"))

    if math.isnan(start_sec) or math.isnan(duration_sec):
        return "invalid_time_interval"
    if start_sec < 0 or duration_sec <= 0:
        return "invalid_time_interval"
    if duration_sec * 1000 < min_duration_ms:
        return "segment_too_short"
    return None
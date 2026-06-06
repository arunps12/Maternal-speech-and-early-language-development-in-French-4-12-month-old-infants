from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

MERGE_KEYS: list[str] = ["speakerid", "vowel"]
STATUS_COLUMNS: list[str] = ["feature_status", "feature_error"]
AUDIT_LOG_COLUMNS: list[str] = [
    "row_index",
    "speakerid",
    "vowel",
    "audio_file",
    "start_sec",
    "duration_sec",
    "feature_status",
    "feature_error",
    "formant_ceiling",
]


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    if math.isfinite(result):
        return result
    return math.nan


def empty_feature_values(columns: list[str], fill_value: object = math.nan) -> dict[str, object]:
    return {column: fill_value for column in columns}


def failure_feature_values(
    columns: list[str],
    error_label: str,
    *,
    fill_value: object = math.nan,
    status_column: str = "feature_status",
    error_column: str = "feature_error",
) -> dict[str, object]:
    values = empty_feature_values(columns, fill_value=fill_value)
    values[status_column] = "failed"
    values[error_column] = error_label
    return values


def build_audio_stem(row: Mapping[str, object]) -> str:
    return "_".join(
        [
            str(row["speakerid"]),
            str(row["session"]),
            str(row["activity"]),
            str(row["time"]),
        ]
    )


def build_audit_log_row(
    index: int,
    row: Mapping[str, object],
    *,
    audio_file: str,
    status_column: str,
    error_column: str,
) -> dict[str, object]:
    return {
        "row_index": index,
        "speakerid": row.get("speakerid", ""),
        "vowel": row.get("vowel", ""),
        "audio_file": audio_file,
        "start_sec": row.get("start_sec", math.nan),
        "duration_sec": row.get("duration_sec", math.nan),
        "feature_status": row.get(status_column, ""),
        "feature_error": row.get(error_column, ""),
        "formant_ceiling": row.get("formant_ceiling", math.nan),
    }


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
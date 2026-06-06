from __future__ import annotations

import logging
import math
import statistics
from pathlib import Path
from typing import Mapping

import pandas as pd

from .audio import AudioRepository
from .formants import candidate_f1_f2_values
from .quality_control import validate_cached_ceiling_columns, validate_row_interval
from .utils import MERGE_KEYS

logger = logging.getLogger(__name__)

CURRENT_CEILING_OPTIMIZER_VERSION = 2


def estimate_formant_ceilings(
    metadata_df: pd.DataFrame,
    audio_repository: AudioRepository,
    config: Mapping[str, object],
    *,
    output_path: Path,
    recompute: bool = False,
) -> pd.DataFrame:
    ceiling_config = config["formant_ceiling"]
    logging_config = config.get("logging", {})
    cache_results = bool(ceiling_config.get("cache_results", False))
    log_each_ceiling_group = bool(logging_config.get("log_each_ceiling_group", False))

    if cache_results and output_path.exists() and not recompute:
        cached_df = _load_cached_ceilings(output_path, metadata_df)
        if cached_df is not None:
            logger.info("Reusing cached formant ceilings from %s", output_path)
            return cached_df

    rows: list[dict[str, object]] = []
    grouped = list(metadata_df.groupby(MERGE_KEYS, dropna=False))
    total_groups = len(grouped)
    for group_number, ((speakerid, vowel), group_df) in enumerate(grouped, start=1):
        if log_each_ceiling_group:
            logger.info(
                "Ceiling group %d/%d: speakerid=%s vowel=%s (%d row(s))",
                group_number,
                total_groups,
                speakerid,
                vowel,
                len(group_df),
            )

        rows.append(
            estimate_group_ceiling(
                speakerid=str(speakerid),
                vowel=str(vowel),
                group_df=group_df,
                audio_repository=audio_repository,
                config=config,
            )
        )

        if log_each_ceiling_group:
            group_result = rows[-1]
            logger.info(
                "Ceiling group %d/%d complete: speakerid=%s vowel=%s ceiling=%s status=%s",
                group_number,
                total_groups,
                speakerid,
                vowel,
                group_result["formant_ceiling"],
                group_result["optimization_status"],
            )

    ceiling_df = pd.DataFrame(
        rows,
        columns=[
            "speakerid",
            "vowel",
            "formant_ceiling",
            "n_tokens",
            "optimization_status",
            "optimizer_version",
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ceiling_df.to_csv(output_path, index=False)
    return ceiling_df


def estimate_group_ceiling(
    *,
    speakerid: str,
    vowel: str,
    group_df: pd.DataFrame,
    audio_repository: AudioRepository,
    config: Mapping[str, object],
) -> dict[str, object]:
    ceiling_config = config["formant_ceiling"]
    fallback_ceiling = float(ceiling_config.get("fallback_ceiling_hz", 5500))
    min_tokens = int(ceiling_config.get("min_tokens_per_group", 3))
    min_duration_ms = float(config["filters"]["min_duration_ms"])

    valid_segments: list[object] = []
    for _, row in group_df.iterrows():
        interval_error = validate_row_interval(row, min_duration_ms)
        if interval_error is not None:
            continue
        segment, _, error_label = audio_repository.get_segment_for_row(row)
        if error_label is None and segment is not None:
            valid_segments.append(segment)

    if len(valid_segments) < min_tokens:
        return {
            "speakerid": speakerid,
            "vowel": vowel,
            "formant_ceiling": fallback_ceiling,
            "n_tokens": len(valid_segments),
            "optimization_status": "fallback_too_few_tokens",
            "optimizer_version": CURRENT_CEILING_OPTIMIZER_VERSION,
        }

    coarse_candidates = _build_candidates(
        int(ceiling_config["coarse_search"]["min_ceiling_hz"]),
        int(ceiling_config["coarse_search"]["max_ceiling_hz"]),
        int(ceiling_config["coarse_search"]["step_hz"]),
    )
    best_coarse = _best_ceiling_for_segments(valid_segments, coarse_candidates, ceiling_config)
    if best_coarse is None:
        return {
            "speakerid": speakerid,
            "vowel": vowel,
            "formant_ceiling": fallback_ceiling,
            "n_tokens": len(valid_segments),
            "optimization_status": "fallback_optimization_failed",
            "optimizer_version": CURRENT_CEILING_OPTIMIZER_VERSION,
        }

    fine_window = int(ceiling_config["fine_search"]["window_hz"])
    fine_step = int(ceiling_config["fine_search"]["step_hz"])
    fine_candidates = _build_candidates(best_coarse - fine_window, best_coarse + fine_window, fine_step)
    best_fine = _best_ceiling_for_segments(valid_segments, fine_candidates, ceiling_config)
    if best_fine is None:
        return {
            "speakerid": speakerid,
            "vowel": vowel,
            "formant_ceiling": fallback_ceiling,
            "n_tokens": len(valid_segments),
            "optimization_status": "fallback_optimization_failed",
            "optimizer_version": CURRENT_CEILING_OPTIMIZER_VERSION,
        }

    return {
        "speakerid": speakerid,
        "vowel": vowel,
        "formant_ceiling": float(best_fine),
        "n_tokens": len(valid_segments),
        "optimization_status": "optimized",
        "optimizer_version": CURRENT_CEILING_OPTIMIZER_VERSION,
    }


def _load_cached_ceilings(output_path: Path, metadata_df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        cached_df = pd.read_csv(output_path)
        validate_cached_ceiling_columns(cached_df, CURRENT_CEILING_OPTIMIZER_VERSION)
    except Exception as exc:
        logger.warning("Cached formant ceilings are invalid and will be recomputed: %s", exc)
        return None

    expected_groups = metadata_df[MERGE_KEYS].drop_duplicates().sort_values(MERGE_KEYS).reset_index(drop=True)
    cached_groups = cached_df[MERGE_KEYS].drop_duplicates().sort_values(MERGE_KEYS).reset_index(drop=True)
    if not expected_groups.equals(cached_groups):
        logger.warning("Cached formant ceilings do not cover current metadata groups and will be recomputed")
        return None
    return cached_df


def _best_ceiling_for_segments(
    segments: list[object],
    candidates: list[int],
    config: Mapping[str, object],
) -> int | None:
    best_candidate: int | None = None
    best_score = math.inf
    for candidate in candidates:
        score = _score_candidate(candidate, segments, config)
        if score is None:
            continue
        if score < best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate


def _score_candidate(candidate: int, segments: list[object], config: Mapping[str, object]) -> float | None:
    f1_values: list[float] = []
    f2_values: list[float] = []
    for segment in segments:
        values = candidate_f1_f2_values(segment, candidate, config)
        if values is None:
            continue
        f1, f2 = values
        f1_values.append(f1)
        f2_values.append(f2)

    if len(f1_values) < 2 or len(f2_values) < 2:
        return None

    return statistics.pvariance([math.log(value) for value in f1_values]) + statistics.pvariance(
        [math.log(value) for value in f2_values]
    )


def _build_candidates(start: int, stop: int, step: int) -> list[int]:
    low = min(start, stop)
    high = max(start, stop)
    return list(range(low, high + step, step))
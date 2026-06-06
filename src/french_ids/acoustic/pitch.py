from __future__ import annotations

import math
from typing import Mapping

try:
    import parselmouth
except ImportError:  # pragma: no cover
    parselmouth = None  # type: ignore


def compute_pitch_features(segment: object, config: Mapping[str, object]) -> tuple[dict[str, float], str | None]:
    if parselmouth is None:
        raise RuntimeError("praat-parselmouth is required for acoustic extraction")

    try:
        pitch = segment.to_pitch(
            time_step=float(config.get("time_step", 0.0)),
            pitch_floor=float(config.get("pitch_floor_hz", 100.0)),
            pitch_ceiling=float(config.get("pitch_ceiling_hz", 600.0)),
        )
        values = [
            float(value)
            for value in pitch.selected_array["frequency"]
            if math.isfinite(float(value)) and float(value) > 0
        ]
    except Exception:
        return {
            "mean_pitch": math.nan,
            "min_pitch": math.nan,
            "max_pitch": math.nan,
            "pitch_range": math.nan,
        }, "parselmouth_error"

    if not values:
        return {
            "mean_pitch": math.nan,
            "min_pitch": math.nan,
            "max_pitch": math.nan,
            "pitch_range": math.nan,
        }, "pitch_not_detected"

    min_pitch = min(values)
    max_pitch = max(values)
    mean_pitch = sum(values) / len(values)
    return {
        "mean_pitch": mean_pitch,
        "min_pitch": min_pitch,
        "max_pitch": max_pitch,
        "pitch_range": max_pitch - min_pitch,
    }, None
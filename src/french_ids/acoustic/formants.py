from __future__ import annotations

import math
from typing import Iterable, Mapping

try:
    import parselmouth
except ImportError:  # pragma: no cover
    parselmouth = None  # type: ignore


def compute_formant_features(
    segment: object,
    ceiling_hz: float,
    config: Mapping[str, object],
) -> tuple[dict[str, float], str | None]:
    if parselmouth is None:
        raise RuntimeError("praat-parselmouth is required for acoustic extraction")

    try:
        formants = segment.to_formant_burg(
            time_step=float(config.get("time_step", 0.0025)),
            max_number_of_formants=float(config.get("number_of_formants", 5)),
            maximum_formant=float(ceiling_hz),
            window_length=float(config.get("window_length", 0.025)),
            pre_emphasis_from=float(config.get("pre_emphasis_from_hz", 50.0)),
        )
    except Exception:
        return _empty_formant_values(ceiling_hz), "parselmouth_error"

    duration = _get_duration(segment)
    sample_times = _sample_times(duration, float(config.get("time_step", 0.0025)))

    features: dict[str, float] = {"formant_ceiling": float(ceiling_hz)}
    valid_any = False
    for formant_number in range(1, 5):
        values = _collect_values(formants, formant_number, sample_times)
        if values:
            valid_any = True
            features[f"mean_F{formant_number}"] = sum(values) / len(values)
        else:
            features[f"mean_F{formant_number}"] = math.nan

    midpoint = duration / 2 if duration > 0 else 0.0
    for formant_number in range(1, 5):
        central = _value_at_time(formants, formant_number, midpoint)
        if math.isfinite(central) and central > 0:
            valid_any = True
            features[f"central_F{formant_number}"] = central
        else:
            features[f"central_F{formant_number}"] = math.nan

    if not valid_any:
        return _empty_formant_values(ceiling_hz), "formant_extraction_failed"
    return features, None


def candidate_f1_f2_values(
    segment: object,
    ceiling_hz: float,
    config: Mapping[str, object],
) -> tuple[float, float] | None:
    if parselmouth is None:
        raise RuntimeError("praat-parselmouth is required for acoustic extraction")

    try:
        formants = segment.to_formant_burg(
            time_step=float(config.get("time_step", 0.0025)),
            max_number_of_formants=float(config.get("number_of_formants", 5)),
            maximum_formant=float(ceiling_hz),
            window_length=float(config.get("window_length", 0.025)),
            pre_emphasis_from=float(config.get("pre_emphasis_from_hz", 50.0)),
        )
    except Exception:
        return None

    midpoint = _get_duration(segment) / 2
    f1 = _value_at_time(formants, 1, midpoint)
    f2 = _value_at_time(formants, 2, midpoint)
    if not math.isfinite(f1) or not math.isfinite(f2) or f1 <= 0 or f2 <= 0:
        return None
    return f1, f2


def _collect_values(formants: object, formant_number: int, sample_times: Iterable[float]) -> list[float]:
    values: list[float] = []
    for sample_time in sample_times:
        value = _value_at_time(formants, formant_number, sample_time)
        if math.isfinite(value) and value > 0:
            values.append(value)
    return values


def _sample_times(duration: float, time_step: float) -> list[float]:
    if duration <= 0:
        return [0.0]
    if time_step <= 0:
        return [duration / 2]

    times: list[float] = []
    current = max(time_step / 2, 0.0)
    while current < duration:
        times.append(current)
        current += time_step
    if not times:
        times.append(duration / 2)
    return times


def _value_at_time(formants: object, formant_number: int, sample_time: float) -> float:
    try:
        return float(formants.get_value_at_time(formant_number, sample_time, "Hertz"))
    except Exception:
        return math.nan


def _get_duration(segment: object) -> float:
    if hasattr(segment, "get_total_duration"):
        return float(segment.get_total_duration())
    xmin = getattr(segment, "xmin", 0.0)
    xmax = getattr(segment, "xmax", 0.0)
    return float(xmax) - float(xmin)


def _empty_formant_values(ceiling_hz: float) -> dict[str, float]:
    features: dict[str, float] = {"formant_ceiling": float(ceiling_hz)}
    for prefix in ("mean", "central"):
        for formant_number in range(1, 5):
            features[f"{prefix}_F{formant_number}"] = math.nan
    return features
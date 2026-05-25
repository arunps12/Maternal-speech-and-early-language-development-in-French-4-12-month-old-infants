"""Placeholder module for acoustic feature extraction via parselmouth / Praat.

All public functions currently return ``NaN`` for every column.  The module
structure is designed so that real Praat extraction logic can be added here
once the corresponding feature flags are enabled in ``config/config.yaml``.

To enable features:
    Set ``features.pitch: true`` and/or ``features.formants_mean: true``
    (and/or ``features.formants_central_frame: true``) in config/config.yaml,
    then implement the TODO blocks below using parselmouth calls.
"""
from __future__ import annotations

import math
from typing import Optional

try:
    import parselmouth
    from parselmouth.praat import call as praat_call  # noqa: F401 (for future use)
except ImportError:  # pragma: no cover
    parselmouth = None  # type: ignore

# ---------------------------------------------------------------------------
# Ordered column list — must stay in sync with build_metadata_csv.ALL_COLUMNS
# ---------------------------------------------------------------------------
ACOUSTIC_COLUMNS: list[str] = [
    "mean_pitch",
    "min_pitch",
    "max_pitch",
    "pitch_range",
    "formant_ceiling",
    "mean_F1",
    "mean_F2",
    "mean_F3",
    "mean_F4",
    "central_F1",
    "central_F2",
    "central_F3",
    "central_F4",
]


def _nan_dict(keys: list[str]) -> dict[str, float]:
    return {k: math.nan for k in keys}


# ---------------------------------------------------------------------------
# Pitch extraction (placeholder)
# ---------------------------------------------------------------------------

def extract_pitch(
    sound: Optional[object],
    start: float,
    end: float,
    config: dict,
) -> dict[str, float]:
    """Return pitch features for the vowel interval ``[start, end]``.

    Currently returns ``NaN`` for all pitch columns.

    Args:
        sound:  ``parselmouth.Sound`` for the full recording (unused for now).
        start:  Interval start time in seconds.
        end:    Interval end time in seconds.
        config: Pipeline config dict (``config["features"]`` section).

    Returns:
        Dict with keys ``mean_pitch``, ``min_pitch``, ``max_pitch``,
        ``pitch_range``.

    Future implementation sketch::

        part   = sound.extract_part(from_time=start, to_time=end)
        pitch  = part.to_pitch()
        values = pitch.selected_array["frequency"]
        values = values[values > 0]          # drop unvoiced frames
        if len(values):
            mean_pitch  = float(np.mean(values))
            min_pitch   = float(np.min(values))
            max_pitch   = float(np.max(values))
            pitch_range = max_pitch - min_pitch
        else:
            mean_pitch = min_pitch = max_pitch = pitch_range = math.nan
    """
    pitch_keys = ["mean_pitch", "min_pitch", "max_pitch", "pitch_range"]
    # TODO: implement when config["features"]["pitch"] is True
    return _nan_dict(pitch_keys)


# ---------------------------------------------------------------------------
# Formant extraction (placeholder)
# ---------------------------------------------------------------------------

def extract_formants(
    sound: Optional[object],
    start: float,
    end: float,
    config: dict,
) -> dict[str, float]:
    """Return formant features for the vowel interval ``[start, end]``.

    Currently returns ``NaN`` for all formant columns.

    Args:
        sound:  ``parselmouth.Sound`` for the full recording (unused for now).
        start:  Interval start time in seconds.
        end:    Interval end time in seconds.
        config: Pipeline config dict (``config["features"]`` section).

    Returns:
        Dict with keys ``formant_ceiling``, ``mean_F1``–``mean_F4``,
        ``central_F1``–``central_F4``.

    Future implementation sketch (Burg method with optimal ceiling)::

        part     = sound.extract_part(from_time=start, to_time=end)
        ceiling  = select_optimal_formant_ceiling(part, ...)  # see temp/Algo.txt
        formants = part.to_formant_burg(maximum_formant=ceiling)
        mid      = (start + end) / 2
        mean_F1  = formants.get_value_at_time(1, mid)
        ...
    """
    formant_keys = [
        "formant_ceiling",
        "mean_F1", "mean_F2", "mean_F3", "mean_F4",
        "central_F1", "central_F2", "central_F3", "central_F4",
    ]
    # TODO: implement when config["features"]["formants_mean"] or
    #       config["features"]["formants_central_frame"] is True
    return _nan_dict(formant_keys)

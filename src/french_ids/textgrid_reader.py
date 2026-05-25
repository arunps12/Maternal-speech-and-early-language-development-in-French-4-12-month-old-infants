from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import parselmouth
from parselmouth.praat import call as _praat

logger = logging.getLogger(__name__)


def find_textgrid_files(folder: Path) -> list[Path]:
    """Recursively find all ``.TextGrid`` and ``.textgrid`` files under *folder*.

    Duplicates that arise from case-insensitive filesystems are deduplicated by
    resolved path.
    """
    seen: set[Path] = set()
    unique: list[Path] = []
    for pattern in ("**/*.TextGrid", "**/*.textgrid"):
        for f in folder.glob(pattern):
            key = f.resolve()
            if key not in seen:
                seen.add(key)
                unique.append(f)
    return sorted(unique)


def find_vowel_tier(tg: parselmouth.TextGrid) -> Optional[SimpleNamespace]:
    """Return a SimpleNamespace with an ``.intervals`` list for the vowel tier.

    Each interval has ``.xmin``, ``.xmax``, and ``.text`` attributes,
    mirroring the parselmouth Interval interface so ``build_metadata_csv.py``
    needs no changes.

    Uses ``parselmouth.praat.call()`` because parselmouth.TextGrid does not
    expose a ``.tiers`` Python attribute in v0.4.x.

    Logs a warning with available tier names when no vowel tier is found.
    Returns ``None`` on failure.
    """
    try:
        n_tiers = int(_praat(tg, "Get number of tiers"))
    except Exception as exc:
        logger.warning("Could not get number of tiers: %s", exc)
        return None

    vowel_idx: Optional[int] = None
    available: list[str] = []
    for i in range(1, n_tiers + 1):
        try:
            name: str = _praat(tg, "Get tier name", i)
        except Exception:
            name = f"<tier {i}>"
        available.append(name)
        if name.strip().lower() == "vowel" and vowel_idx is None:
            vowel_idx = i

    if vowel_idx is None:
        logger.warning("No 'vowel' tier found. Available tiers: %s", available)
        return None

    # Verify it is an interval tier, not a point tier
    try:
        is_interval = bool(_praat(tg, "Is interval tier", vowel_idx))
    except Exception:
        is_interval = True  # assume interval tier if the check is unsupported
    if not is_interval:
        logger.warning("Tier 'vowel' is not an interval tier.")
        return None

    try:
        n_intervals = int(_praat(tg, "Get number of intervals", vowel_idx))
    except Exception as exc:
        logger.warning("Could not get number of intervals: %s", exc)
        return None

    intervals: list[SimpleNamespace] = []
    for j in range(1, n_intervals + 1):
        try:
            xmin = float(_praat(tg, "Get start time of interval", vowel_idx, j))
            xmax = float(_praat(tg, "Get end time of interval", vowel_idx, j))
            text: str = _praat(tg, "Get label of interval", vowel_idx, j)
            intervals.append(SimpleNamespace(xmin=xmin, xmax=xmax, text=text))
        except Exception as exc:
            logger.warning("Could not read interval %d: %s", j, exc)

    return SimpleNamespace(intervals=intervals)


def read_textgrid(path: Path) -> Optional[parselmouth.TextGrid]:
    """Read *path* with parselmouth and return a :class:`parselmouth.TextGrid`.

    Returns ``None`` and logs a warning if the file cannot be read or is not a
    TextGrid object.
    """
    try:
        obj = parselmouth.read(str(path))
        if not isinstance(obj, parselmouth.TextGrid):
            logger.warning("File is not a TextGrid: %s", path)
            return None
        return obj
    except Exception as exc:
        logger.warning("Failed to read TextGrid '%s': %s", path, exc)
        return None

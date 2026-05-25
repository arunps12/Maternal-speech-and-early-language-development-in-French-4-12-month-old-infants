from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import parselmouth

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


def find_vowel_tier(tg: parselmouth.TextGrid) -> Optional[parselmouth.IntervalTier]:
    """Return the first tier whose name is ``'vowel'`` (case-insensitive).

    Logs a warning with available tier names when no vowel tier is found.
    Returns ``None`` on failure.
    """
    for tier in tg.tiers:
        if tier.name.strip().lower() == "vowel":
            return tier
    available = [t.name for t in tg.tiers]
    logger.warning(
        "No 'vowel' tier found. Available tiers: %s",
        available,
    )
    return None


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

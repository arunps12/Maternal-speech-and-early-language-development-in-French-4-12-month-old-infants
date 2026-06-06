from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from .utils import build_audio_stem, safe_float

try:
    import parselmouth
except ImportError:  # pragma: no cover
    parselmouth = None  # type: ignore


class AudioRepository:
    def __init__(self, input_folder: Path):
        self.input_folder = input_folder
        self._audio_path_cache: dict[str, Path | None] = {}
        self._sound_cache: dict[Path, object] = {}

    def locate_audio_file(self, row: Mapping[str, object]) -> Path | None:
        stem = build_audio_stem(row)
        if stem in self._audio_path_cache:
            return self._audio_path_cache[stem]

        for pattern in (f"**/{stem}.wav", f"**/{stem}.WAV"):
            matches = list(self.input_folder.glob(pattern))
            if matches:
                resolved = matches[0].resolve()
                self._audio_path_cache[stem] = resolved
                return resolved

        self._audio_path_cache[stem] = None
        return None

    def load_sound(self, audio_path: Path) -> object:
        if audio_path in self._sound_cache:
            return self._sound_cache[audio_path]
        if parselmouth is None:
            raise RuntimeError("praat-parselmouth is required for acoustic extraction")
        sound = parselmouth.Sound(str(audio_path))
        self._sound_cache[audio_path] = sound
        return sound

    def extract_segment(self, sound: object, start_sec: float, end_sec: float) -> object:
        return sound.extract_part(from_time=start_sec, to_time=end_sec, preserve_times=False)

    def get_segment_for_row(
        self,
        row: Mapping[str, object],
    ) -> tuple[object | None, Path | None, str | None]:
        audio_path = self.locate_audio_file(row)
        if audio_path is None:
            return None, None, "audio_file_missing"

        start_sec = safe_float(row.get("start_sec"))
        duration_sec = safe_float(row.get("duration_sec"))
        end_sec = start_sec + duration_sec
        if math.isnan(start_sec) or math.isnan(duration_sec) or end_sec <= start_sec:
            return None, audio_path, "invalid_time_interval"

        sound = self.load_sound(audio_path)
        total_duration = self._get_sound_duration(sound)
        if start_sec < 0 or end_sec > total_duration:
            return None, audio_path, "invalid_time_interval"

        segment = self.extract_segment(sound, start_sec, end_sec)
        segment_duration = self._get_sound_duration(segment)
        if segment_duration <= 0:
            return None, audio_path, "empty_segment"
        return segment, audio_path, None

    @staticmethod
    def _get_sound_duration(sound: object) -> float:
        if hasattr(sound, "get_total_duration"):
            return float(sound.get_total_duration())
        xmin = getattr(sound, "xmin", 0.0)
        xmax = getattr(sound, "xmax", 0.0)
        return float(xmax) - float(xmin)
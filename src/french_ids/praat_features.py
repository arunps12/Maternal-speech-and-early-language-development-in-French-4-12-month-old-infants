"""Acoustic feature extraction pipeline built on praat-parselmouth."""
from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .acoustic import (
    AUDIT_LOG_COLUMNS,
    MERGE_KEYS,
    AudioRepository,
    build_audit_log_row,
    compute_formant_features,
    compute_pitch_features,
    empty_feature_values,
    estimate_formant_ceilings,
    failure_feature_values,
    validate_acoustic_config,
    validate_required_metadata_columns,
    validate_row_interval,
)
from .acoustic.utils import resolve_path

logger = logging.getLogger(__name__)

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


class AcousticFeatureExtractor:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        config_path: Path | None = None,
        recompute_ceilings: bool = False,
        metadata_csv: str | Path | None = None,
        output_csv: str | Path | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.config_path = config_path.resolve() if config_path is not None else None
        self.base_dir = self.config_path.parent.parent if self.config_path is not None else Path.cwd()
        self.recompute_ceilings = recompute_ceilings
        if metadata_csv is not None:
            self.config["paths"]["metadata_csv"] = str(metadata_csv)
        if output_csv is not None:
            self.config["paths"]["output_csv"] = str(output_csv)

        validate_acoustic_config(self.config, base_dir=self.base_dir)

        self.paths = self._resolve_paths()
        self.audio_repository = AudioRepository(self.paths["input_folder"])

        error_handling = self.config["error_handling"]
        self.status_column = str(error_handling["status_column"])
        self.error_column = str(error_handling["error_column"])
        self.fill_failed_features_with = error_handling.get("fill_failed_features_with", math.nan)
        logging_config = self.config.get("logging", {})
        self.progress_every_n_rows = int(logging_config.get("progress_every_n_rows", 0) or 0)

        self.metadata_df: pd.DataFrame | None = None
        self.ceiling_df: pd.DataFrame | None = None
        self.audit_log_df: pd.DataFrame | None = None
        self._ceilings_assigned = False
        self._audio_paths: dict[int, str] = {}

    def load_metadata(self) -> pd.DataFrame:
        metadata_path = self.paths["metadata_csv"]
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata CSV does not exist: {metadata_path}")

        logger.info("Loading metadata CSV from %s", metadata_path)
        df = pd.read_csv(metadata_path)
        validate_required_metadata_columns(df)

        for column in ACOUSTIC_COLUMNS:
            if column not in df.columns:
                df[column] = math.nan

        if self.status_column not in df.columns:
            df[self.status_column] = ""
        if self.error_column not in df.columns:
            df[self.error_column] = ""

        self.metadata_df = df
        logger.info("Loaded %d metadata row(s)", len(df))
        return df

    def estimate_formant_ceilings(self) -> pd.DataFrame:
        df = self._require_metadata()
        output_path = self.paths["formant_ceiling_csv"]
        logger.info("Estimating formant ceilings for %d speaker-vowel group(s)", len(df[MERGE_KEYS].drop_duplicates()))
        self.ceiling_df = estimate_formant_ceilings(
            df,
            self.audio_repository,
            self.config,
            output_path=output_path,
            recompute=self.recompute_ceilings,
        )

        merged = df.drop(columns=["formant_ceiling"], errors="ignore").merge(
            self.ceiling_df[[*MERGE_KEYS, "formant_ceiling"]],
            on=MERGE_KEYS,
            how="left",
        )
        if merged["formant_ceiling"].isna().any():
            missing = merged[merged["formant_ceiling"].isna()][MERGE_KEYS].drop_duplicates()
            raise RuntimeError(
                "Missing merged formant ceilings for groups: "
                + ", ".join(f"({row.speakerid}, {row.vowel})" for row in missing.itertuples(index=False))
            )

        self.metadata_df = merged
        self._ceilings_assigned = True
        logger.info("Assigned formant ceilings to %d metadata row(s)", len(merged))
        return self.ceiling_df

    def compute_formant_features(self) -> pd.DataFrame:
        if not self._ceilings_assigned:
            raise RuntimeError("Formant ceilings must be estimated before computing formant features")

        df = self._require_metadata()
        min_duration_ms = float(self.config["filters"]["min_duration_ms"])
        formant_config = self.config["formants"]
        logger.info("Computing formant features for %d row(s)", len(df))

        total_rows = len(df)
        for row_number, (index, row) in enumerate(df.iterrows(), start=1):
            self._maybe_log_row_progress("formant", row_number, total_rows)

            if self._is_failed(row):
                continue

            interval_error = validate_row_interval(row, min_duration_ms)
            if interval_error is not None:
                self._mark_failed(index, interval_error)
                continue

            segment, audio_path, error_label = self.audio_repository.get_segment_for_row(row)
            self._remember_audio_path(index, audio_path)
            if error_label is not None or segment is None:
                self._mark_failed(index, error_label or "unknown_error")
                continue

            features, formant_error = compute_formant_features(
                segment,
                float(row["formant_ceiling"]),
                formant_config,
            )
            if formant_error is not None:
                self._mark_failed(index, formant_error)
                continue

            for column, value in features.items():
                df.at[index, column] = value

        return df

    def compute_pitch_features(self) -> pd.DataFrame:
        df = self._require_metadata()
        min_duration_ms = float(self.config["filters"]["min_duration_ms"])
        pitch_config = self.config["pitch"]
        logger.info("Computing pitch features for %d row(s)", len(df))

        total_rows = len(df)
        for row_number, (index, row) in enumerate(df.iterrows(), start=1):
            self._maybe_log_row_progress("pitch", row_number, total_rows)

            if self._is_failed(row):
                continue

            interval_error = validate_row_interval(row, min_duration_ms)
            if interval_error is not None:
                self._mark_failed(index, interval_error)
                continue

            segment, audio_path, error_label = self.audio_repository.get_segment_for_row(row)
            self._remember_audio_path(index, audio_path)
            if error_label is not None or segment is None:
                self._mark_failed(index, error_label or "unknown_error")
                continue

            features, pitch_error = compute_pitch_features(segment, pitch_config)
            if pitch_error is not None:
                self._mark_failed(index, pitch_error)
                continue

            for column, value in features.items():
                df.at[index, column] = value

            df.at[index, self.status_column] = "success"
            df.at[index, self.error_column] = ""

        return df

    def compute_all_features(self) -> pd.DataFrame:
        logger.info("Stage 1/5: load metadata")
        self.load_metadata()
        logger.info("Stage 2/5: estimate and merge formant ceilings")
        self.estimate_formant_ceilings()
        logger.info("Stage 3/5: compute formant features")
        self.compute_formant_features()
        logger.info("Stage 4/5: compute pitch features")
        self.compute_pitch_features()
        logger.info("Stage 5/5: finalize rows and build audit log")
        self._finalize_success_rows()
        self._build_audit_log()
        return self._require_metadata()

    def save_outputs(self) -> None:
        df = self._require_metadata()
        output_path = self.paths["output_csv"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved acoustic feature CSV -> %s  (%d rows)", output_path, len(df))

        if self.ceiling_df is not None:
            ceiling_path = self.paths["formant_ceiling_csv"]
            ceiling_path.parent.mkdir(parents=True, exist_ok=True)
            self.ceiling_df.to_csv(ceiling_path, index=False)
            logger.info("Saved formant ceiling CSV -> %s  (%d rows)", ceiling_path, len(self.ceiling_df))

        if bool(self.config["logging"].get("save_log", True)):
            if self.audit_log_df is None:
                self._build_audit_log()
            log_path = self.paths["log_file"]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.audit_log_df.to_csv(log_path, index=False)
            logger.info("Saved acoustic audit log -> %s  (%d rows)", log_path, len(self.audit_log_df))

    def _resolve_paths(self) -> dict[str, Path]:
        paths = self.config["paths"]
        return {
            "input_folder": resolve_path(paths["input_folder"], self.base_dir),
            "metadata_csv": resolve_path(paths["metadata_csv"], self.base_dir),
            "output_csv": resolve_path(paths["output_csv"], self.base_dir),
            "formant_ceiling_csv": resolve_path(paths["formant_ceiling_csv"], self.base_dir),
            "log_file": resolve_path(paths["log_file"], self.base_dir),
        }

    def _require_metadata(self) -> pd.DataFrame:
        if self.metadata_df is None:
            raise RuntimeError("Metadata must be loaded before computing acoustic features")
        return self.metadata_df

    def _is_failed(self, row: pd.Series) -> bool:
        return str(row.get(self.status_column, "")) == "failed"

    def _remember_audio_path(self, index: int, audio_path: Path | None) -> None:
        self._audio_paths[index] = "" if audio_path is None else str(audio_path)

    def _mark_failed(self, index: int, error_label: str) -> None:
        df = self._require_metadata()
        updates = failure_feature_values(
            ACOUSTIC_COLUMNS,
            error_label,
            fill_value=self.fill_failed_features_with,
            status_column=self.status_column,
            error_column=self.error_column,
        )
        for column, value in updates.items():
            df.at[index, column] = value

    def _finalize_success_rows(self) -> None:
        df = self._require_metadata()
        for index, row in df.iterrows():
            if self._is_failed(row):
                continue
            df.at[index, self.status_column] = "success"
            df.at[index, self.error_column] = ""

    def _build_audit_log(self) -> pd.DataFrame:
        df = self._require_metadata()
        ceiling_lookup: dict[tuple[object, object], float] = {}
        if self.ceiling_df is not None:
            for ceiling_row in self.ceiling_df.itertuples(index=False):
                ceiling_lookup[(ceiling_row.speakerid, ceiling_row.vowel)] = float(ceiling_row.formant_ceiling)

        audit_rows: list[dict[str, object]] = []
        for index, row in df.iterrows():
            row_mapping = row.to_dict()
            if pd.isna(row_mapping.get("formant_ceiling")):
                key = (row_mapping.get("speakerid"), row_mapping.get("vowel"))
                if key in ceiling_lookup:
                    row_mapping["formant_ceiling"] = ceiling_lookup[key]

            audit_rows.append(
                build_audit_log_row(
                    index,
                    row_mapping,
                    audio_file=self._audio_paths.get(index, ""),
                    status_column=self.status_column,
                    error_column=self.error_column,
                )
            )

        self.audit_log_df = pd.DataFrame(audit_rows, columns=AUDIT_LOG_COLUMNS)
        return self.audit_log_df

    def _maybe_log_row_progress(self, phase: str, row_number: int, total_rows: int) -> None:
        if self.progress_every_n_rows <= 0:
            return
        if row_number % self.progress_every_n_rows == 0 or row_number == total_rows:
            logger.info("%s progress: processed %d/%d row(s)", phase.capitalize(), row_number, total_rows)


def extract_pitch(sound: object, start: float, end: float, config: dict[str, Any]) -> dict[str, float]:
    del sound, start, end, config
    values = empty_feature_values(["mean_pitch", "min_pitch", "max_pitch", "pitch_range"])
    return {key: float(value) for key, value in values.items()}


def extract_formants(sound: object, start: float, end: float, config: dict[str, Any]) -> dict[str, float]:
    del sound, start, end, config
    values = empty_feature_values(
        [
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
    )
    return {key: float(value) for key, value in values.items()}
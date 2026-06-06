from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from french_ids.acoustic import ceilings
from french_ids.acoustic import formants as formant_helpers
from french_ids.config import load_config
from french_ids.praat_features import ACOUSTIC_COLUMNS, AcousticFeatureExtractor


def _base_config(tmp_path: Path) -> dict:
    input_folder = tmp_path / "input"
    input_folder.mkdir()
    return {
        "paths": {
            "input_folder": str(input_folder),
            "metadata_csv": str(tmp_path / "metadata.csv"),
            "output_csv": str(tmp_path / "acoustic.csv"),
            "formant_ceiling_csv": str(tmp_path / "formant_ceilings.csv"),
            "skipped_labels_csv": str(tmp_path / "skipped_labels.csv"),
            "log_file": str(tmp_path / "acoustic_feature_log.csv"),
        },
        "filters": {
            "min_duration_ms": 30,
            "remove_registers": ["IDS(chant)"],
        },
        "features": {
            "pitch": True,
            "formant_ceiling": True,
            "formants_mean": True,
            "formants_central_frame": True,
        },
        "pitch": {
            "method": "praat_parselmouth",
            "time_step": 0.0,
            "pitch_floor_hz": 100,
            "pitch_ceiling_hz": 600,
            "unit": "Hertz",
        },
        "formant_ceiling": {
            "method": "speaker_vowel_optimization",
            "group_by": ["speakerid", "vowel"],
            "coarse_search": {
                "min_ceiling_hz": 4500,
                "max_ceiling_hz": 6500,
                "step_hz": 100,
            },
            "fine_search": {
                "window_hz": 100,
                "step_hz": 10,
            },
            "number_of_formants": 5,
            "time_step": 0.0025,
            "window_length": 0.025,
            "pre_emphasis_from_hz": 50,
            "selection_criterion": "min_variance_log_F1_F2",
            "min_tokens_per_group": 3,
            "fallback_ceiling_hz": 5500,
            "cache_results": True,
        },
        "formants": {
            "method": "praat_burg_parselmouth",
            "number_of_formants": 5,
            "time_step": 0.0025,
            "window_length": 0.025,
            "pre_emphasis_from_hz": 50,
            "use_speaker_vowel_ceiling": True,
            "fallback_ceiling_hz": 5500,
            "output_formants": ["F1", "F2", "F3", "F4"],
            "central_frame": {"method": "midpoint"},
        },
        "error_handling": {
            "continue_on_error": True,
            "fill_failed_features_with": None,
            "add_status_columns": True,
            "status_column": "feature_status",
            "error_column": "feature_error",
            "catch_parselmouth_errors": True,
        },
        "logging": {
            "save_log": True,
            "verbose": True,
        },
    }


def _metadata_df(rows: list[dict] | None = None) -> pd.DataFrame:
    if rows is None:
        rows = [
            {
                "speakerid": "M01",
                "session": "4m",
                "activity": "bath",
                "time": "0800",
                "word": "papa",
                "vowel": "a",
                "register": "IDS",
                "start_sec": 0.10,
                "duration_sec": 0.08,
                "duration_ms": 80.0,
            }
        ]
    return pd.DataFrame(rows)


def _write_metadata(tmp_path: Path, config: dict, rows: list[dict] | None = None) -> Path:
    metadata_path = Path(config["paths"]["metadata_csv"])
    df = _metadata_df(rows)
    df.to_csv(metadata_path, index=False)
    return metadata_path


def _success_formants(ceiling: float) -> dict[str, float]:
    return {
        "formant_ceiling": ceiling,
        "mean_F1": 500.0,
        "mean_F2": 1500.0,
        "mean_F3": 2500.0,
        "mean_F4": 3500.0,
        "central_F1": 510.0,
        "central_F2": 1510.0,
        "central_F3": 2510.0,
        "central_F4": 3510.0,
    }


def _success_pitch() -> dict[str, float]:
    return {
        "mean_pitch": 240.0,
        "min_pitch": 220.0,
        "max_pitch": 260.0,
        "pitch_range": 40.0,
    }


def test_acoustic_config_keys_load_correctly():
    config = load_config(Path("config/config.yaml"))
    assert config["paths"]["metadata_csv"].endswith("french_vowels_metadata.csv")
    assert config["paths"]["formant_ceiling_csv"].endswith("formant_ceilings.csv")
    assert config["pitch"]["pitch_floor_hz"] == 100
    assert config["formant_ceiling"]["fallback_ceiling_hz"] == 5500
    assert config["logging"]["progress_every_n_rows"] == 500
    assert config["logging"]["log_each_ceiling_group"] is True


def test_required_metadata_columns_are_validated(tmp_path: Path):
    config = _base_config(tmp_path)
    pd.DataFrame([{"speakerid": "M01"}]).to_csv(config["paths"]["metadata_csv"], index=False)

    extractor = AcousticFeatureExtractor(config)
    with pytest.raises(ValueError, match="Missing required metadata columns"):
        extractor.load_metadata()


def test_acoustic_output_columns_are_created(tmp_path: Path):
    config = _base_config(tmp_path)
    _write_metadata(tmp_path, config)

    extractor = AcousticFeatureExtractor(config)
    df = extractor.load_metadata()

    for column in ACOUSTIC_COLUMNS:
        assert column in df.columns
    assert "feature_status" in df.columns
    assert "feature_error" in df.columns


def test_missing_audio_file_does_not_crash_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = _base_config(tmp_path)
    _write_metadata(tmp_path, config)

    monkeypatch.setattr(
        "french_ids.praat_features.estimate_formant_ceilings",
        lambda metadata_df, audio_repository, config, output_path, recompute=False: pd.DataFrame(
            [{"speakerid": "M01", "vowel": "a", "formant_ceiling": 5500.0, "n_tokens": 0, "optimization_status": "fallback_too_few_tokens"}]
        ),
    )

    extractor = AcousticFeatureExtractor(config)
    monkeypatch.setattr(
        extractor.audio_repository,
        "get_segment_for_row",
        lambda row: (None, None, "audio_file_missing"),
    )

    result = extractor.compute_all_features()
    extractor.save_outputs()
    log_df = pd.read_csv(config["paths"]["log_file"])

    assert result.loc[0, "feature_status"] == "failed"
    assert result.loc[0, "feature_error"] == "audio_file_missing"
    assert pd.isna(result.loc[0, "mean_pitch"])
    assert pd.isna(result.loc[0, "mean_F1"])
    assert log_df.loc[0, "formant_ceiling"] == 5500.0


def test_too_few_token_group_uses_fallback_ceiling(tmp_path: Path):
    config = _base_config(tmp_path)
    group_df = _metadata_df(
        [
            {
                "speakerid": "M01",
                "session": "4m",
                "activity": "bath",
                "time": "0800",
                "word": "papa",
                "vowel": "a",
                "register": "IDS",
                "start_sec": 0.10,
                "duration_sec": 0.08,
                "duration_ms": 80.0,
            },
            {
                "speakerid": "M01",
                "session": "4m",
                "activity": "bath",
                "time": "0801",
                "word": "mama",
                "vowel": "a",
                "register": "IDS",
                "start_sec": 0.30,
                "duration_sec": 0.09,
                "duration_ms": 90.0,
            },
        ]
    )

    class FakeAudioRepository:
        def get_segment_for_row(self, row):
            return object(), tmp_path / "dummy.wav", None

    result = ceilings.estimate_group_ceiling(
        speakerid="M01",
        vowel="a",
        group_df=group_df,
        audio_repository=FakeAudioRepository(),
        config=config,
    )

    assert result["formant_ceiling"] == 5500
    assert result["optimization_status"] == "fallback_too_few_tokens"


def test_pipeline_continues_if_one_row_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = _base_config(tmp_path)
    _write_metadata(
        tmp_path,
        config,
        rows=[
            {
                "speakerid": "M01",
                "session": "4m",
                "activity": "bath",
                "time": "0800",
                "word": "papa",
                "vowel": "a",
                "register": "IDS",
                "start_sec": 0.10,
                "duration_sec": 0.08,
                "duration_ms": 80.0,
            },
            {
                "speakerid": "M02",
                "session": "4m",
                "activity": "bath",
                "time": "0801",
                "word": "mama",
                "vowel": "i",
                "register": "IDS",
                "start_sec": 0.20,
                "duration_sec": 0.08,
                "duration_ms": 80.0,
            },
        ],
    )

    monkeypatch.setattr(
        "french_ids.praat_features.estimate_formant_ceilings",
        lambda metadata_df, audio_repository, config, output_path, recompute=False: pd.DataFrame(
            [
                {"speakerid": "M01", "vowel": "a", "formant_ceiling": 5400.0, "n_tokens": 1, "optimization_status": "fallback_too_few_tokens"},
                {"speakerid": "M02", "vowel": "i", "formant_ceiling": 5600.0, "n_tokens": 1, "optimization_status": "fallback_too_few_tokens"},
            ]
        ),
    )

    extractor = AcousticFeatureExtractor(config)

    def fake_get_segment(row):
        if row["speakerid"] == "M01":
            return None, None, "audio_file_missing"
        return object(), tmp_path / "M02.wav", None

    monkeypatch.setattr(extractor.audio_repository, "get_segment_for_row", fake_get_segment)
    monkeypatch.setattr(
        "french_ids.praat_features.compute_formant_features",
        lambda segment, ceiling_hz, config: (_success_formants(ceiling_hz), None),
    )
    monkeypatch.setattr(
        "french_ids.praat_features.compute_pitch_features",
        lambda segment, config: (_success_pitch(), None),
    )

    result = extractor.compute_all_features()

    assert result.loc[0, "feature_status"] == "failed"
    assert result.loc[0, "feature_error"] == "audio_file_missing"
    assert result.loc[1, "feature_status"] == "success"
    assert result.loc[1, "feature_error"] == ""
    assert result.loc[1, "mean_pitch"] == 240.0


def test_formants_are_not_computed_before_ceiling_estimation(tmp_path: Path):
    config = _base_config(tmp_path)
    _write_metadata(tmp_path, config)

    extractor = AcousticFeatureExtractor(config)
    extractor.load_metadata()

    with pytest.raises(RuntimeError, match="Formant ceilings must be estimated"):
        extractor.compute_formant_features()


def test_formant_extraction_uses_row_specific_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = _base_config(tmp_path)
    _write_metadata(
        tmp_path,
        config,
        rows=[
            {
                "speakerid": "M01",
                "session": "4m",
                "activity": "bath",
                "time": "0800",
                "word": "papa",
                "vowel": "a",
                "register": "IDS",
                "start_sec": 0.10,
                "duration_sec": 0.08,
                "duration_ms": 80.0,
            },
            {
                "speakerid": "M02",
                "session": "4m",
                "activity": "bath",
                "time": "0801",
                "word": "mama",
                "vowel": "i",
                "register": "IDS",
                "start_sec": 0.20,
                "duration_sec": 0.08,
                "duration_ms": 80.0,
            },
        ],
    )

    monkeypatch.setattr(
        "french_ids.praat_features.estimate_formant_ceilings",
        lambda metadata_df, audio_repository, config, output_path, recompute=False: pd.DataFrame(
            [
                {"speakerid": "M01", "vowel": "a", "formant_ceiling": 5100.0, "n_tokens": 1, "optimization_status": "fallback_too_few_tokens"},
                {"speakerid": "M02", "vowel": "i", "formant_ceiling": 5900.0, "n_tokens": 1, "optimization_status": "fallback_too_few_tokens"},
            ]
        ),
    )

    extractor = AcousticFeatureExtractor(config)
    monkeypatch.setattr(
        extractor.audio_repository,
        "get_segment_for_row",
        lambda row: (object(), tmp_path / f"{row['speakerid']}.wav", None),
    )

    ceilings_seen: list[float] = []

    def fake_formants(segment, ceiling_hz, config):
        ceilings_seen.append(ceiling_hz)
        return _success_formants(ceiling_hz), None

    monkeypatch.setattr("french_ids.praat_features.compute_formant_features", fake_formants)
    monkeypatch.setattr(
        "french_ids.praat_features.compute_pitch_features",
        lambda segment, config: (_success_pitch(), None),
    )

    result = extractor.compute_all_features()

    assert ceilings_seen == [5100.0, 5900.0]
    assert list(result["formant_ceiling"]) == [5100.0, 5900.0]


def test_invalid_cached_ceiling_table_is_recomputed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = _base_config(tmp_path)
    metadata_df = _metadata_df()
    cache_path = Path(config["paths"]["formant_ceiling_csv"])
    pd.DataFrame([{"speakerid": "M01", "vowel": "a"}]).to_csv(cache_path, index=False)

    monkeypatch.setattr(
        ceilings,
        "estimate_group_ceiling",
        lambda speakerid, vowel, group_df, audio_repository, config: {
            "speakerid": speakerid,
            "vowel": vowel,
            "formant_ceiling": 5500.0,
            "n_tokens": len(group_df),
            "optimization_status": "fallback_too_few_tokens",
        },
    )

    class FakeAudioRepository:
        def get_segment_for_row(self, row):
            return object(), tmp_path / "dummy.wav", None

    result = ceilings.estimate_formant_ceilings(
        metadata_df,
        FakeAudioRepository(),
        config,
        output_path=cache_path,
        recompute=False,
    )

    assert result.loc[0, "formant_ceiling"] == 5500.0
    assert result.loc[0, "optimization_status"] == "fallback_too_few_tokens"


def test_legacy_cached_ceiling_table_without_optimizer_version_is_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _base_config(tmp_path)
    metadata_df = _metadata_df()
    cache_path = Path(config["paths"]["formant_ceiling_csv"])
    pd.DataFrame(
        [
            {
                "speakerid": "M01",
                "vowel": "a",
                "formant_ceiling": 5500.0,
                "n_tokens": 3,
                "optimization_status": "optimized",
            }
        ]
    ).to_csv(cache_path, index=False)

    monkeypatch.setattr(
        ceilings,
        "estimate_group_ceiling",
        lambda speakerid, vowel, group_df, audio_repository, config: {
            "speakerid": speakerid,
            "vowel": vowel,
            "formant_ceiling": 5120.0,
            "n_tokens": len(group_df),
            "optimization_status": "optimized",
            "optimizer_version": ceilings.CURRENT_CEILING_OPTIMIZER_VERSION,
        },
    )

    class FakeAudioRepository:
        def get_segment_for_row(self, row):
            return object(), tmp_path / "dummy.wav", None

    result = ceilings.estimate_formant_ceilings(
        metadata_df,
        FakeAudioRepository(),
        config,
        output_path=cache_path,
        recompute=False,
    )

    assert result.loc[0, "formant_ceiling"] == 5120.0
    assert result.loc[0, "optimizer_version"] == ceilings.CURRENT_CEILING_OPTIMIZER_VERSION


def test_formant_value_lookup_uses_supported_parselmouth_signature():
    class FakeFormant:
        def __init__(self):
            self.calls: list[tuple[int, float]] = []

        def get_value_at_time(self, formant_number: int, sample_time: float) -> float:
            self.calls.append((formant_number, sample_time))
            return 123.4

    fake_formant = FakeFormant()

    value = formant_helpers._value_at_time(fake_formant, 2, 0.05)

    assert value == 123.4
    assert fake_formant.calls == [(2, 0.05)]


def test_save_outputs_writes_audit_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = _base_config(tmp_path)
    _write_metadata(tmp_path, config)

    monkeypatch.setattr(
        "french_ids.praat_features.estimate_formant_ceilings",
        lambda metadata_df, audio_repository, config, output_path, recompute=False: pd.DataFrame(
            [{"speakerid": "M01", "vowel": "a", "formant_ceiling": 5500.0, "n_tokens": 1, "optimization_status": "fallback_too_few_tokens"}]
        ),
    )

    extractor = AcousticFeatureExtractor(config)
    monkeypatch.setattr(
        extractor.audio_repository,
        "get_segment_for_row",
        lambda row: (object(), tmp_path / "M01.wav", None),
    )
    monkeypatch.setattr(
        "french_ids.praat_features.compute_formant_features",
        lambda segment, ceiling_hz, config: (_success_formants(ceiling_hz), None),
    )
    monkeypatch.setattr(
        "french_ids.praat_features.compute_pitch_features",
        lambda segment, config: (_success_pitch(), None),
    )

    extractor.compute_all_features()
    extractor.save_outputs()

    log_df = pd.read_csv(config["paths"]["log_file"])
    assert list(log_df.columns) == [
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
    assert log_df.loc[0, "feature_status"] == "success"


def test_compute_acoustic_script_calls_extractor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script_path = Path("scripts/compute_acoustic_features.py")
    spec = importlib.util.spec_from_file_location("compute_acoustic_features", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths: {}\n", encoding="utf-8")

    called: dict[str, object] = {}

    monkeypatch.setattr(module, "load_config", lambda path: {"loaded_from": str(path), **_base_config(tmp_path)})

    class FakeExtractor:
        def __init__(self, config, config_path, recompute_ceilings, metadata_csv, output_csv):
            called["config_path"] = config_path
            called["recompute_ceilings"] = recompute_ceilings
            called["metadata_csv"] = metadata_csv
            called["output_csv"] = output_csv

        def compute_all_features(self):
            called["computed"] = True

        def save_outputs(self):
            called["saved"] = True

    monkeypatch.setattr(module, "AcousticFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(
        "sys.argv",
        [
            "compute_acoustic_features.py",
            "--config",
            str(config_path),
            "--recompute-ceilings",
            "--metadata-csv",
            str(tmp_path / "override_metadata.csv"),
            "--output-csv",
            str(tmp_path / "override_output.csv"),
        ],
    )

    module.main()

    assert called["recompute_ceilings"] is True
    assert called["computed"] is True
    assert called["saved"] is True
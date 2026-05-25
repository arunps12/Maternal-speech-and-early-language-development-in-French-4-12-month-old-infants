# Maternal Speech and Early Language Development in French 4–12-Month-Old Infants

This repository provides a reproducible pipeline for extracting vowel metadata
from Praat TextGrid files produced in a study of French maternal
infant-directed speech (IDS).  The pipeline reads paired .TextGrid and .wav
files, applies label corrections, and writes a structured CSV ready for
downstream acoustic analysis.

The study examines how acoustic characteristics of maternal IDS change as
infants grow from 4 to 12 months of age, with a focus on vowel acoustics,
variability, and distinctiveness, based on 107 recordings and 10 671 annotated
vowels.

---

## Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) — fast Python package and project manager

### Install uv

On macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, restart your terminal so the `uv` command is available.

---

## Setup

Clone the repository and install all dependencies into an isolated virtual
environment managed by uv:

```bash
git clone https://github.com/arunps12/Maternal-speech-and-early-language-development-in-French-4-12-month-old-infants.git
cd Maternal-speech-and-early-language-development-in-French-4-12-month-old-infants
uv sync
```

`uv sync` reads `pyproject.toml`, creates a `.venv/` virtual environment, and
installs `pandas`, `pyyaml`, and `praat-parselmouth` automatically.

---

## Configuration

Open `config/config.yaml` and set `input_folder` to the directory that contains
your .TextGrid and .wav files:

```yaml
paths:
  input_folder: "PATH_TO_TEXTGRID_AND_WAV_FOLDER"   # <-- edit this
  output_csv: "outputs/french_vowels_metadata.csv"
  skipped_labels_csv: "outputs/skipped_labels.csv"

filters:
  min_duration_ms: 30
  remove_registers:
    - "IDS(chant)"

features:
  pitch: false
  formants_mean: false
  formants_central_frame: false
```

All other values can be left as-is for the default metadata-only run.  The
`features` flags are reserved for a future acoustic extraction step; set them
to `false` to produce a metadata-only CSV with acoustic columns present but
empty.

---

## Running the pipeline

```bash
uv run python scripts/build_french_vowel_metadata.py --config config/config.yaml
```

The script prints a summary on completion:

```
=== Pipeline Summary ===
  TextGrid files found          : 107
  Files successfully processed  : 107
  Files skipped                 : 0
  Intervals read (non-empty)    : 11 204
  Removed (duration < 30 ms)    : 533
  Removed (IDS(chant))          : 0
  Skipped (parse errors)        : 0
  Final rows                    : 10 671
  Output CSV                    : outputs/french_vowels_metadata.csv
  Skipped labels CSV            : outputs/skipped_labels.csv
========================
```

---

## Output files

Both files are written to the `outputs/` directory.

### `french_vowels_metadata.csv`

One row per vowel interval.  Columns:

| Column | Description |
|--------|-------------|
| `speakerid` | Speaker identifier as found in the filename (e.g. `c012`) |
| `session` | Recording session (e.g. `4m`, `8m`, `12m`) |
| `activity` | Activity label from the filename (e.g. `bath`) |
| `time` | Time code from the filename (e.g. `1925`) |
| `word` | Word containing the vowel |
| `vowel` | Vowel category label |
| `register` | `IDS` or `ADS` |
| `start_sec` | Interval onset in seconds, rounded to 2 decimal places |
| `duration_sec` | Duration in seconds, rounded to 2 decimal places |
| `duration_ms` | Duration in milliseconds, rounded to 2 decimal places |
| `mean_pitch` to `central_F4` | Acoustic feature columns — empty (NaN) until features are enabled |

Full column order: `speakerid`, `session`, `activity`, `time`, `word`, `vowel`,
`register`, `start_sec`, `duration_sec`, `duration_ms`, `mean_pitch`,
`min_pitch`, `max_pitch`, `pitch_range`, `formant_ceiling`, `mean_F1`,
`mean_F2`, `mean_F3`, `mean_F4`, `central_F1`, `central_F2`, `central_F3`,
`central_F4`.

### `skipped_labels.csv`

Rows that could not be processed, with columns `file`, `raw_label`, `reason`.
Created only when there are skipped entries.

---

## Filename convention

TextGrid files must follow the pattern:

```
speakerid_session_activity_time.TextGrid
```

Example: `c012_8m_bath_1925.TextGrid` is parsed as:

| Field | Value |
|-------|-------|
| speakerid | c012 |
| session | 8m |
| activity | bath |
| time | 1925 |

---

## Filters applied

- Intervals shorter than `min_duration_ms` (default 30 ms) are removed.
- Rows where `register` is `IDS(chant)` are removed.
- Intervals with empty labels (silences) are skipped silently.
- Vowel `en` is preserved as-is and is not converted to `an`.

---

## Running the tests

```bash
uv run pytest tests/ -v
```

48 tests cover filename parsing, all 12 label corrections, register
normalisation, label parsing, `en` vowel preservation, and `IDS(chant)`
filtering.

---

## Project structure

```
config/
  config.yaml                     paths, filters, and feature flags

src/french_ids/
  __init__.py
  config.py                       YAML config loader
  filename_parser.py              speakerid_session_activity_time parser
  label_cleaning.py               label corrections and register normalisation
  textgrid_reader.py              parselmouth-based TextGrid reading
  build_metadata_csv.py           main pipeline logic and CLI entry
  praat_features.py               placeholder for future Praat extraction

scripts/
  build_french_vowel_metadata.py  command-line entry point

outputs/                          generated CSV files (directory tracked by git)

tests/
  test_filename_parser.py
  test_label_cleaning.py

pyproject.toml                    uv / hatchling packaging and pytest config
```

---

## Citation

If you use or build upon this work, please cite:

```bibtex
@misc{Maternal_French_Infant_Speech,
  author       = {Arun Prakash Singh},
  title        = {Maternal Speech and Early Language Development in French 4--12-Month-Old Infants},
  year         = {2025},
  howpublished = {\url{https://github.com/arunps12/Maternal-speech-and-early-language-development-in-French-4-12-month-old-infants}},
  note         = {GPL-3.0 License}
}
```

---

## Contact

**Arun Prakash Singh**
Department of Linguistics and Scandinavian Studies, University of Oslo
arunps@uio.no
[https://github.com/arunps12](https://github.com/arunps12)

---

**License:** [GNU GPL v3.0](LICENSE)

---

## About

I am Arun Prakash Singh, a Marie Curie Research Fellow at the University of Oslo (UiO).
My research focuses on speech technology, data engineering, and machine learning, with an
emphasis on building intelligent, data-driven systems that model human communication and
learning.  I am passionate about integrating AI, analytics, and large-scale data pipelines
to advance our understanding of how humans process and acquire language.
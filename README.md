# Maternal Speech and Early Language Development in French 4–12-Month-Old Infants

This repository contains the analysis pipeline, scripts, and results for the study **“Maternal speech and early language development in French 4–12-month-old infants.”**  
The project investigates how the acoustic characteristics of maternal *infant-directed speech* (IDS) change as infants grow from 4 to 12 months of age, focusing on vowel acoustics, variability, and distinctiveness.

---

## 🧠 Overview

Mothers adjust their speech acoustically when interacting with infants, which is thought to support phonetic learning.  
This project examines whether specific **acoustic measures**—such as pitch, pitch range, vowel duration, vowel space area, vowel variability, and vowel distinctiveness—systematically vary with **child age** in French IDS.

Analyses are based on **107 audio recordings** of French-speaking mothers addressing their infants at 4, 8, and 12 months.  
A total of **10 671 vowels** were annotated and analyzed.

---

## 🔧 Data Processing Pipeline

1. **Annotation**
   - TextGrid files were generated for each recording using *Praat*.
   - Vowel tiers were aligned manually and exported using the Python library `textgrid`.

2. **Feature Extraction**
   - Implemented with [`parselmouth`](https://github.com/YannickJadoul/Parselmouth).
   - Extracted features for each vowel:
     - Mean, minimum, and maximum **pitch (Hz)**
     - **Formants (F1, F2)** using Burg method with optimized formant ceilings
     - **Vowel duration (s)**  
   - All features were stored in structured DataFrames.

3. **Acoustic Measure Computation**  
   Implemented in [`acoustic_measures.py`](acoustic_measures.py):

   | Measure | Description | Unit | Function |
   |----------|--------------|------|-----------|
   | **Pitch** | Mean fundamental frequency converted to semitones above 10 Hz | semitones | `pitch_in_st()` |
   | **Pitch Range** | Max–min pitch difference | semitones | `range_in_st()` |
   | **Duration** | Vowel length | ms | `duration_in_ms()` |
   | **Vowel Space Area** | Area of polygon formed by mean F1–F2 values | Hz² | `vowel_space_expansion()` |
   | **Vowel Variability** | Elliptical area based on σF1×σF2 | Hz² | `vowel_variability()` |
   | **Vowel Distinctiveness** | Ratio of between-vowel to total variance in F1/F2 | unitless | `vowel_distinctiveness()` |

4. **Directory and Path Management**  
   Defined in [`path.py`](path.py) using utility functions from [`utils.py`](utils.py):
   - `create_dir()` ensures that output directories (`acoustic_measures/`, `StatPlots/`, etc.) exist.
   - `Hz_to_semitones()` converts raw pitch values for perceptual scaling.

---

## 📊 Statistical Analysis

Statistical modeling was carried out in **R (4.2.3)** using the packages `lme4`, `lmerTest`, `car`, and `boot`.

- **Linear Mixed-Effects Models (LMMs)** tested how each acoustic measure varied with infant age.
- **Fixed effects:** `AgeInDays (z-scaled)`, `SES`, `Gender`, and their interactions.
- **Random effects:** participant intercepts and random slopes (simplified when singular fits occurred).
- **Model comparison:** Likelihood-ratio test between full and null models.
- **Confidence intervals:** obtained via bootstrapping (1 000 iterations).
- **Collinearity diagnostics:** Variance Inflation Factors (VIF < 2).
- **Model validation:** residual inspection and DHARMa diagnostics.

Implementation and outputs are documented in [`stat_analyses.Rmd`](stat_analyses.Rmd) and rendered in [`stat_analyses.pdf`](stat_analyses.pdf).

---

## 📁 Repository Structure

```
Maternal-speech-and-early-language-development-in-French-4-12-month-old-infants/
│
├── Notebook/                         # R notebook and exploratory analyses
├── acoustic_measures.py              # Functions for vowel-based acoustic metrics
├── utils.py                          # Utility functions (directory creation, Hz→st conversion)
├── path.py                           # Path setup for saving analysis outputs
├── Require_functions_stat_analyses.R # Helper R functions for LMM fitting
├── stat_analyses.Rmd / .pdf          # Main R-based statistical analysis
├── LICENSE                           # GNU General Public License v3
└── README.md                         # (this file)
```

---

## 🧩 Dependencies

### Python
```bash
pip install numpy pandas scipy parselmouth textgrid soundfile matplotlib
```

### R
```r
install.packages(c("lme4", "lmerTest", "car", "boot", "merTools", "DHARMa", "glmmTMB"))
```

---

## 📈 Results Summary

- **Pitch and Vowel Space Area:** no significant change with age.  
- **Pitch Range and Duration:** significantly increased with age.  
- **Vowel Variability & Distinctiveness:** showed no systematic trend across months.  

These results suggest that while mothers modulate prosodic range as infants grow, vowel category structure remains relatively stable during the first year.

---
## 📜 Citation
If you use or build upon this work, please cite:

```bibtex
@misc{Maternal_French_Infant_Speech,
  author       = {Arun Prakash Singh},
  title        = {Maternal Speech and Early Language Development in French 4--12-Month-Old Infants},
  year         = {2025},
  howpublished = {\url{https://github.com/arunps12/Maternal-speech-and-early-language-development-in-French-4-12-month-old-infants}},
  note         = {GPL-3.0 License}
}
---
```
## 📬 Contact

**Arun Prakash Singh**  
Department of Linguistics and Scandinavian Studies, University of Oslo  
📧 arunps@uio.no  
🔗 [https://github.com/arunps12](https://github.com/arunps12)

---

**License:** [GNU GPL v3.0](LICENSE)

---

## 🔄 Reproducible Preprocessing Pipeline (New)

A clean, reproducible TextGrid-to-CSV preprocessing pipeline has been added to
the repository.  It reads all `.TextGrid` files from a configurable input
folder, extracts vowel-tier intervals, applies label corrections, and writes a
structured metadata CSV ready for downstream analysis.

> **Note on legacy scripts** — the original scripts (`acoustic_measures.py`,
> `path.py`, `plots.py`, `utils.py`) and the `Notebook/` directory are kept
> **unchanged** in the repository root.  They may later be moved to
> `scripts/legacy/` once the new pipeline has been fully validated.

### Quick start

```bash
# 1. Install dependencies with uv
uv sync

# 2. Set your data folder in config/config.yaml
#    (replace PATH_TO_TEXTGRID_AND_WAV_FOLDER with the real path)

# 3. Run the pipeline
uv run python scripts/build_french_vowel_metadata.py --config config/config.yaml
```

Outputs written to `outputs/`:

| File | Description |
|------|-------------|
| `french_vowels_metadata.csv` | One row per vowel interval; acoustic columns present but empty (`NaN`) until feature extraction is enabled |
| `skipped_labels.csv` | Files or labels that could not be processed, with reasons |

### Output CSV columns

`speakerid`, `session`, `activity`, `time`, `word`, `vowel`, `register`,
`start_sec`, `duration_sec`, `duration_ms`,
`mean_pitch`, `min_pitch`, `max_pitch`, `pitch_range`,
`formant_ceiling`, `mean_F1`, `mean_F2`, `mean_F3`, `mean_F4`,
`central_F1`, `central_F2`, `central_F3`, `central_F4`

### New project structure

```
config/
  config.yaml                    # paths, filters, feature flags

src/french_ids/
  __init__.py
  config.py                      # YAML config loader
  filename_parser.py             # speakerid_session_activity_time parser
  label_cleaning.py              # label corrections, register normalisation
  textgrid_reader.py             # parselmouth-based TextGrid reading
  build_metadata_csv.py          # main pipeline logic
  praat_features.py              # placeholder for future Praat extraction

scripts/
  build_french_vowel_metadata.py # CLI entry point

outputs/                         # generated CSV files (git-tracked directory)

tests/
  test_filename_parser.py
  test_label_cleaning.py

pyproject.toml                   # uv / hatchling packaging
```

### Running the tests

```bash
uv run pytest tests/
```

---

## 🌟 About Me

Hi there! I'm **Arun Prakash Singh**, a **Marie Curie Research Fellow at the University of Oslo (UiO)**.  
My research focuses on **speech technology, data engineering, and machine learning**, with an emphasis on building intelligent, data-driven systems that model human communication and learning.  
I am passionate about integrating **AI, analytics, and large-scale data pipelines** to advance our understanding of how humans process and acquire language.

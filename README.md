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

@misc{Maternal_French_Infant_Speech,
  author       = {Arun Prakash Singh},
  title        = {Maternal Speech and Early Language Development in French 4--12-Month-Old Infants},
  year         = {2025},
  howpublished = {\url{https://github.com/arunps12/Maternal-speech-and-early-language-development-in-French-4-12-month-old-infants}},
  note         = {GPL-3.0 License}
}
---

## 📬 Contact

**Arun Prakash Singh**  
Department of Linguistics and Scandinavian Studies, University of Oslo  
📧 arunps@uio.no  
🔗 [https://github.com/arunps12](https://github.com/arunps12)

---

**License:** [GNU GPL v3.0](LICENSE)

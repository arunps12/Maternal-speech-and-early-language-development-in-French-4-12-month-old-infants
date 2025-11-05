# Maternal Speech and Early Language Development in French 4‑12 Month‑Old Infants

This repository accompanies our research on how **maternal speech input** supports **early language development** in French‑learning infants (aged 4‑12 months).  
We provide data‑preparation scripts, analysis pipelines, metadata and derived datasets, plus reproducible code for exploration and modelling of child vocalizations and language outcomes.

---

## 🎯 Project Overview

### Research Aim  
To investigate how variations in the quantity and quality of maternal speech directed at infants aged 4‑12 months influence subsequent early language outcomes in French‑learning infants.

### Key Questions  
- How much maternal speech (in words, utterances, types) do infants hear in naturalistic settings?  
- Which acoustic‑prosodic features of maternal speech correlate with infant vocalizations or early lexical growth?  
- Can early infant vocal behaviours (babbling, canonical vocalizations) be predicted from maternal input metrics?

---

## 📂 Repository Structure

```
Maternal‑speech‑and‑early‑language‑development‑in‑French‑4‑12‑month‑old‑infants/
│
├── data/                          ← Raw and processed data folders (not all public)
│   ├── raw_audio/                 ← Long‑form recordings of infant‑caregiver interaction
│   ├── transcripts/               ← Annotation files, utterance boundaries
│   ├── metadata.csv               ← Study metadata: participant IDs, age, hearing status, etc.
│   ├── derived_features/          ← Derived acoustic & prosodic features
│   └── infant_outcomes.csv        ← Infant language outcome variables (e.g., vocabulary size)
│
├── scripts/                       ← Preprocessing and feature‑extraction scripts
│   ├── 01_extract_maternal_input.py       ← Extract maternal speech metrics from transcripts/audio
│   ├── 02_extract_infant_vocalisations.py ← Detect infant vocalisations from recordings
│   ├── 03_compute_acoustic_features.py    ← Compute prosodic/acoustic features of maternal & infant speech
│
├── notebooks/                     ← Jupyter notebooks for exploratory analysis and modelling
│   ├── EDA_maternal_input.ipynb
│   ├── EDA_infant_vocalisations.ipynb
│   └── Modelling_language_outcome.ipynb
│
├── results/                       ← Output from analyses (figures, tables)
│
├── README.md                      ← This file
├── requirements.txt               ← Python dependencies for reproducibility
└── LICENSE                        ← Open‑source licence
```

---

## 🛠️ Setup & Dependencies

### Recommended: Virtual environment  
Create an isolated environment before installing dependencies:

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Then install dependencies
pip install -r requirements.txt
```

### Sample `requirements.txt`  
```txt
numpy
pandas
scipy
librosa
torchaudio
matplotlib
seaborn
scikit‑learn
jupyter
```

---

## 🔍 Data Preparation Workflow

### 1. Extract maternal speech metrics  
```bash
python scripts/01_extract_maternal_input.py \
  --audio_dir data/raw_audio/ \
  --transcripts_dir data/transcripts/ \
  --output_csv data/derived_features/maternal_input_metrics.csv
```

### 2. Extract infant vocalisations  
```bash
python scripts/02_extract_infant_vocalisations.py \
  --audio_dir data/raw_audio/ \
  --output_csv data/derived_features/infant_vocalisations.csv
```

### 3. Compute acoustic/prosodic features  
```bash
python scripts/03_compute_acoustic_features.py \
  --input_metrics data/derived_features/maternal_input_metrics.csv \
  --output_features data/derived_features/acoustic_features.csv
```

### 4. Merge with infant outcomes  
Use `metadata.csv` and `infant_outcomes.csv` to merge predictors and outcomes for modelling.

---

## 📊 Analysis & Modelling

Explore data and fit statistical or machine‑learning models using notebooks in `notebooks/`.  
Typical analyses include:

- Relationship between maternal speech quantity (utterances/hour) and infant canonical babbling rate.  
- Prosodic features (pitch, rhythm) of maternal speech predicting infant vocal output.  
- Regression or classification models: early infant vocalisations → 12‑month vocabulary size.

---

## 🧮 Reproducibility & Results  
The `results/` folder contains:

- Figures (PNG/PDF) of key findings  
- Tables summarising model coefficients  
- Model performance metrics (R², accuracy, etc.)

Feel free to regenerate these by running the notebooks after completing data preparation.

---

## 🧾 Citation  
If you use this dataset or pipeline in your research, please cite:

```
Author(s). (2025). Maternal Speech and Early Language Development in French 4‑12 Month‑Old Infants [Data set and code]. GitHub repository. https://github.com/arunps12/Maternal‑speech‑and‑early‑language‑development‑in‑French‑4‑12‑month‑old‑infants
```

---

## 📬 Contact  
**Arun Singh**  
Affiliation: University of Oslo, Norway  
Email: arunps@uio.no  
GitHub: https://github.com/arunps12  
Project repo: https://github.com/arunps12/Maternal‑speech‑and‑early‑language‑development‑in‑French‑4‑12‑month‑old‑infants


Thank you for exploring this research project! Feel free to open issues or pull requests if you’d like to contribute or reuse code/data.

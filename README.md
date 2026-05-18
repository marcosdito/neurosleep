# neurosleep
"EEG sleep stage classification with deep learning"

# NeuroSleep

**Automated sleep stage classification from EEG signals using deep learning.**

A complete machine learning pipeline that processes polysomnographic recordings from the Sleep-EDF dataset, engineers physiologically-motivated features, trains both classical and deep learning models, and persists experiments to PostgreSQL for reproducibility.

---

## Why this project

Sleep stage classification is a clinically relevant task usually performed by trained technicians who manually score 30-second windows of EEG into 5 stages (W, N1, N2, N3, REM) — hours of work per patient. This project builds an end-to-end automated pipeline as a study in applied biosignal machine learning, with a focus on **methodological rigor**: subject-aware splitting, multiple metrics for imbalanced data, and honest comparison of classical features against deep learning.

## Technical highlights

- **Signal processing pipeline** built on MNE-Python: bandpass filtering, epoching, robust per-recording normalization.
- **Feature engineering in three domains**: time (Hjorth parameters, RMS, zero-crossings, moments), frequency (relative band power for δ/θ/α/β/γ, spectral entropy, spectral edge), and wavelet (Daubechies-4 multi-level energy).
- **Classical baselines**: Random Forest and SVM (RBF) trained on engineered features.
- **Deep learning model**: 1D CNN in TensorFlow/Keras trained directly on raw signals.
- **Subject-aware cross-validation**: no patient appears in both training and test sets.
- **Imbalance-aware evaluation**: macro-F1, balanced accuracy, and Cohen's kappa rather than plain accuracy.
- **Experiment persistence**: results stored in PostgreSQL with a normalized schema (`experiments`, `metrics`, `predictions`).
- **Full reproducibility**: containerized with Docker and docker-compose.

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Signal processing | MNE-Python, SciPy, PyWavelets |
| Classical ML | Scikit-learn |
| Deep learning | TensorFlow / Keras |
| Data | NumPy, Pandas, PyArrow (Parquet) |
| Database | PostgreSQL 16, SQLAlchemy |
| Visualization | Matplotlib, Seaborn |
| Orchestration | Docker, Docker Compose |
| Notebooks | Jupyter |

## Architecture

```
┌─────────────────────┐
│  Sleep-EDF dataset  │  (PhysioNet, fetched via MNE)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│  Preprocessing pipeline                          │
│  - Bandpass filter (0.3 - 35 Hz)                 │
│  - 30-s epoching (AASM standard)                 │
│  - Robust normalization (median / IQR)           │
└──────────┬──────────────────────────┬───────────┘
           │                          │
           ▼                          ▼
┌──────────────────────┐    ┌─────────────────────┐
│  Feature engineering │    │  Raw normalized     │
│  - Time domain       │    │  signals (.npz)     │
│  - Frequency domain  │    │                     │
│  - Wavelet domain    │    │                     │
└──────────┬───────────┘    └──────────┬──────────┘
           │                           │
           ▼                           ▼
┌──────────────────────┐    ┌─────────────────────┐
│  Classical models    │    │  1D CNN (Keras)     │
│  - Random Forest     │    │                     │
│  - SVM (RBF)         │    │                     │
└──────────┬───────────┘    └──────────┬──────────┘
           │                           │
           └─────────────┬─────────────┘
                         ▼
              ┌────────────────────┐
              │  PostgreSQL        │
              │  - experiments     │
              │  - metrics         │
              │  - predictions     │
              └────────────────────┘
```

## Project structure

```
neurosleep/
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA on raw Sleep-EDF data
│   ├── 02_preprocessing_features.ipynb # Preprocessing + feature engineering
│   ├── 03_baselines.ipynb              # Random Forest + SVM baselines
│   └── 04_cnn_keras.ipynb              # 1D CNN in TensorFlow/Keras
├── src/
│   └── db/
│       ├── models.py                   # SQLAlchemy ORM models
│       └── session.py                  # Engine and session management
├── scripts/
│   └── save_to_db.py                   # Persist results into PostgreSQL
├── data/
│   ├── raw/                            # MNE-cached PSG recordings
│   ├── processed/                      # features.parquet, epochs.npz
│   └── results/                        # JSON metrics, prediction parquets
├── models/                             # Saved Keras checkpoints
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick start

### Option A — Local Python environment

```bash
# 1. Clone the repo and enter it
git clone https://github.com/marcosdito/neurosleep.git
cd neurosleep

# 2. Create a virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start PostgreSQL in a container
docker run -d --name neurosleep-pg \
    -e POSTGRES_USER=neurosleep \
    -e POSTGRES_PASSWORD=neurosleep \
    -e POSTGRES_DB=neurosleep \
    -p 5432:5432 \
    postgres:16

# 5. Run the notebooks in order
jupyter notebook
# -> Open and execute 01, 02, 03, 04 in sequence

# 6. Persist results to PostgreSQL
python scripts/save_to_db.py
```

### Option B — Full Docker

```bash
# Start Postgres and run the pipeline in a container
docker compose up --build
```

## Notebook walkthrough

### 01 — Data exploration
Downloads a single Sleep-EDF recording via MNE, inspects channel layout and sampling rate, visualizes raw signals across EEG/EOG channels, parses the expert-annotated hypnogram, segments the recording into 30-second epochs, and characterizes class imbalance.

### 02 — Preprocessing and feature engineering
Builds the full multi-subject preprocessing pipeline: bandpass filtering, epoching, robust normalization. Extracts time, frequency, and wavelet features for every epoch and persists them as Parquet for the modeling notebooks. Also saves the raw normalized signals as a compressed NumPy archive for the deep learning notebook.

### 03 — Classical baselines
Implements **subject-aware** train/test splitting via `GroupShuffleSplit` to prevent data leakage. Trains Random Forest and SVM with balanced class weights, evaluates with metrics appropriate for imbalanced data (macro-F1, balanced accuracy, Cohen's kappa), and visualizes feature importances and confusion matrices. Cross-validates the Random Forest using `StratifiedGroupKFold` to get an honest variance estimate.

### 04 — Deep learning with TensorFlow/Keras
Trains a 1D Convolutional Neural Network on raw normalized signals. The architecture consists of three Conv1D + BatchNorm + MaxPool + Dropout blocks (32 → 64 → 128 filters), followed by global average pooling and a dense classifier. Uses the same subject-aware split as the baselines for a fair comparison. Standard training discipline: early stopping, learning rate reduction on plateau, class weights, custom macro-F1 logging callback.

## Database schema

Three normalized tables capture the full experiment lifecycle:

| Table | Purpose |
|---|---|
| `experiments` | One row per modeling run. Stores model name, dataset, sample counts, train/test subject IDs, and a JSONB hyperparameters field. |
| `metrics` | Long-format (one row per metric value). Makes cross-model comparison trivial via SQL aggregation. |
| `predictions` | Per-epoch predictions on the test set. Enables per-subject error analysis. |

Example queries:

```sql
-- Best model by macro-F1
SELECT e.model_name, m.value
FROM experiments e
JOIN metrics m ON m.experiment_id = e.id
WHERE m.name = 'macro_f1'
ORDER BY m.value DESC;

-- Confusion-like analysis: which subjects are hardest?
SELECT subject_id,
       COUNT(*) FILTER (WHERE true_label = predicted_label) * 1.0 / COUNT(*) AS accuracy
FROM predictions
GROUP BY subject_id
ORDER BY accuracy ASC;
```

## Design decisions

**Subject-aware splitting.** Standard random splits cause severe data leakage in EEG: the same patient's epochs end up in both train and test sets, and models learn patient-specific patterns instead of stage-specific ones. Test metrics become inflated and fail to generalize. We use grouped splits at every step (train/val/test in Notebook 04, K-fold cross-validation in Notebook 03).

**Macro-F1 over accuracy.** The dataset is class-imbalanced (N2 dominates, N1 and REM are rare). Plain accuracy is biased toward the majority class. Macro-F1 averages per-class F1 equally, surfacing model performance on minority classes.

**Robust normalization over z-score.** EEG signals contain occasional high-amplitude artifacts that destabilize mean/std-based normalization. Median / IQR normalization is robust to these outliers.

**Three feature families.** Time-domain features capture statistical and morphological properties. Frequency-domain features capture spectral composition, which is the most informative dimension for sleep staging. Wavelet features capture transient events (sleep spindles, K-complexes) that are localized in both time and frequency.

**JSONB hyperparameters.** Different models have different hyperparameter sets. Storing them as JSONB rather than dedicated columns keeps the schema stable across model types while preserving query capability via PostgreSQL's JSON operators.

## Honest limitations

- **Epoch independence assumption.** We classify each 30-second window in isolation. Real sleep scoring uses temporal context — neighboring epochs constrain plausible transitions. Adding a sequence model (LSTM, Transformer) on top of CNN features is a known way to add 3-8 macro-F1 points.
- **Small subject pool.** This is a study project; the pipeline is run on a small subset of Sleep-EDF subjects. Increasing the subject count is the simplest path to better generalization.
- **No hyperparameter search.** Architectures and hyperparameters were chosen by reasoning, not by random or Bayesian search. A proper study would include Optuna or similar.
- **No artifact handling beyond filtering.** Independent component analysis (ICA) for ocular and muscle artifact removal is standard in clinical EEG but not applied here.

## Future work

- Sequence modeling: stack a bidirectional LSTM or small Transformer over per-epoch CNN features.
- Cross-dataset validation on additional sleep datasets (SHHS, MASS).
- Hyperparameter optimization with Optuna.
- Active artifact handling via ICA.
- Multi-channel modeling treating each EEG derivation as a separate input branch.

## License

MIT — see `LICENSE`.

## Author

**Marcos Vinícius Rocha Gomes**
- Undergraduate researcher at Instituto Santos Dumont (ISD)
- Bachelor of IT, Universidade Federal do Rio Grande do Norte (UFRN)
- GitHub: [@marcosdito](https://github.com/marcosdito)s
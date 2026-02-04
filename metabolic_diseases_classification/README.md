# Metabolic Diseases Prediction Module

Identifies oral microbiome features associated with metabolic diseases and trains prediction models (case vs healthy controls).

## Run order

1. **`generate_fold_splits.py`** — Saves canonical 5-fold train/test split (`fold_splits.csv`).
2. **`select_diseases_associated_oral_features.py`** — Top20 phenotypes and significant markers per fold (training set only).
3. **`classification.py`** — 5-fold CV with the same fold split.
4. **`predict_baseline.py`** — Baseline predictor (age, sex, BMI only) using the same fold splits for comparison.

Update paths and `LAYER` (pathway/strain) in each script as needed; keep `LAYER` the same across all three.

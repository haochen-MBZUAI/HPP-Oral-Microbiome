#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline predictor using age, sex, and BMI only.

Uses the same cohort and 5-fold splits as classification.py (via fold_splits.csv).
Run generate_fold_splits.py first, then this script.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from lightgbm import LGBMClassifier

# --- Configuration (paths and labels must match generate_fold_splits.py / classification.py) ---
PHENOTYPE_FILE = 'home/ec2-user/Studies/Oral_HPP/oral_data/phenotype.csv'
LABELS_FILE = 'home/ec2-user/Studies/Oral_HPP/oral_data/metabolic_diseases.csv'
# Need feature file only for collection_date to compute time-window case ids
FEATURES_FILE = "home/ec2-user/Studies/Oral_HPP/oral_data/pathway_processed.csv"
FOLD_SPLITS_FILE = 'home/ec2-user/Studies/Oral_HPP/oral_data/hypertension/fold_splits.csv'

TARGET_DISEASE = 'hypertension'
HEALTHY_CONTROL_LABEL = 'no medical diagnoses'
TIME_WINDOW_DAYS = 180
N_SPLITS = 5
RANDOM_STATE = 42

# Baseline features: use these column names from phenotype.csv (if age is 'age_at_collection', set below)
BASELINE_COLS = ['age', 'sex', 'bmi']
# If your phenotype uses 'age_at_collection' instead of 'age', set this and we use it for 'age'
AGE_COL_ALT = 'age_at_collection'


def _resolve_age_col(df):
    if 'age' in df.columns:
        return 'age'
    if AGE_COL_ALT and AGE_COL_ALT in df.columns:
        return AGE_COL_ALT
    raise ValueError("Phenotype must contain 'age' or 'age_at_collection'.")


def _encode_sex(ser):
    if ser.dtype == object or ser.dtype.name == 'category' or not np.issubdtype(ser.dtype, np.number):
        uniq = ser.dropna().unique().tolist()
        mapping = {}
        for i, v in enumerate(sorted(str(x) for x in uniq)):
            mapping[v] = i
        return ser.astype(str).map(mapping)
    return ser


def main():
    print("--- Baseline predictor (age, sex, BMI) using canonical fold splits ---")

    # 1) Load phenotype and restrict to baseline columns (with age column name resolved)
    if not os.path.exists(PHENOTYPE_FILE):
        print(f"ERROR: Phenotype file not found: {PHENOTYPE_FILE}")
        exit(1)
    pheno = pd.read_csv(PHENOTYPE_FILE)
    age_col = _resolve_age_col(pheno)
    use_cols = [c for c in BASELINE_COLS if c != 'age'] + [age_col]
    missing = [c for c in use_cols if c not in pheno.columns]
    if missing:
        print(f"ERROR: Phenotype missing columns: {missing}. Available: {list(pheno.columns)}")
        exit(1)
    pheno = pheno[['participant_id'] + use_cols].copy()
    pheno = pheno.rename(columns={age_col: 'age'})
    # Deduplicate by participant_id
    pheno = pheno.groupby('participant_id').sample(n=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # 2) Load fold splits and restrict to this cohort
    if not os.path.exists(FOLD_SPLITS_FILE):
        print(f"ERROR: Fold splits not found: {FOLD_SPLITS_FILE}. Run generate_fold_splits.py first.")
        exit(1)
    fold_splits = pd.read_csv(FOLD_SPLITS_FILE)
    cohort_ids = fold_splits['participant_id'].unique()
    pheno = pheno[pheno['participant_id'].isin(cohort_ids)].copy()
    if len(pheno) == 0:
        print("ERROR: No phenotype rows left after restricting to fold_splits cohort.")
        exit(1)

    # 3) Compute labels (same logic as generate_fold_splits: case = hypertension + time window)
    df_labels = pd.read_csv(LABELS_FILE, parse_dates=['collection_date'])
    df_features = pd.read_csv(FEATURES_FILE, parse_dates=['collection_date'], index_col=0)
    case_participants_ids = set(df_labels[df_labels['medical_condition'] == TARGET_DISEASE]['participant_id'])
    healthy_participants_ids = set(df_labels[df_labels['medical_condition'] == HEALTHY_CONTROL_LABEL]['participant_id'])
    clean_healthy_ids = healthy_participants_ids - case_participants_ids
    df_case_labels = df_labels[df_labels['participant_id'].isin(case_participants_ids) & (df_labels['medical_condition'] == TARGET_DISEASE)]
    df_merged_case = pd.merge(df_features['collection_date'].reset_index(), df_case_labels, on='participant_id')
    df_merged_case['date_diff'] = (df_merged_case['collection_date_x'] - df_merged_case['collection_date_y']).dt.days.abs()
    is_valid_case = df_merged_case['date_diff'] <= TIME_WINDOW_DAYS
    final_case_ids = set(df_merged_case[is_valid_case]['participant_id'])
    pheno['label'] = pheno['participant_id'].isin(final_case_ids).astype(int)

    # 4) Build feature matrix: age, sex, bmi (encode sex if categorical)
    X_df = pheno[['participant_id', 'age', 'sex', 'bmi']].copy()
    X_df['sex'] = _encode_sex(X_df['sex'])
    for col in ['age', 'sex', 'bmi']:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
    y = pheno['label'].values
    participant_ids = pheno['participant_id'].values
    X = X_df[['age', 'sex', 'bmi']].values

    # Row order: same as pheno (participant_id order); we'll index by position when using fold_splits
    pid_to_row = {pid: i for i, pid in enumerate(participant_ids)}

    acc_list, auc_list = [], []
    for fold_idx in range(1, N_SPLITS + 1):
        train_ids = set(fold_splits[(fold_splits['fold'] == fold_idx) & (fold_splits['role'] == 'train')]['participant_id'])
        test_ids = set(fold_splits[(fold_splits['fold'] == fold_idx) & (fold_splits['role'] == 'test')]['participant_id'])
        train_idx = [pid_to_row[pid] for pid in train_ids if pid in pid_to_row]
        test_idx = [pid_to_row[pid] for pid in test_ids if pid in pid_to_row]
        if not train_idx or not test_idx:
            print(f"WARNING: Fold {fold_idx} has empty train or test. Skipping.")
            continue

        X_train = X[train_idx].copy()
        X_test = X[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Impute missing with training median (per column)
        medians = np.nanmedian(X_train, axis=0)
        for j in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, j]), j] = medians[j]
            X_test[np.isnan(X_test[:, j]), j] = medians[j]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1, verbosity=-1)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_pred_proba = model.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            auc = float('nan')
        acc_list.append(acc)
        auc_list.append(auc)

        print(f"\n===== Fold {fold_idx}/{N_SPLITS} (baseline: age, sex, BMI) =====")
        print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}" if not np.isnan(auc) else f"Accuracy: {acc:.4f}, AUC: NaN")
        print(classification_report(y_test, y_pred, target_names=['Control', 'Case']))
        print(confusion_matrix(y_test, y_pred))

    print("\n=== Baseline (age, sex, BMI) CV Summary ===")
    if acc_list:
        print(f"Accuracy (mean ± std): {np.nanmean(acc_list):.4f} ± {np.nanstd(acc_list):.4f}")
    if auc_list:
        print(f"AUC (mean ± std): {np.nanmean(auc_list):.4f} ± {np.nanstd(auc_list):.4f}")
    print("Done.")


if __name__ == '__main__':
    main()

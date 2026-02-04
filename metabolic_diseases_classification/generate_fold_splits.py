#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate canonical 5-fold train/test split for metabolic disease classification.

Builds the same cohort as classification.py (case + healthy controls, time-window,
undersampling) and saves (participant_id, fold, role) so that:
- select_diseases_associated_oral_features.py uses the same train set per fold for
  top phenotypes and marker selection.
- classification.py uses the same train/test indices for evaluation.

Run this script first, then select_diseases_associated_oral_features.py, then classification.py.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold

# --- Must match classification.py (same LAYER and FEATURES_FILE) ---
LAYER = 'pathway'  # 'pathway' or 'strain'; must match classification.py
FEATURES_FILE = (
    "home/ec2-user/Studies/Oral_HPP/oral_data/pathway_processed.csv"
    if LAYER == 'pathway' else
    "home/ec2-user/Studies/Oral_HPP/oral_data/strain_processed.csv"
)
LABELS_FILE = 'home/ec2-user/Studies/Oral_HPP/oral_data/metabolic_diseases.csv'
TARGET_DISEASE = 'hypertension'
HEALTHY_CONTROL_LABEL = 'no medical diagnoses'
TIME_WINDOW_DAYS = 180
N_SPLITS = 5
RANDOM_STATE = 42
NON_FEATURE_COLS = [
    TARGET_DISEASE, 'collection_date', 'cohort',
    'research_stage', 'array_index', 'participant_id'
]

# Output: same directory as other disease outputs
FOLD_SPLITS_OUTPUT = 'home/ec2-user/Studies/Oral_HPP/oral_data/hypertension/fold_splits.csv'

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("--- Generating canonical fold splits (same cohort as classification.py) ---")

    df_features = pd.read_csv(FEATURES_FILE, parse_dates=['collection_date'], index_col=0)
    df_labels_raw = pd.read_csv(LABELS_FILE, parse_dates=['collection_date'])

    case_participants_ids = set(df_labels_raw[df_labels_raw['medical_condition'] == TARGET_DISEASE]['participant_id'])
    healthy_participants_ids = set(df_labels_raw[df_labels_raw['medical_condition'] == HEALTHY_CONTROL_LABEL]['participant_id'])
    clean_healthy_ids = healthy_participants_ids - case_participants_ids

    df_case_labels = df_labels_raw[df_labels_raw['participant_id'].isin(case_participants_ids) & (df_labels_raw['medical_condition'] == TARGET_DISEASE)]
    df_merged_case = pd.merge(df_features['collection_date'].reset_index(), df_case_labels, on='participant_id')
    df_merged_case['date_diff'] = (df_merged_case['collection_date_x'] - df_merged_case['collection_date_y']).dt.days.abs()
    is_valid_case = df_merged_case['date_diff'] <= TIME_WINDOW_DAYS
    final_case_ids = set(df_merged_case[is_valid_case]['participant_id'])

    final_participant_ids = final_case_ids.union(clean_healthy_ids)
    df_model_data = df_features[df_features.index.isin(final_participant_ids)].copy()
    df_model_data[TARGET_DISEASE] = df_model_data.index.isin(final_case_ids).astype(int)

    df_cases = df_model_data[df_model_data[TARGET_DISEASE] == 1]
    df_controls = df_model_data[df_model_data[TARGET_DISEASE] == 0]
    n_controls_to_keep = len(df_cases)
    if n_controls_to_keep > 0 and n_controls_to_keep <= len(df_controls):
        df_controls_undersampled = df_controls.sample(n=n_controls_to_keep, random_state=RANDOM_STATE)
    else:
        df_controls_undersampled = df_controls.head(0) if n_controls_to_keep == 0 else df_controls

    df_temp_combined = pd.concat([df_cases, df_controls_undersampled])
    df_temp_combined['participant_id'] = df_temp_combined.index
    df_model_data = df_temp_combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    X = df_model_data.drop(columns=[c for c in NON_FEATURE_COLS if c in df_model_data.columns], errors='ignore').select_dtypes(include=np.number)
    y = df_model_data[TARGET_DISEASE]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        for i in train_idx:
            rows.append({'participant_id': df_model_data['participant_id'].iloc[i], 'fold': fold_idx, 'role': 'train'})
        for i in test_idx:
            rows.append({'participant_id': df_model_data['participant_id'].iloc[i], 'fold': fold_idx, 'role': 'test'})

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(FOLD_SPLITS_OUTPUT), exist_ok=True)
    out_df.to_csv(FOLD_SPLITS_OUTPUT, index=False)
    print(f"Saved fold splits to {FOLD_SPLITS_OUTPUT} (n={len(out_df)} rows, {len(df_model_data)} participants).")
    print("Run select_diseases_associated_oral_features.py then classification.py.")

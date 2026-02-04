import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import SelectKBest, f_classif

# --- 1. Configuration: File Paths and Column Names ---
# NOTE: Please verify and update all placeholder paths and column names below.

# --- Feature layer: 'pathway' or 'strain' ---
LAYER = 'pathway'  # set to 'strain' for strain-level oral markers

# --- Input Files ---
PHENOTYPE_DATA_PATH = "home/ec2-user/Studies/Oral_HPP/oral_data/phenotype_cleaned.csv"
DISEASE_DATA_PATH = 'home/ec2-user/Studies/Oral_HPP/oral_data/metabolic_diseases.csv'
REGRESSION_RESULTS_FILE = (
    "home/ec2-user/Studies/Oral_HPP/oral_data/regression_result/pathway_phenotype_regression_results_corrected.csv"
    if LAYER == 'pathway' else
    "home/ec2-user/Studies/Oral_HPP/oral_data/regression_result/strain_phenotype_regression_results_corrected.csv"
)

# --- Output Files ---
#  To predict other diseases, change the hypertension to other diseases
STAT_IMPORTANCE_OUTPUT_FILE = 'home/ec2-user/Studies/Oral_HPP/oral_data/hypertension/hypertension_pheno_statistic.csv'
SIGNIFICANT_PATHWAYS_OUTPUT_FILE = (
    'home/ec2-user/Studies/Oral_HPP/oral_data/hypertension/significant_pathways_from_statistic.csv'
    if LAYER == 'pathway' else
    'home/ec2-user/Studies/Oral_HPP/oral_data/hypertension/significant_strains_from_statistic.csv'
)

# --- Column names: regression results and output CSV ---
COL_REGRESSION_FEATURE = 'Pathway_Feature' if LAYER == 'pathway' else 'Strain_Feature'
COL_REGRESSION_PHENOTYPE = 'Phenotype_Feature'
COL_REGRESSION_PVALUE = 'p_corrected_bonferroni'
COL_SIGNIFICANT_OUTPUT = 'Significant_Pathway' if LAYER == 'pathway' else 'Significant_Strain'

# NEW: CV 配置
N_SPLITS = 5
RANDOM_STATE = 42

# --- Case vs control (must match classification.py / Methods: controls from healthy pool only) ---
TARGET_DISEASE = 'Hypertension'   # or 'Diabetes', 'Fatty liver', etc.; used with str.contains
HEALTHY_CONTROL_LABEL = 'no medical diagnoses'  # controls = healthy pool only (Methods)

# --- Canonical fold split (run generate_fold_splits.py first so fold aligns with classification.py) ---
FOLD_SPLITS_FILE = 'home/ec2-user/Studies/Oral_HPP/oral_data/hypertension/fold_splits.csv'

# ==============================================================================
# SCRIPT EXECUTION STARTS HERE
# ==============================================================================

# --- Part 1: Calculate Phenotype Importance (Case vs Healthy Controls Only) ---

print(f"--- Part 1: Calculating Phenotype Importance for {TARGET_DISEASE} (case vs healthy controls only) ---")

# --- Step 1.1: Load Data Files ---
print("\nStep 1.1: Loading data files...")
try:
    phenotype_df = pd.read_csv(PHENOTYPE_DATA_PATH)
    disease_df = pd.read_csv(DISEASE_DATA_PATH)
    print(f"Successfully loaded phenotype data. Initial shape: {phenotype_df.shape}")
except FileNotFoundError as e:
    print(f"\nERROR: Could not find a data file. Please check your paths. Details: {e}")
    exit()

# --- Step 1.2: Handle Duplicate Participants by Random Sampling ---
print("\nStep 1.2: Handling duplicate entries by random sampling...")
phenotype_df = phenotype_df.groupby('participant_id').sample(n=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Data shape after de-duplication: {phenotype_df.shape}")

# --- Step 1.3: Define case and control (Methods: controls from healthy pool only) ---
print(f"\nStep 1.3: Defining case ({TARGET_DISEASE}) vs control (healthy pool: '{HEALTHY_CONTROL_LABEL}')...")
case_ids = set(disease_df[disease_df['medical_condition'].str.contains(TARGET_DISEASE, case=False, na=False)]['participant_id'].unique())
healthy_ids = set(disease_df[disease_df['medical_condition'] == HEALTHY_CONTROL_LABEL]['participant_id'].unique())
clean_healthy_ids = healthy_ids - case_ids  # exclude anyone who is also case
participant_ids_keep = case_ids | clean_healthy_ids
phenotype_df = phenotype_df[phenotype_df['participant_id'].isin(participant_ids_keep)].copy()
phenotype_df['target'] = phenotype_df['participant_id'].isin(case_ids).astype(int)
print(f"Case (n={len(case_ids)}), healthy controls (n={len(clean_healthy_ids)}); analysis sample n={len(phenotype_df)}.")
print("Distribution of target (1=case, 0=healthy control):")
print(phenotype_df['target'].value_counts())

# --- Step 1.4: Restrict to canonical cohort (same as classification.py) so fold indices align ---
print("\nStep 1.4: Loading canonical fold splits (run generate_fold_splits.py first)...")
if not os.path.exists(FOLD_SPLITS_FILE):
    print(f"ERROR: Fold splits file not found: {FOLD_SPLITS_FILE}")
    print("Run generate_fold_splits.py first to create the same train/test split used by classification.py.")
    exit(1)
fold_splits = pd.read_csv(FOLD_SPLITS_FILE)
cohort_ids = fold_splits['participant_id'].unique()
phenotype_df = phenotype_df[phenotype_df['participant_id'].isin(cohort_ids)].copy()
print(f"Restricted to canonical cohort: n={len(phenotype_df)} participants (same as classification).")

non_feature_cols = [
    'participant_id', 'collection_date', 'collection_date_x', 'collection_date_y',
    'cohort', 'research_stage', 'array_index', 'id',
    'target'
]
feature_cols = phenotype_df.drop(columns=non_feature_cols, errors='ignore').select_dtypes(include=np.number).columns.tolist()

# --- Step 1.5: Per-fold: use TRAINING participants only (same fold as classification) ---
print("\nStep 1.5: Calculating feature relevance on TRAINING fold only (fold-aligned with classification)...")
fold_top_phenotypes = []
all_stat_files = []  # list of (fold_idx, path) so concatenated summary has correct fold numbers

for fold_idx in range(1, N_SPLITS + 1):
    train_ids = fold_splits[(fold_splits['fold'] == fold_idx) & (fold_splits['role'] == 'train')]['participant_id'].unique()
    phenotype_train = phenotype_df[phenotype_df['participant_id'].isin(train_ids)].copy()
    if phenotype_train.empty or phenotype_train['target'].nunique() < 2:
        print(f"[Fold {fold_idx}] WARNING: No or single-class training data; skipping.")
        fold_top_phenotypes.append([])
        continue

    X_train = phenotype_train[feature_cols].copy()
    y_train = phenotype_train['target']
    X_train = X_train.fillna(X_train.median())

    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X_train, y_train)

    feature_scores_df = pd.DataFrame({
        'Feature': feature_cols,
        'Score': selector.scores_,
        'P_value': selector.pvalues_
    }).sort_values(by=['Score', 'P_value'], ascending=[False, True]).reset_index(drop=True)

    stat_out_file_fold = STAT_IMPORTANCE_OUTPUT_FILE.replace('.csv', f'_fold{fold_idx}.csv')
    feature_scores_df.to_csv(stat_out_file_fold, index=False)
    all_stat_files.append((fold_idx, stat_out_file_fold))
    print(f"[Fold {fold_idx}/{N_SPLITS}] Fitted on n={len(phenotype_train)} train participants; saved to {stat_out_file_fold}")
    print("--- Top 20 Features (training-only) ---")
    print(feature_scores_df.head(20).to_string())
    fold_top_phenotypes.append(feature_scores_df['Feature'].head(20).tolist())

# 为了兼容原路径，也输出一个汇总版（将每折结果叠加，带 fold 列；用实际 fold 编号）
try:
    concatenated = []
    for (fold_idx, path) in all_stat_files:
        tmp = pd.read_csv(path)
        tmp.insert(0, 'fold', fold_idx)
        concatenated.append(tmp)
    if concatenated:
        pd.concat(concatenated, ignore_index=True).to_csv(STAT_IMPORTANCE_OUTPUT_FILE, index=False)
        print(f"\n[Summary] Concatenated per-fold stats saved to: {STAT_IMPORTANCE_OUTPUT_FILE}")
except Exception as e:
    print(f"WARNING: Failed to write concatenated stats: {e}")

# --- Part 2: Find Significant Oral Markers (Pathway or Strain) Linked to Top Phenotypes ---
# Marker selection uses per-fold regression results (training fold only), not full-data results.

print(f"\n{'='*60}\n--- Part 2: Finding Significant {LAYER.capitalize()} Markers from Top Phenotypes ---\n{'='*60}")

# --- Step 2.1 & 2.2: Per fold: load that fold's regression results and filter significant markers ---
print("\nStep 2.1/2.2: Loading per-fold regression results and filtering significant markers (per fold)...")
union_pathways = set()
any_found = False

for fold_idx, top_phenos in enumerate(fold_top_phenotypes, start=1):
    if not top_phenos:
        print(f"[Fold {fold_idx}] Skipping (no top phenotypes from this fold).")
        continue
    # Read regression results for this fold (training fold only; file must exist, e.g. *_fold1.csv)
    fold_regression_path = REGRESSION_RESULTS_FILE.replace('.csv', f'_fold{fold_idx}.csv')
    try:
        fold_results_df = pd.read_csv(fold_regression_path)
        print(f"[Fold {fold_idx}] Loaded regression results from '{fold_regression_path}' ({len(fold_results_df)} rows).")
    except FileNotFoundError:
        print(f"\nERROR: Per-fold regression file not found: {fold_regression_path}")
        print("Provide regression results computed on each training fold (e.g. pathway_phenotype_regression_results_corrected_fold1.csv).")
        exit(1)
    except Exception as e:
        print(f"\nERROR: Failed to load {fold_regression_path}. Details: {e}")
        exit(1)

    filtered_by_pheno = fold_results_df[fold_results_df[COL_REGRESSION_PHENOTYPE].isin(top_phenos)]
    significant_results_df = filtered_by_pheno[filtered_by_pheno[COL_REGRESSION_PVALUE] < 0.05].copy()

    if not significant_results_df.empty:
        any_found = True
        unique_features = significant_results_df[COL_REGRESSION_FEATURE].unique()
        union_pathways.update(unique_features)

        unique_pathways_df = pd.DataFrame(unique_features, columns=[COL_SIGNIFICANT_OUTPUT])
        # 保存每折通路清单
        pathways_file_fold = SIGNIFICANT_PATHWAYS_OUTPUT_FILE.replace('.csv', f'_fold{fold_idx}.csv')
        unique_pathways_df.to_csv(pathways_file_fold, index=False)
        print(f"[Fold {fold_idx}] {len(unique_pathways_df)} unique significant {LAYER} markers saved to '{pathways_file_fold}'")
    else:
        print(f"[Fold {fold_idx}] No significant {LAYER} markers (p < 0.05) were found for its top 20 phenotypes.")

# 保存并集到原路径，便于兼容（注意：用于 CV 训练时优先用每折文件）
if any_found:
    union_df = pd.DataFrame(sorted(union_pathways), columns=[COL_SIGNIFICANT_OUTPUT])
    union_df.to_csv(SIGNIFICANT_PATHWAYS_OUTPUT_FILE, index=False)
    print(f"\n[Summary] Union of significant {LAYER} markers across folds saved to '{SIGNIFICANT_PATHWAYS_OUTPUT_FILE}'")
else:
    print(f"\nNo significant {LAYER} markers found across all folds.")

print(f"\n{'='*40}\nScript finished.\n{'='*40}")
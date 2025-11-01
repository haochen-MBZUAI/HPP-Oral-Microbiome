import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
# NEW:
from sklearn.model_selection import StratifiedKFold  # 仅用于在训练集上做特征挑选

# --- 1. Configuration: File Paths and Column Names ---
# NOTE: Please verify and update all placeholder paths and column names below.

# --- Input Files ---
PHENOTYPE_DATA_PATH = "home/ec2-user/Stidies/Oral_HPP/oral_data/phenotype_cleaned.csv"
DISEASE_DATA_PATH = 'home/ec2-user/Stidies/Oral_HPP/oral_data/metabolic_diseases.csv'
REGRESSION_RESULTS_FILE = "home/ec2-user/Stidies/Oral_HPP/oral_data/pathway_phenotype_processed.csv"

# --- Output Files ---
#  To predict other diseases, change the hypertension to other diseases
STAT_IMPORTANCE_OUTPUT_FILE = 'home/ec2-user/Stidies/Oral_HPP/oral_data/hypertension/hypertension_pheno_statistic.csv'
SIGNIFICANT_PATHWAYS_OUTPUT_FILE = 'home/ec2-user/Stidies/Oral_HPP/oral_data/hypertension/significant_pathways_from_statistic.csv'

# --- Column Name Configuration for Pathway Analysis ---
# In your regression results file:
COL_REGRESSION_PATHWAY = 'Pathway_Feature'
COL_REGRESSION_PHENOTYPE = 'Phenotype_Feature'
# MODIFICATION: Updated comment to be accurate.
# The column name for phenotype features in the regression results file.
COL_REGRESSION_PVALUE = 'p_corrected_bonferroni'

# NEW: CV 配置
N_SPLITS = 5
RANDOM_STATE = 42

# ==============================================================================
# SCRIPT EXECUTION STARTS HERE
# ==============================================================================

# --- Part 1: Calculate Phenotype Importance for Hypertension ---

print("--- Part 1: Calculating Phenotype Importance for Hypertension ---")

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

# --- Step 1.3: Create the Binary Target Variable for Hypertension ---
print("\nStep 1.3: Creating the target variable for hypertension...")
hypertension_records = disease_df[disease_df['medical_condition'].str.contains('Hypertension', case=False, na=False)]
hypertensive_ids = hypertension_records['participant_id'].unique()
phenotype_df['has_hypertension'] = phenotype_df['participant_id'].isin(hypertensive_ids).astype(int)
print("Distribution of target variable 'has_hypertension':")
print(phenotype_df['has_hypertension'].value_counts())

# --- Step 1.4: Finalize the Feature Matrix (X) and Target Vector (y) ---
print("\nStep 1.4: Preparing the final feature matrix (X) and target vector (y)...")
non_feature_cols = [
    'participant_id', 'collection_date', 'collection_date_x', 'collection_date_y',
    'cohort', 'research_stage', 'array_index', 'id',
    'has_hypertension'
]
y = phenotype_df['has_hypertension']
X = phenotype_df.drop(columns=non_feature_cols, errors='ignore').select_dtypes(include=np.number)
# NEW: 不在全量数据上填充，以免泄漏；每折用训练集统计量
print(f"Final shape of feature matrix X: {X.shape}")
print(f"Final shape of target vector y: {y.shape}")

# --- Step 1.5: Calculate and Save Feature Relevance (Training-Only, 5-Fold CV) ---
print("\nStep 1.5: Calculating feature relevance using ANOVA F-test on TRAINING folds only (5-fold)...")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_top_phenotypes = []  # 用于第2部分找通路
all_stat_files = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n[Fold {fold_idx}/{N_SPLITS}] Fitting SelectKBest on TRAINING data only...")
    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()

    # NEW: 仅用训练集的中位数填充
    X_train.fillna(X_train.median(), inplace=True)

    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X_train, y_train)

    feature_scores_df = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_,
        'P_value': selector.pvalues_
    }).sort_values(by='Score', ascending=False).reset_index(drop=True)

    # 保存每折统计结果
    stat_out_file_fold = STAT_IMPORTANCE_OUTPUT_FILE.replace('.csv', f'_fold{fold_idx}.csv')
    feature_scores_df.to_csv(stat_out_file_fold, index=False)
    all_stat_files.append(stat_out_file_fold)

    print(f"Saved fold-{fold_idx} stats to: {stat_out_file_fold}")
    print("\n--- Top 20 Features on this fold (training-only) ---")
    print(feature_scores_df.head(20).to_string())

    # 记录每折 Top20 phenotype 列表，供第2部分使用
    fold_top_phenotypes.append(feature_scores_df['Feature'].head(20).tolist())

# 为了兼容原路径，也输出一个汇总版（将每折结果叠加，带 fold 列）
try:
    concatenated = []
    for fold_idx, path in enumerate(all_stat_files, start=1):
        tmp = pd.read_csv(path)
        tmp.insert(0, 'fold', fold_idx)
        concatenated.append(tmp)
    if concatenated:
        pd.concat(concatenated, ignore_index=True).to_csv(STAT_IMPORTANCE_OUTPUT_FILE, index=False)
        print(f"\n[Summary] Concatenated per-fold stats saved to: {STAT_IMPORTANCE_OUTPUT_FILE}")
except Exception as e:
    print(f"WARNING: Failed to write concatenated stats: {e}")

# --- Part 2: Find Significant Pathways Linked to Top Phenotypes ---

print(f"\n{'='*60}\n--- Part 2: Finding Significant Pathways from Top Phenotypes ---\n{'='*60}")

# --- Step 2.1: Load Combined Regression Results ---
print("\nStep 2.1: Loading the combined regression results file...")
try:
    all_results_df = pd.read_csv(REGRESSION_RESULTS_FILE)
    print(f"Combined regression data has {len(all_results_df)} rows.")
except Exception as e:
    print(f"\nERROR: Failed to load the regression file. Please check the path. Details: {e}")
    exit()

# --- Step 2.2: Identify Significant Pathways (per fold, training-selected phenotypes) ---
print("\nStep 2.2: Filtering for significant pathways linked to top phenotypes (per fold)...")
union_pathways = set()
any_found = False

for fold_idx, top_phenos in enumerate(fold_top_phenotypes, start=1):
    filtered_by_pheno = all_results_df[all_results_df[COL_REGRESSION_PHENOTYPE].isin(top_phenos)]
    significant_results_df = filtered_by_pheno[filtered_by_pheno[COL_REGRESSION_PVALUE] < 0.05].copy()

    if not significant_results_df.empty:
        any_found = True
        unique_pathways = significant_results_df[COL_REGRESSION_PATHWAY].unique()
        union_pathways.update(unique_pathways)

        unique_pathways_df = pd.DataFrame(unique_pathways, columns=['Significant_Pathway'])
        # 保存每折通路清单
        pathways_file_fold = SIGNIFICANT_PATHWAYS_OUTPUT_FILE.replace('.csv', f'_fold{fold_idx}.csv')
        unique_pathways_df.to_csv(pathways_file_fold, index=False)
        print(f"[Fold {fold_idx}] {len(unique_pathways_df)} unique significant pathways saved to '{pathways_file_fold}'")
    else:
        print(f"[Fold {fold_idx}] No significant pathways (p < 0.05) were found for its top 20 phenotypes.")

# 保存并集到原路径，便于兼容（注意：用于 CV 训练时优先用每折文件）
if any_found:
    union_df = pd.DataFrame(sorted(union_pathways), columns=['Significant_Pathway'])
    union_df.to_csv(SIGNIFICANT_PATHWAYS_OUTPUT_FILE, index=False)
    print(f"\n[Summary] Union of significant pathways across folds saved to '{SIGNIFICANT_PATHWAYS_OUTPUT_FILE}'")
else:
    print(f"\nNo significant pathways found across all folds.")

print(f"\n{'='*40}\nScript finished.\n{'='*40}")
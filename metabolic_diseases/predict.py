import pandas as pd
from datetime import timedelta
import numpy as np
import re

# --- Imports for Modeling ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# NEW:
from sklearn.model_selection import StratifiedKFold  # 5-fold cross-validation
import os  # Used only to check if per-fold feature files exist

############################################################################
### CONFIGURATION AREA ###
### All user settings are here. Please edit this section only. ###
############################################################################

# --- 1. File Paths ---
IMPORTANT_PATHWAYS_FILE = 'home/ec2-user/Stidies/Oral_HPP/oral_data/hypertension/significant_pathways_from_statistic.csv'

# Paths from your original script
FEATURES_FILE = "home/ec2-user/Stidies/Oral_HPP/oral_data/pathway_processed"
LABELS_FILE = 'home/ec2-user/Stidies/Oral_HPP/oral_data/metabolic_diseases.csv'

# --- 2. Data & Modeling Parameters ---
TARGET_DISEASE = 'hypertension'
HEALTHY_CONTROL_LABEL = 'no medical diagnoses'
TIME_WINDOW_DAYS = 180

# --- 3. Columns to Exclude from Features ---
NON_FEATURE_COLS = [
    TARGET_DISEASE, 'collection_date', 'cohort',
    'research_stage', 'array_index'
]

# --- 4. Experiment Switches ---
# Choose the model you want to run: 'svm', 'random_forest', 'xgboost', or 'lightgbm'
MODEL_TO_USE = 'lightgbm'

# If set to True, pathway lists from _fold{n}.csv will be prioritized within each fold
USE_FEATURE_SELECTION = False

# NEW: CV Configuration
N_SPLITS = 5
RANDOM_STATE = 42

# --- STEP 1 to 5 ---

# --- STEP 1: LOAD RAW DATA ---
print("--- STEP 1: Loading raw feature and label data ---")
try:
    df_features = pd.read_csv(FEATURES_FILE, parse_dates=['collection_date'], index_col=0)
    df_labels_raw = pd.read_csv(LABELS_FILE, parse_dates=['collection_date'])
    print(f"Original number of features loaded: {len(df_features.columns) - 1}")
except Exception as e:
    print(f"ERROR: Could not load data files. Please check paths. Details: {e}")
    exit()

# --- STEP 2: CONDITIONAL FEATURE SELECTION (deferred to per-fold) ---
print(f"\n--- STEP 2: Conditional Feature Selection (USE_FEATURE_SELECTION = {USE_FEATURE_SELECTION}) ---")
if USE_FEATURE_SELECTION:
    print("Feature selection will be applied INSIDE each CV fold using TRAINING-only lists.")
    print("If per-fold files like *_fold1.csv are not found, will fallback to the union file or all features.")
else:
    print("Feature selection is DISABLED. Using all loaded features.")
print(f"Number of features currently loaded (pre-filter): {len(df_features.columns) - 1}")

# --- STEP 3: DEFINE CASE AND CLEAN CONTROL GROUPS ---
print("\n--- STEP 3: Defining Case and Clean Control Groups ---")
case_participants_ids = set(df_labels_raw[df_labels_raw['medical_condition'] == TARGET_DISEASE]['participant_id'])
healthy_participants_ids = set(df_labels_raw[df_labels_raw['medical_condition'] == HEALTHY_CONTROL_LABEL]['participant_id'])
clean_healthy_ids = healthy_participants_ids - case_participants_ids
print(f"Found {len(case_participants_ids)} total case participants and {len(clean_healthy_ids)} clean control participants.")

# --- STEP 4: APPLY TIME-WINDOW FILTER TO THE CASE GROUP ---
print("\n--- STEP 4: Applying time-window filter to the case group ---")
df_case_labels = df_labels_raw[df_labels_raw['participant_id'].isin(case_participants_ids) & (df_labels_raw['medical_condition'] == TARGET_DISEASE)]
df_merged_case = pd.merge(df_features['collection_date'].reset_index(), df_case_labels, on='participant_id')
df_merged_case['date_diff'] = (df_merged_case['collection_date_x'] - df_merged_case['collection_date_y']).dt.days.abs()
is_valid_case = df_merged_case['date_diff'] <= TIME_WINDOW_DAYS
final_case_ids = set(df_merged_case[is_valid_case]['participant_id'])
print(f"Found {len(final_case_ids)} case participants who meet the {TIME_WINDOW_DAYS}-day time window criteria.")

# --- STEP 5: BUILD THE INTERMEDIATE DATASET FOR MODELING ---
print("\n--- STEP 5: Building the intermediate dataset for modeling ---")
final_participant_ids = final_case_ids.union(clean_healthy_ids)
df_model_data = df_features[df_features.index.isin(final_participant_ids)].copy()
df_model_data[TARGET_DISEASE] = df_model_data.index.isin(final_case_ids).astype(int)
print(f"Intermediate dataset has {len(df_model_data)} participants.")
print("Class distribution before undersampling:")
print(df_model_data[TARGET_DISEASE].value_counts())

# --- STEP 6: UNDERSAMPLING THE HEALTHY CONTROL GROUP TO BALANCE CLASSES ---
print(f"\n--- STEP 6: Undersampling the healthy control group for 1:1 balance ---")
df_cases = df_model_data[df_model_data[TARGET_DISEASE] == 1]
df_controls = df_model_data[df_model_data[TARGET_DISEASE] == 0]

n_controls_to_keep = len(df_cases)

if n_controls_to_keep > 0 and n_controls_to_keep <= len(df_controls):
    print(f"Strategy: Random sampling. Keeping {n_controls_to_keep} controls to match {len(df_cases)} cases.")
    df_controls_undersampled = df_controls.sample(n=n_controls_to_keep, random_state=RANDOM_STATE)
elif len(df_controls) < n_controls_to_keep:
    print(f"Warning: Not enough controls ({len(df_controls)}) to match cases ({len(df_cases)}). Using all available controls.")
    df_controls_undersampled = df_controls
else:
    df_controls_undersampled = df_controls.head(0)
    print("Warning: No cases found, so no controls will be kept.")

df_temp_combined = pd.concat([df_cases, df_controls_undersampled])
df_model_data = df_temp_combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print("Undersampling and shuffling complete.")
print(f"Final dataset size after undersampling: {len(df_model_data)}")
print("Final class distribution:")
print(df_model_data[TARGET_DISEASE].value_counts())

##############################################################################
### STEP 7 (REVISED): PREPARE FINAL X AND Y WITH COLUMN NAME SANITIZING ###
##############################################################################
print("\n--- STEP 7: Preparing Final X and y for Modeling ---")
y = df_model_data[TARGET_DISEASE]
X = df_model_data.drop(columns=NON_FEATURE_COLS, errors='ignore')

# Sanitize feature names for XGBoost AND LightGBM compatibility
print("Sanitizing feature names for Tree-based models compatibility...")
regex = re.compile(r"\[|\]|<|:", re.IGNORECASE)
X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<', ':'))) else col for col in X.columns.values]
print("Column names have been sanitized.")
print(f"Final shape of X for modeling: {X.shape}")

# --- STEP 8: 5-FOLD CROSS-VALIDATION TRAINING AND EVALUATION ---
print(f"\n--- STEP 8: 5-Fold Cross-Validation (MODEL: {MODEL_TO_USE}) ---")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

acc_list, auc_list = [], []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n===== Fold {fold_idx}/{N_SPLITS} =====")

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # --- [Per Fold] Load feature selection pathway list on demand (Priority: Per-fold file -> Summary file -> All features) ---
    if USE_FEATURE_SELECTION:
        # Generate per-fold filename
        if IMPORTANT_PATHWAYS_FILE.endswith('.csv'):
            per_fold_path = IMPORTANT_PATHWAYS_FILE.replace('.csv', f'_fold{fold_idx}.csv')
        else:
            per_fold_path = IMPORTANT_PATHWAYS_FILE + f'_fold{fold_idx}.csv'

        selected_cols = None

        try:
            if os.path.exists(per_fold_path):
                df_imp = pd.read_csv(per_fold_path)
                selected_cols = [c for c in df_imp['Significant_Pathway'].tolist() if c in X_train.columns]
                print(f"Applied per-fold pathway list: {per_fold_path} -> {len(selected_cols)} features")
            elif os.path.exists(IMPORTANT_PATHWAYS_FILE):
                df_imp = pd.read_csv(IMPORTANT_PATHWAYS_FILE)
                selected_cols = [c for c in df_imp['Significant_Pathway'].tolist() if c in X_train.columns]
                print(f"Per-fold list not found. Fallback to union file: {IMPORTANT_PATHWAYS_FILE} -> {len(selected_cols)} features")
            else:
                print("WARNING: Pathway list not found. Using all features.")
        except Exception as e:
            print(f"WARNING: Failed to read pathway file ({e}). Using all features.")
            selected_cols = None

        if selected_cols and len(selected_cols) > 0:
            X_train = X_train[selected_cols]
            X_test = X_test[selected_cols]
        else:
            print("NOTE: No matching pathway columns found in feature matrix. Using all features.")

    # --- [Per Fold] Auto-clean (ensure numeric) + Imputation using only training set statistics ---
    dirty_columns = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            if X_train[col].isnull().any():
                dirty_columns.append(col)
    if dirty_columns:
        print(f"WARNING: Cleaned non-numeric columns in training set: {dirty_columns}")

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # --- [Per Fold] Training and Prediction ---
    if MODEL_TO_USE == 'svm':
        print("Applying StandardScaler for SVM...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    elif MODEL_TO_USE == 'random_forest':
        model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    elif MODEL_TO_USE == 'xgboost':
        model = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    elif MODEL_TO_USE == 'lightgbm':
        model = LGBMClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1,
            verbosity=-1  # Suppress warnings
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    else:
        print(f"ERROR: Model '{MODEL_TO_USE}' is not recognized. Exiting.")
        exit()

    # --- [Per Fold] Evaluation ---
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except Exception as e:
        print(f"WARNING: AUC failed on this fold ({e}). Setting to NaN.")
        auc = float('nan')

    acc_list.append(acc)
    auc_list.append(auc)

    print("\n--- Fold Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC Score: {auc:.4f}" if not np.isnan(auc) else "AUC Score: NaN")
    print("\nClassification Report:")
    try:
        print(classification_report(y_test, y_pred, target_names=['Not Hypertensive (0)', 'Hypertensive (1)']))
    except Exception:
        # If a fold only has a single class, target_names might raise an error
        print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# --- CV Summary ---
print("\n=== Cross-Validation Summary ===")
if len(acc_list) > 0:
    print(f"Accuracy (mean ± std): {np.nanmean(acc_list):.4f} ± {np.nanstd(acc_list):.4f}")
if len(auc_list) > 0:
    print(f"AUC (mean ± std): {np.nanmean(auc_list):.4f} ± {np.nanstd(auc_list):.4f}")

print(f"\n{'='*40}\nScript finished.\n{'='*40}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Run Association Analysis for Replication Study

Linear regression between genus-level oral microbiome features and target
phenotypes (BMI and waist circumference), controlling for age, sex, and smoking.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from tqdm import tqdm
import os

# =============================================================================
# Configuration
# =============================================================================

_BASE_DIR = os.environ.get("REPLICATION_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

GENUS_DATA_FILE = os.path.join(_BASE_DIR, "genus_processed.csv")
PHENOTYPE_DATA_FILE = os.path.join(_BASE_DIR, "phenotype_cleaned.csv")
SMOKING_DATA_FILE = os.path.join(_BASE_DIR, "raw_csv", "smoking_status.csv")

TARGET_PHENOTYPES = ['BMXBMI', 'BMXWAIST']
AGE_COL = 'RIDAGEYR'
SEX_COL = 'RIAGENDR'
SMOKING_COL = 'smoking_status'

OUTPUT_FILE = os.path.join(_BASE_DIR, "regression_result", "replication_association_results.csv")
TAXONOMY_MAPPING_FILE = None

# =============================================================================
# Functions
# =============================================================================

def load_csv(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File not found at path: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV file from: {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"An unexpected error occurred while loading the CSV file {file_path}: {e}")
        return None


def run_genus_phenotype_regression(df: pd.DataFrame):
    print("Starting regression analysis...")
    print(f"Input DataFrame shape: {df.shape}")

    required_cols = [AGE_COL, SEX_COL]
    if SMOKING_COL in df.columns:
        required_cols.append(SMOKING_COL)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return pd.DataFrame()

    genus_cols = [col for col in df.columns if col.startswith('RB_genus')]
    if not genus_cols:
        print("Error: No genus columns found (expected columns starting with 'RB_genus')")
        return pd.DataFrame()
    print(f"Found {len(genus_cols)} genus features")

    available_phenotypes = [p for p in TARGET_PHENOTYPES if p in df.columns]
    if not available_phenotypes:
        print(f"Error: None of the target phenotypes {TARGET_PHENOTYPES} found in data")
        return pd.DataFrame()
    print(f"Target phenotypes: {available_phenotypes}")

    safe_target_name = 'TARGET'
    safe_predictor_name = 'PREDICTOR'
    safe_age_name = 'AGE'
    safe_sex_name = 'SEX'
    safe_smoking_name = 'SMOKING'

    results_list = []
    total_iterations = len(genus_cols) * len(available_phenotypes)
    print(f"Total regression analyses to perform: {total_iterations}")

    with tqdm(total=total_iterations, desc="Running regressions") as pbar:
        for genus_col_name in genus_cols:
            for phenotype_col_name in available_phenotypes:
                try:
                    cols_to_select = [phenotype_col_name, genus_col_name, AGE_COL, SEX_COL]
                    if SMOKING_COL in df.columns:
                        cols_to_select.append(SMOKING_COL)
                    temp_df = df[cols_to_select].copy()
                    temp_df.columns = [safe_target_name, safe_predictor_name, safe_age_name, safe_sex_name] + \
                                    ([safe_smoking_name] if SMOKING_COL in df.columns else [])

                    for col in temp_df.columns:
                        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                    temp_df.dropna(inplace=True)

                    if len(temp_df) >= 30:
                        if SMOKING_COL in df.columns:
                            formula = f'{safe_target_name} ~ {safe_predictor_name} + {safe_age_name} + C({safe_sex_name}) + C({safe_smoking_name})'
                        else:
                            formula = f'{safe_target_name} ~ {safe_predictor_name} + {safe_age_name} + C({safe_sex_name})'
                        model = smf.ols(formula, data=temp_df).fit()

                        param_predictor = model.params.get(safe_predictor_name, float('nan'))
                        pval_predictor = model.pvalues.get(safe_predictor_name, float('nan'))
                        param_age = model.params.get(safe_age_name, float('nan'))
                        pval_age = model.pvalues.get(safe_age_name, float('nan'))
                        sex_param_name = next((p for p in model.params.index if p.startswith(f'C({safe_sex_name})[T.')), None)
                        param_sex = model.params.get(sex_param_name, float('nan')) if sex_param_name else float('nan')
                        pval_sex = model.pvalues.get(sex_param_name, float('nan')) if sex_param_name else float('nan')
                        smoking_param_name = None
                        param_smoking = float('nan')
                        pval_smoking = float('nan')
                        if SMOKING_COL in df.columns:
                            smoking_param_name = next((p for p in model.params.index if p.startswith(f'C({safe_smoking_name})[T.')), None)
                            param_smoking = model.params.get(smoking_param_name, float('nan')) if smoking_param_name else float('nan')
                            pval_smoking = model.pvalues.get(smoking_param_name, float('nan')) if smoking_param_name else float('nan')

                        results_list.append({
                            'Genus_Feature': genus_col_name,
                            'Phenotype_Feature': phenotype_col_name,
                            'N_Observations': model.nobs,
                            'R_Squared': model.rsquared,
                            'Predictor_Coeff': param_predictor,
                            'Predictor_PValue': pval_predictor,
                            'Age_Coeff': param_age,
                            'Age_PValue': pval_age,
                            'Sex_Coeff': param_sex,
                            'Sex_PValue': pval_sex,
                            'Smoking_Coeff': param_smoking,
                            'Smoking_PValue': pval_smoking,
                            'Sex_Param_Name_Used': sex_param_name if sex_param_name else "N/A",
                            'Smoking_Param_Name_Used': smoking_param_name if smoking_param_name else "N/A",
                            'Error_Message': None
                        })
                    else:
                        results_list.append({
                            'Genus_Feature': genus_col_name,
                            'Phenotype_Feature': phenotype_col_name,
                            'N_Observations': len(temp_df),
                            'R_Squared': float('nan'),
                            'Predictor_Coeff': float('nan'),
                            'Predictor_PValue': float('nan'),
                            'Error_Message': 'Insufficient data after NaN drop (less than 30 observations)'
                        })
                except Exception as e_regression:
                    print(f"\nRegression error: Genus={genus_col_name}, Phenotype={phenotype_col_name}. Error: {e_regression}")
                    results_list.append({
                        'Genus_Feature': genus_col_name,
                        'Phenotype_Feature': phenotype_col_name,
                        'N_Observations': float('nan'),
                        'R_Squared': float('nan'),
                        'Predictor_Coeff': float('nan'),
                        'Predictor_PValue': float('nan'),
                        'Error_Message': str(e_regression)
                    })
                finally:
                    pbar.update(1)

    print("\nRegression analysis completed.")
    return pd.DataFrame(results_list)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Replication Study: Association Analysis")
    print("=" * 80)

    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("\nStep 1: Loading data...")
    genus_df = load_csv(GENUS_DATA_FILE)
    phenotype_df = load_csv(PHENOTYPE_DATA_FILE)
    if genus_df is None or phenotype_df is None:
        print("Error: Could not load required data files. Exiting.")
        exit(1)

    print("\nStep 2: Merging genus and phenotype data...")
    merged_df = pd.merge(phenotype_df, genus_df, on='SEQN', how='inner')
    print(f"Merged genus and phenotype data shape: {merged_df.shape}")

    if os.path.exists(SMOKING_DATA_FILE):
        print("\nStep 3: Loading and merging smoking status data...")
        smoking_df = load_csv(SMOKING_DATA_FILE)
        if smoking_df is not None:
            merged_df = pd.merge(merged_df, smoking_df, on='SEQN', how='left')
            print(f"After adding smoking data: {merged_df.shape}")
            smoking_missing = merged_df[SMOKING_COL].isna().sum()
            print(f"Smoking status missing: {smoking_missing} ({smoking_missing/len(merged_df)*100:.2f}%)")
            merged_df[SMOKING_COL] = merged_df[SMOKING_COL].fillna(0).astype(int)
            print(f"Smoking rate: {merged_df[SMOKING_COL].mean()*100:.2f}%")
    else:
        print("\nStep 3: Smoking status file not found. Proceeding without smoking covariate.")

    if merged_df.empty:
        print("Error: Merged data is empty. Exiting.")
        exit(1)

    print("\nStep 4: Running regression analysis...")
    regression_results_df = run_genus_phenotype_regression(merged_df)
    if regression_results_df.empty:
        print("Error: No regression results generated. Exiting.")
        exit(1)

    if TAXONOMY_MAPPING_FILE and os.path.exists(TAXONOMY_MAPPING_FILE):
        print("\nStep 5: Adding Taxonomy column from mapping file...")
        try:
            taxonomy_map = pd.read_csv(TAXONOMY_MAPPING_FILE)
            if 'Taxonomy' in taxonomy_map.columns:
                id_col = 'Genus_Feature' if 'Genus_Feature' in taxonomy_map.columns else taxonomy_map.columns[0]
                taxonomy_dict = dict(zip(taxonomy_map[id_col], taxonomy_map['Taxonomy']))
                regression_results_df['Taxonomy'] = regression_results_df['Genus_Feature'].map(taxonomy_dict)
                print(f"  Added Taxonomy column for {regression_results_df['Taxonomy'].notna().sum()} associations")
        except Exception as e:
            print(f"  Warning: Could not add Taxonomy column: {e}")

    print("\nStep 6: Saving results...")
    regression_results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to: {OUTPUT_FILE}")
    print(regression_results_df.head(10))

    print("\n" + "=" * 80)
    print("Association analysis completed! Next: run 02_calculate_mapping.py")
    print("=" * 80)

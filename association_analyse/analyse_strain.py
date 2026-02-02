import pandas as pd
import statsmodels.formula.api as smf
from tqdm import tqdm
import sys
import os  # Added for file operations


# MODIFICATION: Added a robust load_csv function for completeness.
def load_csv(file_path):
    """
    Loads a CSV file into a Pandas DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV file from: {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"An unexpected error occurred while loading the CSV file {file_path}: {e}")
        return None


def merge_with_time_window(df_microbiome, df_phenotype, time_window_days=180):
    """
    Merges microbiome data with phenotype data based on participant_id and collection_date.
    Only merges rows where the collection_date difference is within the specified time window.
    
    Args:
        df_microbiome: DataFrame with microbiome data (must have 'participant_id' and 'collection_data' columns)
        df_phenotype: DataFrame with phenotype data (must have 'participant_id' and 'collection_data' columns)
        time_window_days: Maximum allowed difference in days between collection dates (default: 180)
    
    Returns:
        Merged DataFrame with only rows where collection dates are within the time window
    """
    print(f"Merging data with {time_window_days}-day time window...")
    
    # Ensure collection_data is datetime
    if 'collection_data' in df_microbiome.columns:
        df_microbiome = df_microbiome.copy()
        df_microbiome['collection_data'] = pd.to_datetime(df_microbiome['collection_data'], errors='coerce')
    
    if 'collection_data' in df_phenotype.columns:
        df_phenotype = df_phenotype.copy()
        df_phenotype['collection_data'] = pd.to_datetime(df_phenotype['collection_data'], errors='coerce')
    
    # Merge on participant_id first
    merged = pd.merge(
        df_microbiome,
        df_phenotype,
        on='participant_id',
        how='inner',
        suffixes=('_microbiome', '_phenotype')
    )
    
    print(f"After merging on participant_id: {len(merged)} rows")
    
    # Calculate date difference
    if 'collection_data_microbiome' in merged.columns and 'collection_data_phenotype' in merged.columns:
        merged['date_diff'] = (merged['collection_data_microbiome'] - merged['collection_data_phenotype']).dt.days.abs()
        
        # Filter by time window
        merged = merged[merged['date_diff'] <= time_window_days].copy()
        print(f"After filtering by {time_window_days}-day time window: {len(merged)} rows")
        
        # Keep only one collection_data column (use microbiome's)
        if 'collection_data_microbiome' in merged.columns:
            merged['collection_data'] = merged['collection_data_microbiome']
            merged = merged.drop(columns=['collection_data_microbiome', 'collection_data_phenotype', 'date_diff'])
    elif 'collection_data' in merged.columns:
        # If both have the same column name, keep it
        pass
    
    # Remove duplicate column suffixes if they exist
    cols_to_drop = [col for col in merged.columns if col.endswith('_microbiome') or col.endswith('_phenotype')]
    # But keep collection_data if it was renamed
    cols_to_drop = [col for col in cols_to_drop if col != 'collection_data_microbiome' and col != 'collection_data_phenotype']
    
    # Handle overlapping columns (keep microbiome version for features, phenotype version for covariates)
    overlapping_cols = set(df_microbiome.columns) & set(df_phenotype.columns)
    overlapping_cols.discard('participant_id')
    overlapping_cols.discard('collection_data')
    
    for col in overlapping_cols:
        if col + '_microbiome' in merged.columns and col + '_phenotype' in merged.columns:
            # For overlapping columns, prefer phenotype version (for covariates like age, sex, smoking)
            if col in ['age_at_collection', 'sex', 'smoking', 'cohort', 'research_stage', 'array_index']:
                merged[col] = merged[col + '_phenotype']
                merged = merged.drop(columns=[col + '_microbiome', col + '_phenotype'])
            else:
                # For other overlapping columns, keep microbiome version
                merged[col] = merged[col + '_microbiome']
                merged = merged.drop(columns=[col + '_microbiome', col + '_phenotype'])
    
    print(f"Final merged data shape: {merged.shape}")
    return merged


# MODIFICATION: Function name and docstring updated to 'strain' and 'phenotype'.
def run_strain_phenotype_regression(df: pd.DataFrame):
    """
    Performs linear regression analysis for strain features vs. phenotype features, controlling for age, sex, and smoking.

    Returns:
        pd.DataFrame: A DataFrame containing all regression results.
                      Returns an empty DataFrame if critical errors occur.
    """
    print("Starting regression analysis...")
    print(f"Input DataFrame shape: {df.shape}")

    # --- 1. Define column names ---
    covariate_age_col_name = "age_at_collection"
    covariate_sex_col_name = "sex"
    covariate_smoking_col_name = "smoking"

    # MODIFICATION: Renamed variables to 'strain' for accuracy.
    strain_start_col_name = "k__Bacteria|p__Actinobacteriota|c__Actinomycetia|o__Actinomycetales|f__Actinomycetaceae|g__Actinobaculum|s__Actinobaculum_sp_oral_taxon_183|t__SGB5892"
    strain_end_col_name = "k__Bacteria|p__Bacteroidetes|c__Chitinophagia|o__Chitinophagales|f__Chitinophagaceae|g__Hydrobacter|s__Hydrobacter_penzbergensis|t__SGB8474"

    # MODIFICATION: Renamed variables to 'phenotype'.
    phenotype_feature_start_col_name = "attenuation_coefficient_qbox"
    phenotype_feature_end_col_name = "body_comp_trunk_lean_mass"

    # --- 2. Check if all specified column names exist in the DataFrame ---
    # MODIFICATION: Updated list with new variable names.
    required_col_names = [
        covariate_age_col_name,
        covariate_sex_col_name,
        covariate_smoking_col_name,
        strain_start_col_name,
        strain_end_col_name,
        phenotype_feature_start_col_name,
        phenotype_feature_end_col_name
    ]

    missing_cols = [name for name in required_col_names if name not in df.columns]
    if missing_cols:
        print(f"Error: The following required column names were not found in the DataFrame: {missing_cols}")
        return pd.DataFrame()

    # --- 3. Get lists of column names for strain and phenotype features ---
    try:
        # MODIFICATION: Renamed variables to 'strain'.
        strain_start_idx = df.columns.get_loc(strain_start_col_name)
        strain_end_idx = df.columns.get_loc(strain_end_col_name)

        # MODIFICATION: Renamed variables to 'phenotype'.
        phenotype_feature_start_idx = df.columns.get_loc(phenotype_feature_start_col_name)
        phenotype_feature_end_idx = df.columns.get_loc(phenotype_feature_end_col_name)

        # MODIFICATION: Renamed variables.
        original_strain_col_names = df.columns[strain_start_idx: strain_end_idx + 1].tolist()
        original_phenotype_feature_col_names = df.columns[
                                               phenotype_feature_start_idx: phenotype_feature_end_idx + 1].tolist()

        print(f"Using '{covariate_age_col_name}' as age covariate.")
        print(f"Using '{covariate_sex_col_name}' as sex covariate.")
        print(f"Using '{covariate_smoking_col_name}' as smoking covariate.")
        # MODIFICATION: Updated print statements for clarity.
        print(f"Found {len(original_strain_col_names)} strain features.")
        print(f"Found {len(original_phenotype_feature_col_names)} phenotype features.")

    except KeyError as e:
        print(f"Error: A specified column name could not be found during index lookup: {e}")
        return pd.DataFrame()

    # MODIFICATION: Renamed variables and updated error message.
    if not original_strain_col_names or not original_phenotype_feature_col_names:
        print("Error: No strain or phenotype feature columns found based on the provided start/end names.")
        return pd.DataFrame()

    # --- 4. Define safe column names for use in the regression formula ---
    safe_target_feature_name = 'TARGET_FEATURE'
    safe_predictor_feature_name = 'PREDICTOR_FEATURE'
    safe_age_name = 'COVARIATE_AGE'
    safe_sex_name = 'COVARIATE_SEX'
    safe_smoking_name = 'COVARIATE_SMOKING'

    results_list = []

    # MODIFICATION: Renamed variables.
    total_iterations = len(original_strain_col_names) * len(original_phenotype_feature_col_names)
    print(f"A total of {total_iterations} regression analyses will be performed...")

    # --- 5. Iterate and perform regression analysis ---
    with tqdm(total=total_iterations, desc="Running Regressions") as pbar:
        # MODIFICATION: Renamed loop variable.
        for strain_name_original in original_strain_col_names:
            # MODIFICATION: Renamed loop variable.
            for phenotype_feature_name_original in original_phenotype_feature_col_names:
                try:
                    # MODIFICATION: Renamed variables.
                    cols_to_select = [
                        phenotype_feature_name_original,
                        strain_name_original,
                        covariate_age_col_name,
                        covariate_sex_col_name,
                        covariate_smoking_col_name
                    ]

                    temp_df_for_regression = df[cols_to_select].copy()

                    temp_df_for_regression.columns = [
                        safe_target_feature_name,
                        safe_predictor_feature_name,
                        safe_age_name,
                        safe_sex_name,
                        safe_smoking_name
                    ]

                    for col in temp_df_for_regression.columns:
                        temp_df_for_regression[col] = pd.to_numeric(temp_df_for_regression[col], errors='coerce')

                    temp_df_for_regression.dropna(inplace=True)

                    if len(temp_df_for_regression) >= 30:
                        formula = f'{safe_target_feature_name} ~ {safe_predictor_feature_name} + {safe_age_name} + C({safe_sex_name}) + C({safe_smoking_name})'
                        model = smf.ols(formula, data=temp_df_for_regression).fit()

                        param_predictor = model.params.get(safe_predictor_feature_name, float('nan'))
                        pval_predictor = model.pvalues.get(safe_predictor_feature_name, float('nan'))
                        param_age = model.params.get(safe_age_name, float('nan'))
                        pval_age = model.pvalues.get(safe_age_name, float('nan'))

                        sex_param_name = next((p for p in model.params.index if p.startswith(f'C({safe_sex_name})[T.')),
                                              None)
                        param_sex = model.params.get(sex_param_name, float('nan')) if sex_param_name else float('nan')
                        pval_sex = model.pvalues.get(sex_param_name, float('nan')) if sex_param_name else float('nan')

                        smoking_param_name = next(
                            (p for p in model.params.index if p.startswith(f'C({safe_smoking_name})[T.')), None)
                        param_smoking = model.params.get(smoking_param_name,
                                                         float('nan')) if smoking_param_name else float('nan')
                        pval_smoking = model.pvalues.get(smoking_param_name,
                                                         float('nan')) if smoking_param_name else float('nan')

                        # MODIFICATION: Renamed dictionary keys for accurate output.
                        results_list.append({
                            'Strain_Feature': strain_name_original,
                            'Phenotype_Feature': phenotype_feature_name_original,
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
                        # MODIFICATION: Renamed dictionary keys for accurate output.
                        results_list.append({
                            'Strain_Feature': strain_name_original,
                            'Phenotype_Feature': phenotype_feature_name_original,
                            'N_Observations': len(temp_df_for_regression),
                            'R_Squared': float('nan'),
                            'Predictor_Coeff': float('nan'),
                            'Predictor_PValue': float('nan'),
                            'Age_Coeff': float('nan'),
                            'Age_PValue': float('nan'),
                            'Sex_Coeff': float('nan'),
                            'Sex_PValue': float('nan'),
                            'Smoking_Coeff': float('nan'),
                            'Smoking_PValue': float('nan'),
                            'Sex_Param_Name_Used': "N/A",
                            'Smoking_Param_Name_Used': "N/A",
                            'Error_Message': 'Insufficient data after NaN drop (less than 30 observations)'
                        })

                except Exception as e_regression:
                    # MODIFICATION: Updated error message f-string.
                    print(
                        f"\nRegression error: Strain={strain_name_original}, PhenotypeFeature={phenotype_feature_name_original}. Error: {e_regression}")
                    # MODIFICATION: Renamed dictionary keys for accurate output.
                    results_list.append({
                        'Strain_Feature': strain_name_original,
                        'Phenotype_Feature': phenotype_feature_name_original,
                        'N_Observations': float('nan'),
                        'R_Squared': float('nan'),
                        'Predictor_Coeff': float('nan'),
                        'Predictor_PValue': float('nan'),
                        'Age_Coeff': float('nan'),
                        'Age_PValue': float('nan'),
                        'Sex_Coeff': float('nan'),
                        'Sex_PValue': float('nan'),
                        'Smoking_Coeff': float('nan'),
                        'Smoking_PValue': float('nan'),
                        'Sex_Param_Name_Used': "N/A",
                        'Smoking_Param_Name_Used': "N/A",
                        'Error_Message': str(e_regression)
                    })
                finally:
                    pbar.update(1)

    print("\nRegression analysis completed.")
    return pd.DataFrame(results_list)


# --- Main script execution ---
# Load separate files and merge with time window
print("=" * 80)
print("Loading strain and phenotype data...")
print("=" * 80)

strain_df = load_csv("home/ec2-user/Studies/Oral_HPP/oral_data/strain_processed.csv")
phenotype_df = load_csv("home/ec2-user/Studies/Oral_HPP/oral_data/phenotype_cleaned.csv")

if strain_df is not None and phenotype_df is not None and not strain_df.empty and not phenotype_df.empty:
    # Merge with 180-day time window
    strain_phenotype_df = merge_with_time_window(strain_df, phenotype_df, time_window_days=180)

if strain_phenotype_df is not None and not strain_phenotype_df.empty:
    # MODIFICATION: Called the renamed function.
    regression_results_df = run_strain_phenotype_regression(strain_phenotype_df)

    if not regression_results_df.empty:
        print("\nRegression results summary:")
        print(regression_results_df.head())

        output_path = '/home/ec2-user/Studies/Oral_HPP/oral_data/regression_result/strain_phenotype_regression_results.csv'

        regression_results_df.to_csv(output_path, index=False)
        # MODIFICATION: Updated confirmation message.
        print(f"\nResults saved to '{output_path}'")
    else:
        print("\nSkipping regression analysis because merged data is empty.")
else:
    print("\nSkipping regression analysis because the input data could not be loaded or is empty.")

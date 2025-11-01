import pandas as pd
import pyarrow.feather as feather
import os
import argparse
import glob
import statsmodels.formula.api as smf
from tqdm import tqdm
from datetime import datetime


# =============================================================================
# 1. Function Definitions
# =============================================================================

def read_arrow_file(filepath):
    """
    Read an Arrow file, convert it to a DataFrame, and display its dimensions and contents.
    """
    try:
        arrow_table = feather.read_table(filepath)
        df = arrow_table.to_pandas()
        print(f"Successfully loaded Arrow file: {filepath} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading Arrow file {filepath}: {e}")
        return pd.DataFrame()


def load_csv(file_path):
    """
    Loads a CSV file into a Pandas DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV file from: {file_path}")
        return df
    except Exception as e:
        print(f"An unexpected error occurred while loading the CSV file {file_path}: {e}")
        return None


def calculate_age(df):
    """
    Calculates age at collection date based on year and month of birth.
    """
    df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce')

    def compute_age(row):
        if pd.notnull(row['collection_date']) and pd.notnull(row['year_of_birth']) and pd.notnull(
                row['month_of_birth']):
            try:
                year_of_birth = int(row['year_of_birth'])
                month_of_birth = int(row['month_of_birth'])
                if not (1 <= month_of_birth <= 12):
                    return None
                birth_date = datetime(year_of_birth, month_of_birth, 1)
                age_in_days = (row['collection_date'] - birth_date).days
                return round(age_in_days / 365.25, 2)
            except (ValueError, TypeError):
                return None
        return None

    df['age_at_collection'] = df.apply(compute_age, axis=1)

    try:
        cols = list(df.columns)
        age_col = cols.pop(cols.index('age_at_collection'))
        if 'year_of_birth' in cols:
            year_of_birth_index = cols.index('year_of_birth')
            cols.insert(year_of_birth_index + 1, age_col)
        else:
            cols.append(age_col)
        df = df[cols]
    except ValueError as e:
        print(f"Warning: Could not reorder 'age_at_collection' column - {e}")

    return df


def merge_dfs_keep_left_columns(df_left, df_right):
    """
    Merges two DataFrames on 'participant_id', keeping all columns from the left DataFrame
    and adding only non-conflicting columns from the right DataFrame.
    """
    if 'participant_id' not in df_left.columns or 'participant_id' not in df_right.columns:
        raise ValueError("Error: 'participant_id' must exist in both DataFrames.")

    cols_to_remove_from_right = [col for col in df_right.columns if col in df_left.columns and col != 'participant_id']
    df_right_for_merge = df_right.drop(columns=cols_to_remove_from_right)

    merged_df = pd.merge(df_left, df_right_for_merge, on='participant_id', how='left')
    return merged_df


# MODIFICATION: Renamed function and its parameters from 'liver' to 'phenotype' for accuracy.
def run_gene_family_phenotype_regression(df, age_col_idx, sex_col_idx, smoking_col_idx,
                                         pathway_start_col_idx, pathway_end_col_idx,
                                         phenotype_feature_start_col_idx, phenotype_feature_end_col_idx):
    """
    Performs linear regression for gene family features vs. phenotype features, controlling for age, sex, and smoking.
    """
    print("Starting regression analysis...")
    # MODIFICATION: Updated parameter names in the validation check.
    if not all(isinstance(idx, int) for idx in
               [age_col_idx, sex_col_idx, smoking_col_idx, pathway_start_col_idx, pathway_end_col_idx,
                phenotype_feature_start_col_idx, phenotype_feature_end_col_idx]):
        print("Error: All column indices must be integers.")
        return pd.DataFrame()

    max_col_idx = df.shape[1] - 1
    # MODIFICATION: Updated parameter names in the required indices list.
    required_indices = [age_col_idx, sex_col_idx, smoking_col_idx, pathway_start_col_idx, pathway_end_col_idx,
                        phenotype_feature_start_col_idx, phenotype_feature_end_col_idx]
    if not all(0 <= idx <= max_col_idx for idx in required_indices):
        print(
            f"Error: Column indices are out of DataFrame bounds (max index: {max_col_idx}). Please check your inputs.")
        return pd.DataFrame()

    try:
        original_age_col_name = df.columns[age_col_idx]
        original_sex_col_name = df.columns[sex_col_idx]
        original_smoking_col_name = df.columns[smoking_col_idx]
        original_pathway_col_names = df.columns[pathway_start_col_idx: pathway_end_col_idx + 1].tolist()
        # MODIFICATION: Renamed variable to use 'phenotype'.
        original_phenotype_feature_col_names = df.columns[
                                               phenotype_feature_start_col_idx: phenotype_feature_end_col_idx + 1].tolist()
    except IndexError as e:
        print(f"Error extracting column names with provided indices: {e}")
        return pd.DataFrame()

    # MODIFICATION: Updated variable name and print statement.
    if not original_pathway_col_names or not original_phenotype_feature_col_names:
        print(
            "Error: No pathway or phenotype feature columns found. Please check column name patterns or index ranges.")
        return pd.DataFrame()

    # MODIFICATION: Updated variable name and print statement.
    print(
        f"Found {len(original_pathway_col_names)} pathway features and {len(original_phenotype_feature_col_names)} phenotype features.")

    safe_target_feature_name = 'TARGET_FEATURE'
    safe_predictor_feature_name = 'PREDICTOR_FEATURE'
    safe_age_name = 'COVARIATE_AGE'
    safe_sex_name = 'COVARIATE_SEX'
    safe_smoking_name = 'COVARIATE_SMOKING'
    results_list = []

    # MODIFICATION: Updated variable name.
    total_iterations = len(original_pathway_col_names) * len(original_phenotype_feature_col_names)
    print(f"A total of {total_iterations} regression analyses will be performed...")

    with tqdm(total=total_iterations, desc="Running Regressions") as pbar:
        for pathway_name_original in original_pathway_col_names:
            # MODIFICATION: Renamed loop variable to use 'phenotype'.
            for phenotype_feature_name_original in original_phenotype_feature_col_names:
                try:
                    # MODIFICATION: Renamed variable to use 'phenotype'.
                    cols_to_select = [phenotype_feature_name_original, pathway_name_original, original_age_col_name,
                                      original_sex_col_name, original_smoking_col_name]
                    temp_df_for_regression = df[cols_to_select].copy()
                    temp_df_for_regression.columns = [safe_target_feature_name, safe_predictor_feature_name,
                                                      safe_age_name, safe_sex_name, safe_smoking_name]

                    for col in temp_df_for_regression.columns:
                        temp_df_for_regression[col] = pd.to_numeric(temp_df_for_regression[col], errors='coerce')
                    temp_df_for_regression.dropna(inplace=True)

                    if len(temp_df_for_regression) >= 200:
                        formula = f'{safe_target_feature_name} ~ {safe_predictor_feature_name} + {safe_age_name} + C({safe_sex_name}) + C({safe_smoking_name})'
                        model = smf.ols(formula, data=temp_df_for_regression).fit()

                        param_predictor = model.params.get(safe_predictor_feature_name, float('nan'))
                        pval_predictor = model.pvalues.get(safe_predictor_feature_name, float('nan'))
                        param_age = model.params.get(safe_age_name, float('nan'))
                        pval_age = model.pvalues.get(safe_age_name, float('nan'))

                        sex_param_name = next((p for p in model.params.index if p.startswith(f'C({safe_sex_name})[T.')), None)
                        param_sex = model.params.get(sex_param_name, float('nan')) if sex_param_name else float('nan')
                        pval_sex = model.pvalues.get(sex_param_name, float('nan')) if sex_param_name else float('nan')

                        smoking_param_name = next((p for p in model.params.index if p.startswith(f'C({safe_smoking_name})[T.')), None)
                        param_smoking = model.params.get(smoking_param_name, float('nan')) if smoking_param_name else float('nan')
                        pval_smoking = model.pvalues.get(smoking_param_name, float('nan')) if smoking_param_name else float('nan')

                        # MODIFICATION: Renamed 'Liver_Feature' key to 'Phenotype_Feature'.
                        results_list.append({
                            'Pathway_Feature': pathway_name_original,
                            'Phenotype_Feature': phenotype_feature_name_original,
                            'N_Observations': model.nobs,
                            'R_Squared': model.rsquared,
                            'Predictor_Coeff': param_predictor, 'Predictor_PValue': pval_predictor,
                            'Age_Coeff': param_age, 'Age_PValue': pval_age,
                            'Sex_Coeff': param_sex, 'Sex_PValue': pval_sex,
                            'Smoking_Coeff': param_smoking, 'Smoking_PValue': pval_smoking,
                            'Error_Message': None
                        })
                    else:
                        # MODIFICATION: Renamed 'Liver_Feature' key to 'Phenotype_Feature'.
                        results_list.append({
                            'Pathway_Feature': pathway_name_original,
                            'Phenotype_Feature': phenotype_feature_name_original,
                            'N_Observations': len(temp_df_for_regression),
                            'R_Squared': float('nan'),
                            'Predictor_Coeff': float('nan'), 'Predictor_PValue': float('nan'),
                            'Age_Coeff': float('nan'), 'Age_PValue': float('nan'),
                            'Sex_Coeff': float('nan'), 'Sex_PValue': float('nan'),
                            'Smoking_Coeff': float('nan'), 'Smoking_PValue': float('nan'),
                            'Error_Message': 'Insufficient data after NaN drop (< 200 observations)'
                        })
                except Exception as e_regression:
                    # MODIFICATION: Renamed 'Liver_Feature' key to 'Phenotype_Feature'.
                    results_list.append({
                        'Pathway_Feature': pathway_name_original,
                        'Phenotype_Feature': phenotype_feature_name_original,
                        'N_Observations': float('nan'),
                        'R_Squared': float('nan'),
                        'Predictor_Coeff': float('nan'), 'Predictor_PValue': float('nan'),
                        'Age_Coeff': float('nan'), 'Age_PValue': float('nan'),
                        'Sex_Coeff': float('nan'), 'Sex_PValue': float('nan'),
                        'Smoking_Coeff': float('nan'), 'Smoking_PValue': float('nan'),
                        'Error_Message': str(e_regression)
                    })
                finally:
                    pbar.update(1)

    print("\nRegression analysis completed.")
    if not results_list:
        print("Warning: No regression results were generated.")
    return pd.DataFrame(results_list)


# =============================================================================
# 2. Main Execution Block
# =============================================================================

def main():
    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(
        # MODIFICATION: Updated description to use 'phenotype'.
        description="Run regression analysis between gene family features and phenotype metrics.")
    parser.add_argument("--arrow_dir", type=str,
                        default="home/ec2-user/Stidies/Oral_HPP/oral_data/gene_family_processed",
                        help="Directory containing the Arrow files.")
    # MODIFICATION: Renamed argument and help text from 'liver' to 'phenotype'.
    parser.add_argument("--phenotype_data", type=str,
                        default="home/ec2-user/Stidies/Oral_HPP/oral_data/phenotype_processed.csv",
                        help="Path to the phenotype data CSV file.")
    parser.add_argument("--output_dir", type=str,
                        default="/home/ec2-user/Stidies/Oral_HPP/oral_data/regression_result/gene_family_phenotype_regression_results",
                        help="Directory to save the regression results.")
    # MODIFICATION: Renamed argument and help text from 'liver' to 'phenotype'.
    parser.add_argument("--phenotype_start_col", type=str, default="attenuation_coefficient_qbox",
                        help="Name of the first phenotype feature column.")
    # MODIFICATION: Renamed argument and help text from 'liver' to 'phenotype'.
    parser.add_argument("--phenotype_end_col", type=str, default="body_comp_trunk_lean_mass",
                        help="Name of the last phenotype feature column.")

    args = parser.parse_args()

    # --- Create output directory if it doesn't exist ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load main phenotype data ---
    # MODIFICATION: Updated print statement and variable name.
    print(f"Loading phenotype data from {args.phenotype_data}...")
    phenotype_df = load_csv(args.phenotype_data)
    if phenotype_df is None or phenotype_df.empty:
        # MODIFICATION: Updated error message.
        print("Error: Phenotype data is empty or could not be loaded. Exiting.")
        return

    # --- Find and process each .arrow file ---
    arrow_files = glob.glob(os.path.join(args.arrow_dir, "*.arrow"))
    if not arrow_files:
        print(f"Error: No .arrow files found in {args.arrow_dir}. Exiting.")
        return

    print(f"Found {len(arrow_files)} .arrow files to process.")

    for arrow_file_path in arrow_files:
        print(f"======== Processing file: {os.path.basename(arrow_file_path)} ========")
        gene_family_df = read_arrow_file(arrow_file_path)
        if gene_family_df.empty:
            print(f"Skipping empty or unreadable file: {arrow_file_path}")
            continue

        gene_family_df = gene_family_df.reset_index()

        # --- Merge, calculate age, and prepare for regression ---
        # MODIFICATION: Renamed variable to use 'phenotype'.
        merged_df = merge_dfs_keep_left_columns(gene_family_df, phenotype_df)
        merged_df_with_age = calculate_age(merged_df)
        print(f"Data merged and age calculated. Final shape for regression: {merged_df_with_age.shape}")

        all_columns = merged_df_with_age.columns.tolist()

        try:
            # --- Dynamically find column indices ---
            age_col_idx = all_columns.index('age_at_collection')
            sex_col_idx = all_columns.index('sex')
            smoking_col_idx = all_columns.index('smoking')

            uniref_cols = [col for col in all_columns if str(col).startswith('UniRef90')]
            if not uniref_cols:
                print(f"Error: No 'UniRef90' columns found in the merged data for {arrow_file_path}. Skipping.")
                continue

            pathway_start_col_idx = all_columns.index(uniref_cols[0])
            pathway_end_col_idx = all_columns.index(uniref_cols[-1])

            # MODIFICATION: Renamed variables to use 'phenotype' and use the new args.
            phenotype_feature_start_col_idx = all_columns.index(args.phenotype_start_col)
            phenotype_feature_end_col_idx = all_columns.index(args.phenotype_end_col)

            print("Successfully identified all required column indices.")

        except ValueError as e:
            print(f"Error finding a required column index in the merged data: {e}. Please check column names.")
            print(f"Skipping this file.")
            continue

        # --- Run Regression ---
        # MODIFICATION: Called the renamed function with the renamed variables.
        regression_results_df = run_gene_family_phenotype_regression(
            merged_df_with_age, age_col_idx, sex_col_idx, smoking_col_idx,
            pathway_start_col_idx, pathway_end_col_idx,
            phenotype_feature_start_col_idx, phenotype_feature_end_col_idx
        )

        # --- Save Results ---
        if not regression_results_df.empty:
            output_filename = os.path.basename(arrow_file_path).replace(".arrow", "_regression_results.csv")
            output_path = os.path.join(args.output_dir, output_filename)
            regression_results_df.to_csv(output_path, index=False)
            print(f"SUCCESS: Regression results for {os.path.basename(arrow_file_path)} saved to {output_path}")
        else:
            print(f"WARNING: No regression results were generated for {os.path.basename(arrow_file_path)}.")

    print("\n\nWorkflow completed for all files.")


if __name__ == "__main__":
    main()
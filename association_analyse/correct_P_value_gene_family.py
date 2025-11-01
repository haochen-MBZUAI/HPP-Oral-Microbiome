# Import required libraries
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
import os
import argparse
from tqdm import tqdm
import glob

# =============================================================================
# 1. FIXED CONFIGURATION (Unchanged from original code)
# =============================================================================
P_VALUE_COLUMN = 'Predictor_PValue'
FEATURE_COLUMN = 'Pathway_Feature'
# Define the significance level (alpha)
ALPHA = 0.05


# =============================================================================
# 2. CORE FUNCTION (Logic is unchanged from original code)
# =============================================================================
def correct_pvalues_in_dataframe(df: pd.DataFrame, p_value_col: str, feature_col: str, alpha: float, methods: list):
    """
    Performs multiple p-value corrections on a given DataFrame by grouping by a feature.
    """
    if df.empty:
        return df

    df_copy = df.copy()

    if p_value_col not in df_copy.columns:
        raise KeyError(f"P-value column '{p_value_col}' not found in the file.")
    if feature_col not in df_copy.columns:
        raise KeyError(f"Feature column '{feature_col}' not found in the file.")

    for method in methods:
        all_indices = []
        all_corrected_pvals = []
        unique_features = [f for f in df_copy[feature_col].unique() if pd.notna(f)]

        feature_iterator = tqdm(
            unique_features,
            desc=f'-> Method {method.upper():<10}',
            leave=False,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
        )

        for feature_name in feature_iterator:
            feature_mask = df_copy[feature_col] == feature_name
            pvals_to_correct = df_copy.loc[feature_mask, p_value_col].dropna()

            if pvals_to_correct.empty:
                continue

            reject, pvals_corrected, _, _ = multipletests(
                pvals_to_correct, alpha=alpha, method=method
            )

            all_indices.append(pvals_to_correct.index)
            all_corrected_pvals.append(pvals_corrected)

        if not all_indices:
            tqdm.write(f"Warning: No valid p-values found to correct for method '{method}'.")
            continue

        final_indices = np.concatenate(all_indices)
        final_corrected_pvals = np.concatenate(all_corrected_pvals)
        pval_col_name = f'p_corrected_{method}'
        df_copy.loc[final_indices, pval_col_name] = final_corrected_pvals

    return df_copy


# =============================================================================
# 3. MAIN EXECUTION LOGIC (MODIFIED FOR GLOBAL CORRECTION)
# =============================================================================
def main(args):
    """
    Main execution function to process all CSV files in a folder with global correction.
    """
    input_folder = args.input_folder
    output_folder = args.output_folder
    methods_to_run = args.methods

    if not os.path.isdir(input_folder):
        print(f"\nError: Input folder not found at '{input_folder}'")
        return

    os.makedirs(output_folder, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    if not csv_files:
        print(f"No CSV files found in the folder '{input_folder}'.")
        return

    print(f"Found {len(csv_files)} CSV files to process.")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Correction methods to be applied: {', '.join(methods_to_run)}\n")

    # MODIFICATION 1: Read and combine all CSV files into a single DataFrame
    all_dfs = []
    print("Step 1: Reading and combining all CSV files...")
    for file_path in tqdm(csv_files, desc="Reading files"):
        try:
            df = pd.read_csv(file_path)
            # Add a temporary column to track the original file of each row
            df['original_filename'] = os.path.basename(file_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read or process file {os.path.basename(file_path)}. Error: {e}. Skipping.")

    if not all_dfs:
        print("No valid data could be loaded from any file. Exiting.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"All files combined. Total rows for correction: {len(combined_df)}")

    # MODIFICATION 2: Perform a single, global p-value correction on the combined data
    print("\nStep 2: Running global p-value correction on all data...")
    try:
        df_corrected_combined = correct_pvalues_in_dataframe(
            df=combined_df,
            p_value_col=P_VALUE_COLUMN,
            feature_col=FEATURE_COLUMN,
            alpha=ALPHA,
            methods=methods_to_run
        )
    except KeyError as e:
        print(
            f"\nCritical Error during correction: {e}. Please check your column names ('{P_VALUE_COLUMN}', '{FEATURE_COLUMN}'). Exiting.")
        return

    # MODIFICATION 3: Split the corrected DataFrame and save results to individual files
    print("\nStep 3: Splitting and saving corrected files...")
    for file_name in tqdm(df_corrected_combined['original_filename'].unique(), desc="Saving files"):
        output_file_path = os.path.join(output_folder, file_name)

        # Filter the data belonging to the current original file
        single_df_corrected = df_corrected_combined[df_corrected_combined['original_filename'] == file_name].copy()

        # Drop the temporary tracking column before saving
        single_df_corrected.drop(columns=['original_filename'], inplace=True)

        # Save the corrected file
        single_df_corrected.to_csv(output_file_path, index=False, encoding='utf-8-sig')

    print("\n\nProcess complete. All files have been processed with global p-value correction.")


# =============================================================================
# 4. COMMAND-LINE ARGUMENT PARSING
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process all CSV files in a folder to perform a GLOBAL p-value correction.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-i', '--input-folder',
        type=str,
        default="/home/ec2-user/Stidies/Oral_HPP/oral_data/regression_result/gene_family_phenotype_regression_results",
        help="Path to the input folder containing CSV files to be processed.\n(e.g., ./path/to/input_data)"
    )

    parser.add_argument(
        '-o', '--output-folder',
        type=str,
        default="/home/ec2-user/Stidies/Oral_HPP/oral_data/regression_result/gene_family_phenotype_regression_results_corrected",
        help="Path to the output folder where corrected CSV files will be saved.\n(e.g., ./path/to/output_data)"
    )

    parser.add_argument(
        '-m', '--methods',
        nargs='+',
        type=str,
        default=['bonferroni'],
        help="Specify one or more p-value correction methods to use.\n"
             "*Available methods include: bonferroni, holm, fdr_bh, fdr_by, etc.\n"
             "*Default: bonferroni holm fdr_bh"
    )

    args = parser.parse_args()
    main(args)
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
ALPHA = 0.05


# =============================================================================
# 2. GLOBAL BONFERRONI WITHIN LAYER (per-text: threshold 0.05 / (N_t * K))
# =============================================================================
def correct_pvalues_global_layer(df: pd.DataFrame, p_value_col: str, alpha: float, methods: list):
    """
    Bonferroni correction applied to the entire layer: all N_t*K tests in one go.
    Per-test threshold = 0.05 / (N_t * K).
    """
    if df.empty:
        return df

    if p_value_col not in df.columns:
        raise KeyError(f"P-value column '{p_value_col}' not found.")

    valid_mask = df[p_value_col].notna()
    indices = df.index[valid_mask]
    pvals = df.loc[indices, p_value_col].astype(float)

    if len(pvals) == 0:
        return df

    for method in methods:
        # multipletests returns (reject, pvals_corrected, alphacSidak, alphacBonf)
        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method=method)
        pval_col_name = f'p_corrected_{method}'
        df.loc[indices, pval_col_name] = pvals_corrected

    return df


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
    n_tests = combined_df[P_VALUE_COLUMN].notna().sum()
    print(f"All files combined. Total tests in gene-family layer: {n_tests} (threshold 0.05/N_t*K).")

    # Global Bonferroni within gene-family layer (per-text)
    print("\nStep 2: Running global Bonferroni correction on entire layer...")
    try:
        df_corrected_combined = correct_pvalues_global_layer(
            df=combined_df,
            p_value_col=P_VALUE_COLUMN,
            alpha=ALPHA,
            methods=methods_to_run
        )
    except KeyError as e:
        print(f"\nCritical Error during correction: {e}. Exiting.")
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
        default="/home/ec2-user/Studies/Oral_HPP/oral_data/regression_result/gene_family_phenotype_regression_results",
        help="Path to the input folder containing CSV files to be processed.\n(e.g., ./path/to/input_data)"
    )

    parser.add_argument(
        '-o', '--output-folder',
        type=str,
        default="/home/ec2-user/Studies/Oral_HPP/oral_data/regression_result/gene_family_phenotype_regression_results_corrected",
        help="Path to the output folder where corrected CSV files will be saved.\n(e.g., ./path/to/output_data)"
    )

    parser.add_argument(
        '-m', '--methods',
        nargs='+',
        type=str,
        default=['bonferroni'],
        help="P-value correction method to use (only Bonferroni is supported).\n"
             "Default: bonferroni"
    )

    args = parser.parse_args()
    main(args)
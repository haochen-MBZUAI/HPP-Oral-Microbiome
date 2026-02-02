import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
import os
import argparse

# =============================================================================
# 1. FIXED CONFIGURATION
# =============================================================================
P_VALUE_COLUMN = 'Predictor_PValue'
ALPHA = 0.05

# Feature column names (will be auto-detected)
POSSIBLE_FEATURE_COLUMNS = ['Strain_Feature', 'Pathway_Feature']


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

    valid_mask = df[p_value_col].notna()
    indices = df.index[valid_mask]
    pvals = df.loc[indices, p_value_col].astype(float)

    if len(pvals) == 0:
        return df

    n_tests = len(pvals)
    for method in methods:
        pvals_corrected, _, _, _ = multipletests(pvals, alpha=alpha, method=method)
        pval_col_name = f'p_corrected_{method}'
        df.loc[indices, pval_col_name] = pvals_corrected

    return df


# =============================================================================
# 3. MAIN EXECUTION LOGIC (Modified for a single file)
# =============================================================================
def main(args):
    """
    Main execution function for a single CSV file.
    """
    print("Starting p-value correction process for a single file...")

    input_file = args.input_file
    output_file = args.output_file
    methods_to_run = args.methods

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return

    print(f"\nInput file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Correction methods to be applied: {', '.join(methods_to_run)}")

    try:
        print(f"\n-> Loading data...")
        df_original = pd.read_csv(input_file)

        # Auto-detect feature column name
        feature_col = None
        for col_name in POSSIBLE_FEATURE_COLUMNS:
            if col_name in df_original.columns:
                feature_col = col_name
                break
        
        if feature_col is None:
            print(f"Error: Could not find feature column. Expected one of: {POSSIBLE_FEATURE_COLUMNS}")
            print(f"Available columns: {list(df_original.columns)}")
            return

        print(f"-> Detected feature column: {feature_col}")
        n_valid = df_original[P_VALUE_COLUMN].notna().sum()
        print(f"-> Global Bonferroni within layer: {n_valid} tests (threshold 0.05/N_t*K)...")
        df_corrected = correct_pvalues_global_layer(
            df=df_original.copy(),
            p_value_col=P_VALUE_COLUMN,
            alpha=ALPHA,
            methods=methods_to_run
        )

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"\n-> Saving corrected file to '{output_file}'...")
        df_corrected.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nProcess complete. Corrected file saved successfully.")

    except FileNotFoundError:
        print(f"Error: An issue occurred while reading the file at '{input_file}'.")
    except KeyError as e:
        print(f"Error: Column not found. Please check column names. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a single CSV file to perform p-value correction.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Modified argument: from --input-folder to --input-file
    parser.add_argument(
        '-i', '--input-file',
        type=str,
        default="/home/ec2-user/Studies/Oral_HPP/oral_data/regression_result/strain_phenotype_regression_results.csv",
        help="Path to the single input CSV file.\n(e.g., C:/data/my_data.csv)"
    )

    # Modified argument: from --output-folder to --output-file
    parser.add_argument(
        '-o', '--output-file',
        type=str,
        default="/home/ec2-user/Studies/Oral_HPP/oral_data/regression_result/strain_phenotype_regression_results_corrected.csv",
        help="Path where the single output CSV file will be saved.\n(e.g., C:/data/my_data_corrected.csv)"
    )

    parser.add_argument(
        '-m', '--methods',
        nargs='+',
        default=['bonferroni'],
        help="P-value correction method to use (only Bonferroni is supported).\n"
             "Default: bonferroni"
    )

    args = parser.parse_args()
    main(args)
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
import os
import argparse
from tqdm import tqdm

# =============================================================================
# 1. FIXED CONFIGURATION
# =============================================================================
P_VALUE_COLUMN = 'Predictor_PValue'
FEATURE_COLUMN = 'Pathway_Feature'
ALPHA = 0.05


# =============================================================================
# 2. CORE FUNCTION (This function remains unchanged)
# =============================================================================
def correct_pvalues_in_dataframe(df: pd.DataFrame, p_value_col: str, feature_col: str, alpha: float, methods: list):
    """
    Performs multiple p-value corrections on a given DataFrame by grouping by a feature.
    """
    if df.empty:
        return df

    unique_features = df[feature_col].unique()
    for method in methods:
        all_indices = []
        all_corrected_pvals = []
        feature_iterator = tqdm(
            unique_features,
            desc=f'--> Method {method.upper():<10}',
            leave=False,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
        )
        for feature_name in feature_iterator:
            feature_mask = df[feature_col] == feature_name
            pvals_to_correct = df.loc[feature_mask, p_value_col].dropna()
            if pvals_to_correct.empty:
                continue

            pvals_corrected, _, _, _ = multipletests(
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
        df.loc[final_indices, pval_col_name] = final_corrected_pvals

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

        print(f"-> Running p-value corrections...")
        df_corrected = correct_pvalues_in_dataframe(
            df=df_original,
            p_value_col=P_VALUE_COLUMN,
            feature_col=FEATURE_COLUMN,
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
        default="/home/ec2-user/Stidies/Oral_HPP/oral_data/regression_result/strain_phenotype_regression_results.csv",
        help="Path to the single input CSV file.\n(e.g., C:/data/my_data.csv)"
    )

    # Modified argument: from --output-folder to --output-file
    parser.add_argument(
        '-o', '--output-file',
        type=str,
        default="/home/ec2-user/Stidies/Oral_HPP/oral_data/regression_result/strain_phenotype_regression_results_corrected.csv",
        help="Path where the single output CSV file will be saved.\n(e.g., C:/data/my_data_corrected.csv)"
    )

    parser.add_argument(
        '-m', '--methods',
        nargs='+',
        default=['bonferroni'],
        help="One or more p-value correction methods to use.\n"
             "Default: bonferroni holm fdr_bh"
    )

    args = parser.parse_args()
    main(args)
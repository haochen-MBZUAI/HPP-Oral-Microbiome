import pandas as pd
import pyarrow
import pyarrow.feather as feather
import numpy as np
import os
import traceback  # For printing detailed error information
from tqdm import tqdm  # For displaying a progress bar


def clean_and_transform_arrow(filepath, output_dir):
    """
    Reads an Arrow file, cleans and transforms the data, and saves the result.

    Args:
        filepath (str): Path to the input Arrow file.
        output_dir (str): Directory to save the processed Arrow file.
    """
    try:
        # 1. Read the Arrow file
        table = feather.read_table(filepath)
        df = table.to_pandas()

        if df.empty:
            print(f"Warning: File {filepath} is empty. Skipping.")
            return

        if len(df) < 200:
            print(f"Warning: File {filepath} has {len(df)} samples (< 200). No feature can have >= 200 non-zero. Skipping.")
            return

        if df.shape[1] < 2:
            print(f"Warning: File {filepath} has fewer than 2 columns. Skipping.")
            return

        participant_id_col_name = df.columns[0]
        collection_data_col_name = df.columns[1]
        abundance_cols = list(df.columns[2:])

        df_ids_collection = df[[participant_id_col_name, collection_data_col_name]].copy()

        if not abundance_cols:
            print(f"Warning: No abundance columns found in file {filepath}.")
            return

        # Retain only features with >= 200 non-zero observations across samples (per-text)
        df_abundance_raw = df[abundance_cols].apply(pd.to_numeric, errors='coerce')
        non_zero_count = (df_abundance_raw > 0).sum(axis=0)
        abundance_cols_retained = non_zero_count[non_zero_count >= 200].index.tolist()
        n_dropped = len(abundance_cols) - len(abundance_cols_retained)
        if n_dropped > 0:
            print(f"  Retained {len(abundance_cols_retained)} features with >= 200 non-zero; dropped {n_dropped} (file {os.path.basename(filepath)}).")
        if not abundance_cols_retained:
            print(f"Warning: No features with >= 200 non-zero in {filepath}. Skipping.")
            return

        abundance_cols = abundance_cols_retained
        df_abundance = df[abundance_cols].copy()

        # Process each retained column: zero replacement (half of 1st percentile of positive values)
        for col in abundance_cols:
            # Convert to numeric, coercing errors to NaN, and ensure float32
            df_abundance[col] = pd.to_numeric(df_abundance[col], errors='coerce').astype(np.float32)

            # Methods: threshold = 1st percentile of positive abundances; replace zero entries with half of threshold
            positive_values = df_abundance[col][df_abundance[col] > 0]

            imputation_value = np.float32(0)  # Default to 0 if no positive values
            if not positive_values.empty:
                percentile_1 = np.percentile(positive_values.dropna(), 1)
                imputation_value = np.float32(percentile_1 / 2)
                if np.isnan(imputation_value):
                    pass

            df_abundance[col] = df_abundance[col].fillna(imputation_value)
            df_abundance[col] = np.where(df_abundance[col] > 0, df_abundance[col], imputation_value)

        # Within-sample total-sum normalization (row-wise)
        row_sums = df_abundance.sum(axis=1)
        # Division of float32 DataFrame by float32 Series should retain float32
        df_normalized_abundance = df_abundance.div(row_sums, axis=0).fillna(
            np.float32(0))  # Ensure fill value is float32

        # Scale to parts per million (PPM)
        # Multiplication by scalar should retain float32
        df_ppm_abundance = df_normalized_abundance * np.float32(1_000_000)

        # Base-10 log transformation (PPM > 0 after zero-replacement, so log well-defined)
        df_log_transformed_abundance = np.log10(df_ppm_abundance)
        df_log_transformed_abundance = df_log_transformed_abundance.astype(np.float32)
        df_log_transformed_abundance.replace([-np.inf, np.inf], np.float32(0), inplace=True)  # safety only

        # Combine non-abundance columns with processed abundance data
        # reset_index ensures alignment for simple concat
        df_ids_collection_reset = df_ids_collection.reset_index(drop=True)
        df_final_abundance_reset = df_log_transformed_abundance.reset_index(drop=True)

        df_processed = pd.concat([df_ids_collection_reset, df_final_abundance_reset], axis=1)

        # Save the processed DataFrame
        base_filename = os.path.basename(filepath)
        output_filename = f"processed_{base_filename}"
        output_filepath = os.path.join(output_dir, output_filename)

        # Convert back to pyarrow Table before saving
        # Pandas dtypes will map to Arrow dtypes:
        # int64 -> int64
        # timestamp[ns] (datetime64[ns]) -> timestamp[ns]
        # float32 -> float
        table_to_save = pyarrow.Table.from_pandas(df_processed, preserve_index=False)
        feather.write_feather(table_to_save, output_filepath)
        # print(f"Processed {filepath} and saved to {output_filepath}")

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        traceback.print_exc()  # Print the full traceback


def main():
    # --- Configuration ---
    input_directory = "home/ec2-user/Studies/Oral_HPP/oral_data/gene_family"
    # Output directory will be created if it doesn't exist
    output_directory = "home/ec2-user/Studies/Oral_HPP/oral_data/gene_family_processed"
    # --- End Configuration ---

    if not os.path.isdir(input_directory):
        print(f"Error: Input directory '{input_directory}' not found.")
        print(f"Please ensure the input directory exists and contains your .arrow files.")
        return

    os.makedirs(output_directory, exist_ok=True)  # Create output directory if it doesn't exist

    arrow_files = [f for f in os.listdir(input_directory) if f.endswith(".arrow")]

    if not arrow_files:
        print(f"No .arrow files found in '{input_directory}'.")
        return

    print(f"Found {len(arrow_files)} .arrow files in '{input_directory}'.")
    print(f"Processed files will be saved in '{output_directory}'.")

    # Process files with a progress bar
    for filename in tqdm(arrow_files, desc="Processing Arrow files"):
        filepath = os.path.join(input_directory, filename)
        clean_and_transform_arrow(filepath, output_directory)

    print(f"\nProcessing complete.")
    print(f"All processed files have been saved to '{output_directory}'.")


if __name__ == "__main__":
    main()
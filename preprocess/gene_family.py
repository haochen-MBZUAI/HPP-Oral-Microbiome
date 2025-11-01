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

        # Identify participant_id, collection_data, and abundance columns
        if df.shape[1] < 2:  # Need at least participant id and collection data
            print(f"Warning: File {filepath} has fewer than 2 columns. Skipping.")
            return

        participant_id_col_name = df.columns[0]
        collection_data_col_name = df.columns[1]
        abundance_cols = list(df.columns[2:])  # Ensure it's a list

        # Separate ID/collection columns from original df to preserve their types
        # These columns will retain their original dtypes (e.g., int64 for ID, timestamp for date)
        df_ids_collection = df[[participant_id_col_name, collection_data_col_name]].copy()

        if not abundance_cols:  # Check if abundance_cols is an empty list
            print(f"Warning: No abundance columns found in file {filepath}.")
            # If you want to save a file with only ID and Date columns in this case:
            # output_filename = f"processed_no_abundance_{os.path.basename(filepath)}"
            # output_filepath = os.path.join(output_dir, output_filename)
            # try:
            #     table_to_save = pyarrow.Table.from_pandas(df_ids_collection, preserve_index=False)
            #     feather.write_feather(table_to_save, output_filepath)
            #     print(f"File {filepath} had no abundance columns. Copied ID/Date columns to {output_filepath}")
            # except Exception as e_save:
            #     print(f"Error saving file {output_filepath}: {e_save}")
            return  # Current logic is to skip files with no abundance columns

        # Create a copy for processing abundance data
        df_abundance = df[abundance_cols].copy()

        # 2. Process each abundance column
        for col in abundance_cols:
            # Convert to numeric, coercing errors to NaN, and ensure float32
            df_abundance[col] = pd.to_numeric(df_abundance[col], errors='coerce').astype(np.float32)

            # Calculate imputation value: 0.5 * (1st percentile of positive, non-zero values)
            positive_values = df_abundance[col][df_abundance[col] > 0]

            imputation_value = np.float32(0)  # Default to 0 if no positive values, ensure it's float32
            if not positive_values.empty:
                # np.percentile on a float32 Series/array should return a float32 scalar
                percentile_1 = np.percentile(positive_values.dropna(), 1)  # dropna just in case
                imputation_value = np.float32(percentile_1 / 2)
                if np.isnan(imputation_value):  # Just in case percentile_1/2 results in nan
                    pass

            # Fill NaN values
            df_abundance[col].fillna(imputation_value, inplace=True)

        # 3. Sum-normalization (row-wise)
        row_sums = df_abundance.sum(axis=1)
        # Division of float32 DataFrame by float32 Series should retain float32
        df_normalized_abundance = df_abundance.div(row_sums, axis=0).fillna(
            np.float32(0))  # Ensure fill value is float32

        # 4. Convert to Parts Per Million (PPM)
        # Multiplication by scalar should retain float32
        df_ppm_abundance = df_normalized_abundance * np.float32(1_000_000)

        # 5. Log10 transformation
        # np.log10 can upcast to float64, need to cast it back
        df_log_transformed_abundance = np.log10(df_ppm_abundance)
        df_log_transformed_abundance = df_log_transformed_abundance.astype(np.float32)  # <--- MODIFICATION

        # When original abundance is zero, after normalization and PPM, the log10 transformed value is 0
        # np.log10(0) results in -np.inf. Replace it with 0.
        # Also handle positive infinity, just in case.
        df_log_transformed_abundance.replace([-np.inf, np.inf], np.float32(0), inplace=True)  # <--- MODIFICATION

        # Combine non-abundance columns with processed abundance data
        # reset_index ensures alignment for simple concat
        df_ids_collection_reset = df_ids_collection.reset_index(drop=True)
        df_final_abundance_reset = df_log_transformed_abundance.reset_index(drop=True)

        df_processed = pd.concat([df_ids_collection_reset, df_final_abundance_reset], axis=1)

        # 6. Save the processed DataFrame
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
    input_directory = "home/ec2-user/Stidies/Oral_HPP/oral_data/gene_family"
    # Output directory will be created if it doesn't exist
    output_directory = "home/ec2-user/Stidies/Oral_HPP/oral_data/gene_family_processed"
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
import pandas as pd
import numpy as np
import os


def clean_and_transform_csv(input_csv_path, output_csv_path):
    """
    Reads a CSV file, cleans and transforms specified 'pathway' columns,
    and saves the result to a new CSV file.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the processed CSV file.
    """
    try:
        # 1. Read the CSV file
        print(f"Reading CSV file: {input_csv_path}...")
        df = pd.read_csv(input_csv_path)

        if df.empty:
            print(f"Warning: File {input_csv_path} is empty. No processing done.")
            # Optionally save an empty file or just return
            df.to_csv(output_csv_path, index=False)
            return

        # 2. Identify columns to keep as is and columns to process (pathway columns)
        cols_to_exclude_from_processing_user_defined = [
            'participant_id', 'cohort', 'research_stage', 'array_index', 'collection_data'
        ]

        # Find which of these user-defined columns actually exist in the DataFrame
        actual_cols_to_keep_asis = [col for col in cols_to_exclude_from_processing_user_defined if col in df.columns]

        # Pathway columns are all other columns
        pathway_cols = [col for col in df.columns if col not in actual_cols_to_keep_asis]

        print(f"Columns to keep as-is: {actual_cols_to_keep_asis}")
        print(f"Pathway columns to process: {pathway_cols}")

        if not pathway_cols:
            print(f"Warning: No pathway columns found to process in {input_csv_path}.")
            # Save the original data if no pathway columns are found
            df.to_csv(output_csv_path, index=False)
            print(f"Original data saved to {output_csv_path} as no pathway columns were identified.")
            return

        # Create a copy of pathway data for processing
        df_pathway_data_processed = df[pathway_cols].copy()

        # 3. Process each pathway column (same logic as for abundance cols previously)
        for col in pathway_cols:
            print(f"Processing pathway column: {col}")
            # Convert to numeric (float32), coercing errors (strings -> NaN)
            df_pathway_data_processed[col] = pd.to_numeric(df_pathway_data_processed[col], errors='coerce').astype(
                np.float32)

            # Calculate imputation value: 0.5 * (1st percentile of positive, non-zero values)
            positive_values = df_pathway_data_processed[col][df_pathway_data_processed[col] > 0]

            imputation_value = np.float32(0)  # Default if no positive values
            if not positive_values.empty:
                percentile_1 = np.percentile(positive_values.dropna(), 1)
                imputation_value = percentile_1 / 2
                imputation_value = np.float32(imputation_value)

            # Fill NaN values
            df_pathway_data_processed[col].fillna(imputation_value, inplace=True)

        # 4. Sum-normalization (row-wise) for pathway data
        print("Performing sum-normalization for pathway data...")
        row_sums = df_pathway_data_processed.sum(axis=1)
        df_normalized_pathway = df_pathway_data_processed.div(row_sums, axis=0).fillna(0)

        # 5. Convert to Parts Per Million (PPM)
        print("Converting to PPM...")
        df_ppm_pathway = df_normalized_pathway * 1_000_000

        # 6. Log10 transformation
        print("Applying log10 transformation...")
        df_log_transformed_pathway = np.log10(df_ppm_pathway)
        # Replace -np.inf (from log10(0)) and any np.inf with 0
        df_log_transformed_pathway.replace([np.inf, -np.inf], 0, inplace=True)

        # 7. Combine non-processed columns with processed pathway data
        # Ensure indices are aligned for concatenation if they were somehow changed (reset_index is a safe bet)
        df_kept_asis_part = df[actual_cols_to_keep_asis].reset_index(drop=True)
        df_processed_part = df_log_transformed_pathway.reset_index(drop=True)

        df_output = pd.concat([df_kept_asis_part, df_processed_part], axis=1)

        # Ensure original column order for the processed part as much as possible, if desired
        # This df_output will have actual_cols_to_keep_asis first, then pathway_cols in their original order.

        # 8. Save the processed DataFrame to a new CSV file
        df_output.to_csv(output_csv_path, index=False)
        print(f"Successfully processed data saved to: {output_csv_path}")

    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
    except Exception as e:
        print(f"An error occurred during processing of {input_csv_path}: {e}")
        import traceback
        traceback.print_exc()


def main():
    # --- Configuration ---
    input_csv_file = "home/ec2-user/Stidies/Oral_HPP/oral_data/pathway.csv"
    output_csv_file = "home/ec2-user/Stidies/Oral_HPP/oral_data/pathway_processed"
    # --- End Configuration ---

    # --- for strain ---
    # input_csv_file = "home/ec2-user/Stidies/Oral_HPP/oral_data/strain.csv"
    # output_csv_file = "home/ec2-user/Stidies/Oral_HPP/oral_data/strain_processed"
    # --- for strain ---

    print(f"Starting CSV processing for: {input_csv_file}")
    clean_and_transform_csv(input_csv_file, output_csv_file)
    print("CSV processing finished.")


if __name__ == "__main__":
    main()
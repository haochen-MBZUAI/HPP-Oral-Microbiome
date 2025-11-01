import pandas as pd
import numpy as np
from pathlib import Path


def process_and_save_csv(file_path: str, id_column: str = 'participant_id'):
    """
    Takes a CSV file path, performs all cleaning operations, and automatically saves a new file.
    This version includes explicit print statements for removal and clipping actions.

    This is the only function you need to call.

    Args:
        file_path (str): The path to the CSV file you want to process.
        id_column (str, optional): The name of the ID column to exclude. Defaults to 'participant_id'.
    """

    # =================================================================
    # All complex logic is encapsulated inside this function.
    # You do not need to read or modify anything below this line.
    # =================================================================

    def _clean_and_filter_single_column(data_series: pd.Series) -> pd.Series:
        """A nested helper function to process one column at a time."""
        series = data_series.copy()
        series_nonan = series.dropna()

        if len(series_nonan) < 20:
            return series  # Not enough data, return as is.

        # Find the 95% densest part of the data
        sorted_data = series_nonan.sort_values().reset_index(drop=True)
        n_points = len(sorted_data)
        n_95_percent = int(n_points * 0.95)

        if n_95_percent == 0:
            return series

        min_range = np.inf
        best_subset = None
        for i in range(n_points - n_95_percent + 1):
            subset = sorted_data.iloc[i: i + n_95_percent]
            current_range = subset.iloc[-1] - subset.iloc[0]
            if current_range < min_range:
                min_range = current_range
                best_subset = subset

        mean_val = best_subset.mean()
        std_val = best_subset.std()

        if std_val == 0:
            return series

        # --- Remove outliers >8 standard deviations ---
        lower_bound_8sd = mean_val - 8 * std_val
        upper_bound_8sd = mean_val + 8 * std_val
        indices_to_remove = series.index[(series < lower_bound_8sd) | (series > upper_bound_8sd)]

        # <<< NEW PRINT STATEMENT IS HERE >>>
        if not indices_to_remove.empty:
            print(f"    -> Action: Removing {len(indices_to_remove)} outlier(s) > 8 SD.")
        series.loc[indices_to_remove] = np.nan

        # --- Clip outliers >5 standard deviations ---
        lower_bound_5sd = mean_val - 5 * std_val
        upper_bound_5sd = mean_val + 5 * std_val
        clipped_series = series.clip(lower=lower_bound_5sd, upper=upper_bound_5sd)

        # Calculate how many values were actually clipped
        # We subtract the count of already removed (NaN) values to not double-count them.
        num_clipped = (series.notna() & (series != clipped_series)).sum()

        # <<< NEW PRINT STATEMENT IS HERE >>>
        if num_clipped > 0:
            print(f"    -> Action: Clipping {num_clipped} value(s) > 5 SD.")

        return clipped_series

    # --- Main process begins ---
    print(f"--- Starting to process file: {file_path} ---")
    try:
        # 1. Load data
        df = pd.read_csv(file_path)
        print(f"Successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        df_cleaned = df.copy()

        # 2. Determine columns to process
        if id_column not in df.columns:
            print(f"WARNING: ID column '{id_column}' not found. Processing all numeric columns.")
            columns_to_process = df.select_dtypes(include=np.number).columns.tolist()
        else:
            columns_to_process = [col for col in df.columns if col != id_column]

        # 3. Loop through columns and clean
        for col_name in columns_to_process:
            if pd.api.types.is_numeric_dtype(df_cleaned[col_name]):
                print(f"Processing column: {col_name}...")
                cleaned_column = _clean_and_filter_single_column(df_cleaned[col_name])
                df_cleaned[col_name] = cleaned_column

        # 4. Save results
        input_path_obj = Path(file_path)
        output_path = input_path_obj.parent / f"{input_path_obj.stem}_cleaned{input_path_obj.suffix}"
        df_cleaned.to_csv(output_path, index=False)
        print(f"\n--- Processing Complete ---")
        print(f"Cleaned file has been saved to: {output_path}")

    except FileNotFoundError:
        print(f"ERROR: File not found at '{file_path}'. Please check the path and filename.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


process_and_save_csv('home/ec2-user/Stidies/Oral_HPP/oral_data/phenotype.csv')
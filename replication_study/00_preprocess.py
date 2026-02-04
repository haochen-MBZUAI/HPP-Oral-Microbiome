#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 0: Convert XPT + Merge G/F + Preprocess Genus + Preprocess Phenotype

1. Convert SAS XPT to CSV and merge G/F files (BMX, DEMO, SMQ).
2. Merge phenotype CSVs into all_phenotypes_raw.csv.
3. Preprocess oral genus data (zero-replacement → normalization → PPM → Log₁₀).
4. Preprocess phenotype data (outlier removal >8 SD, clipping >5 SD).
"""

import pandas as pd
import numpy as np
import os
from functools import reduce

# =============================================================================
# Configuration
# =============================================================================

_BASE_DIR = os.environ.get("REPLICATION_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

XPT_DIR = os.path.join(_BASE_DIR, "xpt_files")
CSV_DIR = os.path.join(_BASE_DIR, "raw_csv")

FILES_TO_PROCESS = [
    ("BMX", "BMX_G.xpt", "BMX_F.xpt", "BMX_combined.csv"),
    ("DEMO", "DEMO_G.xpt", "DEMO_F.xpt", "DEMO_combined.csv"),
    ("SMQ", "SMQ_G.xpt", "SMQ_F.xpt", "SMQ_combined.csv"),
]

# Genus preprocess (optional: skip if oral_genus_raw.csv not present)
ORAL_GENUS_RAW = os.path.join(_BASE_DIR, "oral_genus_raw.csv")
GENUS_PROCESSED = os.path.join(_BASE_DIR, "genus_processed.csv")
MIN_SAMPLES = 200  # minimum number of samples (rows) to run genus preprocess
MIN_NONZERO_PER_FEATURE = 200  # drop columns with fewer than this many non-zero readings

# Phenotype preprocess (input from step 2, output cleaned)
PHENOTYPE_CLEANED = os.path.join(_BASE_DIR, "phenotype_cleaned.csv")
ID_COLUMN = "SEQN"

# =============================================================================
# Part 1: XPT convert and merge
# =============================================================================

def read_xpt_file(filepath):
    try:
        df = pd.read_sas(filepath)
        print(f"  Loaded: {filepath} (shape: {df.shape})")
        return df
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def merge_g_and_f(g_file, f_file, output_file):
    df_g = read_xpt_file(g_file)
    if df_g is None:
        return False
    df_f = read_xpt_file(f_file)
    if df_f is None:
        return False
    if 'SEQN' not in df_g.columns or 'SEQN' not in df_f.columns:
        print("  Error: SEQN column not found")
        return False
    df_combined = pd.concat([df_g, df_f], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['SEQN'], keep='first')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_combined.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    return True


def process_smoking_files(g_file, f_file, output_file):
    df_g = read_xpt_file(g_file)
    if df_g is None:
        return False
    df_f = read_xpt_file(f_file)
    if df_f is None:
        return False
    def create_smoking_status(df):
        s = pd.Series(0, index=df.index)
        if 'SMQ040' in df.columns:
            s[df['SMQ040'] == 1.0] = 1
        return s
    smoking_g = df_g[['SEQN']].copy()
    smoking_g['smoking_status'] = create_smoking_status(df_g)
    smoking_f = df_f[['SEQN']].copy()
    smoking_f['smoking_status'] = create_smoking_status(df_f)
    combined = pd.concat([smoking_g, smoking_f], ignore_index=True).drop_duplicates(subset=['SEQN'], keep='first')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    return True


def merge_all_phenotype_files(csv_dir, output_file):
    phenotype_files = [
        os.path.join(csv_dir, "BMX_combined.csv"),
        os.path.join(csv_dir, "DEMO_combined.csv"),
        os.path.join(csv_dir, "GLU_combined.csv"),
    ]
    dataframes = []
    for fp in phenotype_files:
        if os.path.exists(fp):
            dataframes.append(pd.read_csv(fp))
            print(f"  Loaded: {os.path.basename(fp)}")
        else:
            print(f"  Warning: not found {fp}")
    if not dataframes:
        print("  Error: No phenotype files to merge")
        return False
    merged = reduce(lambda l, r: pd.merge(l, r, on='SEQN', how='outer'), dataframes)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    return True


# =============================================================================
# Part 2: Phenotype preprocess (outlier >8 SD remove, clip >5 SD)
# =============================================================================

def _clean_and_filter_single_column(data_series):
    """Clean a single column: remove outliers >8 SD, clip >5 SD (same as main analysis)."""
    series = data_series.copy()
    series_nonan = series.dropna()
    if len(series_nonan) < 20:
        return series
    sorted_data = series_nonan.sort_values().reset_index(drop=True)
    n_points = len(sorted_data)
    n_95 = int(n_points * 0.95)
    if n_95 == 0:
        return series
    min_range = np.inf
    best_subset = None
    for i in range(n_points - n_95 + 1):
        subset = sorted_data.iloc[i : i + n_95]
        r = subset.iloc[-1] - subset.iloc[0]
        if r < min_range:
            min_range, best_subset = r, subset
    mean_val = best_subset.mean()
    std_val = best_subset.std()
    if std_val == 0:
        return series
    lower_8 = mean_val - 8 * std_val
    upper_8 = mean_val + 8 * std_val
    to_remove = series.index[(series < lower_8) | (series > upper_8)]
    if not to_remove.empty:
        print(f"    -> Removing {len(to_remove)} outlier(s) > 8 SD.")
    series.loc[to_remove] = np.nan
    lower_5 = mean_val - 5 * std_val
    upper_5 = mean_val + 5 * std_val
    clipped = series.clip(lower=lower_5, upper=upper_5)
    num_clipped = (series.notna() & (series != clipped)).sum()
    if num_clipped > 0:
        print(f"    -> Clipping {num_clipped} value(s) > 5 SD.")
    return clipped


def process_and_save_phenotype(file_path, id_column, output_path):
    """Process phenotype: clean each numeric column and save to output_path."""
    print(f"--- Preprocess phenotype: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
        print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        df_cleaned = df.copy()
        if id_column not in df.columns:
            columns_to_process = df.select_dtypes(include=np.number).columns.tolist()
        else:
            columns_to_process = [c for c in df.columns if c != id_column]
        for col in columns_to_process:
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                print(f"  Processing column: {col}...")
                df_cleaned[col] = _clean_and_filter_single_column(df_cleaned[col])
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df_cleaned.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")
    except FileNotFoundError:
        print(f"  ERROR: File not found: {file_path}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# Part 3: Genus preprocess
# =============================================================================

def clean_and_transform_genus(input_csv_path, output_csv_path):
    cols_to_exclude = ['SEQN', 'participant_id', 'cohort', 'research_stage', 'array_index', 'collection_date']
    try:
        df = pd.read_csv(input_csv_path)
        if df.empty or len(df) < MIN_SAMPLES:
            print(f"  Skip genus preprocess: empty or < {MIN_SAMPLES} rows")
            return
        actual_keep = [c for c in cols_to_exclude if c in df.columns]
        genus_cols = [c for c in df.columns if c not in actual_keep]
        if not genus_cols:
            print("  No genus columns found; saving as-is.")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            df.to_csv(output_csv_path, index=False)
            return
        df_genus = df[genus_cols].copy()
        for col in genus_cols:
            df_genus[col] = pd.to_numeric(df_genus[col], errors='coerce').astype(np.float32)
        # Drop columns with fewer than MIN_NONZERO_PER_FEATURE non-zero readings
        non_zero_count = (df_genus > 0).sum()
        genus_cols_retained = [c for c in genus_cols if non_zero_count[c] >= MIN_NONZERO_PER_FEATURE]
        dropped = len(genus_cols) - len(genus_cols_retained)
        if dropped > 0:
            print(f"  Dropped {dropped} feature(s) with < {MIN_NONZERO_PER_FEATURE} non-zero readings; {len(genus_cols_retained)} retained.")
        if not genus_cols_retained:
            print("  No genus columns left after filtering; saving as-is.")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            df.to_csv(output_csv_path, index=False)
            return
        genus_cols = genus_cols_retained
        df_genus = df_genus[genus_cols].copy()
        for col in genus_cols:
            pos = df_genus[col][df_genus[col] > 0]
            imp = np.float32(0)
            if not pos.empty:
                imp = np.float32(np.percentile(pos.dropna(), 1) / 2)
            # Methods: replace zero entries with half of 1st percentile (so PPM>0, log well-defined)
            df_genus[col] = df_genus[col].fillna(imp)
            df_genus[col] = np.where(df_genus[col] > 0, df_genus[col], imp)
        row_sums = df_genus.sum(axis=1)
        df_norm = df_genus.div(row_sums, axis=0).fillna(0)
        df_ppm = df_norm * 1_000_000
        df_log = np.log10(df_ppm).replace([np.inf, -np.inf], 0)  # safety only; no zeros expected after zero-replacement
        df_out = pd.concat([df[actual_keep].reset_index(drop=True), df_log.reset_index(drop=True)], axis=1)
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df_out.to_csv(output_csv_path, index=False)
        print(f"  Genus processed -> {output_csv_path} (shape: {df_out.shape})")
    except Exception as e:
        print(f"  Genus preprocess error: {e}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Step 0: Convert XPT + Merge + Preprocess Genus + Phenotype")
    print("=" * 80)

    # Part 1: XPT -> CSV and merge
    print("\n--- Part 1: XPT convert and merge ---")
    for base_name, g_file, f_file, output_name in FILES_TO_PROCESS:
        g_path = os.path.join(XPT_DIR, g_file)
        f_path = os.path.join(XPT_DIR, f_file)
        out_path = os.path.join(CSV_DIR, output_name)
        if base_name == "SMQ":
            process_smoking_files(g_path, f_path, os.path.join(CSV_DIR, "smoking_status.csv"))
        else:
            merge_g_and_f(g_path, f_path, out_path)

    merged_phenotype = os.path.join(CSV_DIR, "all_phenotypes_raw.csv")
    merge_all_phenotype_files(CSV_DIR, merged_phenotype)

    # Part 2: Phenotype preprocess (clean outliers)
    print("\n--- Part 2: Preprocess phenotype ---")
    if os.path.exists(merged_phenotype):
        process_and_save_phenotype(merged_phenotype, ID_COLUMN, PHENOTYPE_CLEANED)
    else:
        print(f"  all_phenotypes_raw.csv not found; skip phenotype preprocess.")

    # Part 3: Genus preprocess (if file exists)
    print("\n--- Part 3: Preprocess genus ---")
    if os.path.exists(ORAL_GENUS_RAW):
        clean_and_transform_genus(ORAL_GENUS_RAW, GENUS_PROCESSED)
    else:
        print(f"  oral_genus_raw.csv not found at {ORAL_GENUS_RAW}; skip genus preprocess.")

    print("\n" + "=" * 80)
    print("Step 0 done. Next: run 01_run_association.py")
    print("=" * 80)

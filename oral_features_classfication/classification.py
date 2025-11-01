import pandas as pd
import numpy as np
import os

# ==============================================================================
# 1. CONFIGURATION SECTION (Logic Modified for Single Input File)
# ==============================================================================
# !! IMPORTANT !!
# Please provide the FULL path to your SINGLE, PRE-MERGED association file here.
# INPUT_ASSOCIATION_FILE_PATH = "home/ec2-user/Stidies/Oral_HPP/oral_data/strain_phenotype_processed.csv"
INPUT_ASSOCIATION_FILE_PATH = "home/ec2-user/Stidies/Oral_HPP/oral_data/pathway_phenotype_processed.csv"
DOMAIN_MAP_FILE = './feature.csv'

# -- Output Files --
OUTPUT_PREFIX = './pathway'
BENEFICIAL_OUTPUT_FILE = f'{OUTPUT_PREFIX}_Beneficial.csv'
DETRIMENTAL_OUTPUT_FILE = f'{OUTPUT_PREFIX}_Detrimental.csv'
MIXED_OUTPUT_FILE = f'{OUTPUT_PREFIX}_Mixed.csv'

# -- Classification Thresholds --
BENEFICIAL_THRESHOLD = 0.90
DETRIMENTAL_THRESHOLD = 0.10

# -- Debugging --
PRINT_WPP_SCORES = False

# -- System and Domain Definitions --
# This map is still used to assign systems and domains to phenotypes from the loaded file.
SYSTEM_DOMAIN_MAP = {
    'body': [0, 1, 2],
    'cgm': [4, 5, 6, 7],
    'liver': [8, 9, 10]
}

# -- Phenotype Orientation --
INVERTED_PHENOTYPES = {
    'bmi', 'body_comp_total_fat_mass', 'body_comp_total_tissue_percent_fat',
    'body_comp_total_tissue_mass', 'waist_circumference', 'hip_circumference',
    'waist_to_hip_ratio', 'body_comp_android_fat_mass', 'body_comp_android_tissue_percent_fat',
    'body_comp_trunk_fat_mass', 'body_comp_trunk_tissue_percent_fat', 'total_scan_vat_mass',
    'total_scan_vat_volume', 'total_scan_sat_mass', 'total_scan_sat_volume',
    'attenuation_coefficient_qbox', 'elasticity_qbox_median', 'velocity_qbox_median',
    'viscosity_qbox_median', 'dispersion_qbox_median',
    'cgm_mean', 'cgm_median', 'cgm_iqr', 'cgm_gmi', 'cgm_cv', 'cgm_mad', 'cgm_mag',
    'cgm_modd', 'cgm_sd_roc', 'cgm_sdhmm', 'cgm_sdw', 'cgm_sdwsh', 'cgm_above_140',
    'cgm_above_180', 'cgm_hbgi', 'cgm_lbgi', 'cgm_adrr', 'cgm_grade',
}


# ==============================================================================
# 2. DATA LOADING AND PREPARATION (LOGIC MODIFIED FOR SINGLE FILE)
# ==============================================================================
def load_all_data(input_file_path, domain_map_file, system_domain_map):
    """Loads a single combined association data file and the domain mapping file."""
    print("Step 1: Loading and preprocessing data...")
    try:
        domain_df = pd.read_csv(domain_map_file)
    except FileNotFoundError:
        print(f"Error: Domain map file not found at {domain_map_file}")
        return None

    phenotype_to_domain = {row['Feature']: row['group'] for _, row in domain_df.iterrows()}
    phenotype_to_system = {}
    for system, domain_ids in system_domain_map.items():
        for domain_id in domain_ids:
            for feature in domain_df[domain_df['group'] == domain_id]['Feature']:
                phenotype_to_system[feature] = system

    combined_df = None
    print(f"Attempting to load file: {input_file_path}")
    try:
        df = pd.read_csv(input_file_path)
        # Dynamically find the phenotype column (e.g., 'body_Feature', 'cgm_Feature')
        phenotype_col = [col for col in df.columns if '_Feature' in col and 'Pathway' not in col][0]

        df = df.rename(columns={phenotype_col: 'Phenotype', 'p_corrected_bonferroni': 'q_value'})
        df_significant = df[df['q_value'] < 0.05].copy()

        print(f"  SUCCESS: Loaded and filtered {len(df_significant)} significant associations.")
        combined_df = df_significant

    except FileNotFoundError:
        print(f"  ERROR: File not found at '{input_file_path}'. Cannot proceed.")
        return None
    except (IndexError, KeyError):
        print(f"  ERROR: Could not find valid phenotype/p-value columns in '{input_file_path}'. Cannot proceed.")
        return None

    if combined_df is None or combined_df.empty:
        print("Error: No significant associations found after loading. Cannot proceed.")
        return None

    combined_df['domain'] = combined_df['Phenotype'].map(phenotype_to_domain)
    combined_df['system'] = combined_df['Phenotype'].map(phenotype_to_system)

    initial_rows = len(combined_df)
    combined_df.dropna(subset=['domain', 'system', 'Predictor_Coeff', 'q_value'], inplace=True)
    if initial_rows > len(combined_df):
        print(f" - Dropped {initial_rows - len(combined_df)} rows with missing essential data.")

    combined_df['domain'] = combined_df['domain'].astype(int)
    print(f"Total significant associations to be processed: {len(combined_df)}")
    return combined_df


# ==============================================================================
# 3. CORE CLASSIFICATION LOGIC (Unchanged)
# ==============================================================================
def calculate_weights_and_direction(df, inverted_phenotypes):
    """Calculates the y (direction) and w (weight) for each association."""
    print("Step 2: Calculating direction (y) and weight (w)...")

    def get_direction(row):
        is_inverted = row['Phenotype'] in inverted_phenotypes
        coeff_positive = row['Predictor_Coeff'] > 0
        if (is_inverted and coeff_positive) or (not is_inverted and not coeff_positive):
            return -1.0
        return 1.0

    df['y'] = df.apply(get_direction, axis=1)
    df['w'] = df['q_value'].apply(lambda q: 3.0 if q <= 1e-3 else -np.log10(q))
    return df


def classify_pathways(df, beneficial_thresh, detrimental_thresh, print_scores=False):
    """Applies the final methodology to classify each pathway."""
    print("Step 3: Applying classification logic...")
    pathway_classifications = {}
    unique_pathways = df['Pathway_Feature'].unique()

    for pathway in unique_pathways:
        pathway_df = df[df['Pathway_Feature'] == pathway]
        covered_systems = pathway_df['system'].unique()
        system_wpps = {}

        for system in covered_systems:
            system_df = pathway_df[pathway_df['system'] == system]
            domain_agg = system_df.groupby('domain').apply(lambda g: pd.Series({
                'weighted_sum': (g['w'] * g['y']).sum(),
                'total_weight': g['w'].sum()
            })).reset_index()

            domain_agg['y_hat'] = np.sign(domain_agg['weighted_sum'])

            total_system_weight = domain_agg['total_weight'].sum()
            domain_agg['normalized_weight'] = domain_agg[
                                                  'total_weight'] / total_system_weight if total_system_weight > 0 else 0

            wpps = domain_agg[domain_agg['y_hat'] == 1]['normalized_weight'].sum()
            system_wpps[system] = wpps

        if print_scores and system_wpps:
            formatted_scores = {sys: f"{score:.3f}" for sys, score in system_wpps.items()}
            print(f"Pathway: {pathway} -> WPPS: {formatted_scores}")

        if not system_wpps:
            pathway_classifications[pathway] = 'Mixed'
            continue

        scores = list(system_wpps.values())
        if all(s >= beneficial_thresh for s in scores):
            pathway_classifications[pathway] = 'Beneficial'
        elif all(s <= detrimental_thresh for s in scores):
            pathway_classifications[pathway] = 'Detrimental'
        else:
            pathway_classifications[pathway] = 'Mixed'

    print("Classification complete.")
    return pathway_classifications


# ==============================================================================
# 4. MAIN EXECUTION AND OUTPUT (Updated function call)
# ==============================================================================
if __name__ == '__main__':
    # The call now passes the single input file path
    master_df = load_all_data(INPUT_ASSOCIATION_FILE_PATH, DOMAIN_MAP_FILE, SYSTEM_DOMAIN_MAP)

    if master_df is not None and not master_df.empty:
        master_df = calculate_weights_and_direction(master_df, INVERTED_PHENOTYPES)

        final_classifications = classify_pathways(
            master_df,
            BENEFICIAL_THRESHOLD,
            DETRIMENTAL_THRESHOLD,
            print_scores=PRINT_WPP_SCORES
        )

        beneficial_list = sorted([p for p, c in final_classifications.items() if c == 'Beneficial'])
        detrimental_list = sorted([p for p, c in final_classifications.items() if c == 'Detrimental'])
        mixed_list = sorted([p for p, c in final_classifications.items() if c == 'Mixed'])

        print("\n--- Classification Summary ---")
        print(f"Beneficial Pathways: {len(beneficial_list)}")
        print(f"Detrimental Pathways: {len(detrimental_list)}")
        print(f"Mixed Pathways: {len(mixed_list)}")
        print("--------------------------\n")

        pd.DataFrame({'Pathway_Feature': beneficial_list}).to_csv(BENEFICIAL_OUTPUT_FILE, index=False)
        print(f"Saved Beneficial pathways to '{BENEFICIAL_OUTPUT_FILE}'")

        pd.DataFrame({'Pathway_Feature': detrimental_list}).to_csv(DETRIMENTAL_OUTPUT_FILE, index=False)
        print(f"Saved Detrimental pathways to '{DETRIMENTAL_OUTPUT_FILE}'")

        pd.DataFrame({'Pathway_Feature': mixed_list}).to_csv(MIXED_OUTPUT_FILE, index=False)
        print(f"Saved Mixed pathways to '{MIXED_OUTPUT_FILE}'")

    else:
        print("Execution halted due to data loading errors or no data found.")
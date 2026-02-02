import pandas as pd
import numpy as np
import os

# ==============================================================================
# 1. CONFIGURATION SECTION
# ==============================================================================
INPUT_FILE = "home/ec2-user/Studies/Oral_HPP/oral_data/regression_result/pathway_phenotype_regression_results_corrected.csv"
DOMAIN_MAP_FILE = '../oral_features_classfication/feature.csv'
OUTPUT_FILE = './key_pathway_features.csv'

TOP_N_FEATURES_PER_SYSTEM = 5
P_VALUE_THRESHOLD = 0.05

SYSTEM_DOMAIN_MAP = {
    'body': [0, 1, 2],
    'cgm': [4, 5, 6, 7],
    'liver': [8, 9, 10]
}


# ==============================================================================
# 2. DATA LOADING AND PREPARATION
# ==============================================================================
def load_domain_mapping(domain_map_file):
    """Load phenotype to domain and system mapping."""
    domain_df = pd.read_csv(domain_map_file)
    phenotype_to_system = {}
    for system, domain_ids in SYSTEM_DOMAIN_MAP.items():
        for domain_id in domain_ids:
            for feature in domain_df[domain_df['group'] == domain_id]['Feature']:
                phenotype_to_system[feature] = system
    return phenotype_to_system


def load_and_prepare_data(input_file, phenotype_to_system):
    """Load regression results and add system information."""
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return None
    
    df = pd.read_csv(input_file)
    
    if 'p_corrected_bonferroni' not in df.columns:
        print(f"Error: 'p_corrected_bonferroni' column not found")
        return None
    
    if 'Pathway_Feature' not in df.columns:
        print(f"Error: 'Pathway_Feature' column not found")
        return None
    
    if 'Phenotype_Feature' not in df.columns:
        print(f"Error: 'Phenotype_Feature' column not found")
        return None
    
    df_significant = df[df['p_corrected_bonferroni'] < P_VALUE_THRESHOLD].copy()
    df_significant['system'] = df_significant['Phenotype_Feature'].map(phenotype_to_system)
    df_significant = df_significant.dropna(subset=['system'])
    
    return df_significant


# ==============================================================================
# 3. ASSOCIATION BREADTH CALCULATION
# ==============================================================================
def calculate_association_breadth(df):
    """Calculate association breadth for each feature within each system."""
    breadth_df = df.groupby(['Pathway_Feature', 'system'])['Phenotype_Feature'].nunique().reset_index()
    breadth_df.columns = ['Pathway_Feature', 'system', 'association_count']
    return breadth_df


def rank_features_by_breadth(breadth_df):
    """Rank features within each system by association count (descending)."""
    return breadth_df.sort_values(['system', 'association_count'], ascending=[True, False])


def select_top_features(ranked_df, top_n=5):
    """Select top N features per system."""
    top_features = []
    for system in ranked_df['system'].unique():
        system_df = ranked_df[ranked_df['system'] == system]
        top_system_features = system_df.head(top_n)['Pathway_Feature'].tolist()
        top_features.extend(top_system_features)
    return top_features


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
def identify_key_pathway_features():
    """Main function to identify key pathway features."""
    print("=" * 80)
    print("Identifying Key Pathway Features")
    print("=" * 80)
    
    print("\nStep 1: Loading phenotype domain mapping...")
    phenotype_to_system = load_domain_mapping(DOMAIN_MAP_FILE)
    print(f"  Mapped {len(phenotype_to_system)} phenotypes to systems")
    
    print("\nStep 2: Loading and preparing pathway data...")
    pathway_df = load_and_prepare_data(INPUT_FILE, phenotype_to_system)
    
    if pathway_df is None or pathway_df.empty:
        print("  No pathway data available. Exiting.")
        return
    
    print(f"  Loaded {len(pathway_df)} significant pathway associations")
    
    print("\nStep 3: Calculating association breadth...")
    pathway_breadth = calculate_association_breadth(pathway_df)
    print(f"  Found {len(pathway_breadth)} feature-system combinations")
    
    print("\nStep 4: Ranking features by association count...")
    pathway_ranked = rank_features_by_breadth(pathway_breadth)
    
    print("\nStep 5: Selecting top features per system...")
    pathway_top = select_top_features(pathway_ranked, top_n=TOP_N_FEATURES_PER_SYSTEM)
    
    key_features = []
    for feature in pathway_top:
        feature_info = pathway_ranked[pathway_ranked['Pathway_Feature'] == feature].iloc[0]
        key_features.append({
            'Pathway_Feature': feature,
            'System': feature_info['system'],
            'Association_Count': feature_info['association_count']
        })
    
    print(f"  Selected {len(pathway_top)} top pathway features")
    
    print("\nStep 6: De-duplicating features across systems...")
    key_features_df = pd.DataFrame(key_features)
    
    deduplicated = key_features_df.groupby('Pathway_Feature').agg({
        'System': lambda x: ', '.join(sorted(set(x))),
        'Association_Count': 'max'
    }).reset_index()
    
    deduplicated = deduplicated.sort_values('Association_Count', ascending=False)
    
    print(f"\n--- Summary ---")
    print(f"Total key pathway features: {len(deduplicated)}")
    print(f"  - Body: {len(deduplicated[deduplicated['System'].str.contains('body')])}")
    print(f"  - CGM: {len(deduplicated[deduplicated['System'].str.contains('cgm')])}")
    print(f"  - Liver: {len(deduplicated[deduplicated['System'].str.contains('liver')])}")
    
    deduplicated.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to '{OUTPUT_FILE}'")
    
    return deduplicated


if __name__ == '__main__':
    identify_key_pathway_features()


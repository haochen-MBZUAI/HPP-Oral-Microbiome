import pandas as pd
import numpy as np
import os

# ==============================================================================
# 1. CONFIGURATION SECTION
# ==============================================================================
# Input: Post-association pruned results (from post_association_pruning.py)
INPUT_FILE = "home/ec2-user/Studies/Oral_HPP/oral_data/gene_family_pruned/gene_family_pruned_associations.csv"
DOMAIN_MAP_FILE = '../oral_features_classfication/feature.csv'
OUTPUT_FILE = './key_gene_family_features.csv'

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


def load_pruned_results(input_file, phenotype_to_system):
    """
    Load post-association pruned gene family results.
    Note: This uses the pruned set obtained after correlation-based clumping (Methods 4.5).
    """
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return None
    
    try:
        df = pd.read_csv(input_file)
        
        if df.empty:
            print("Error: Pruned results file is empty")
            return None
        
        # Pruned results should have Feature_Col and Phenotype_Col columns from post_association_pruning.py
        # Or we need to detect them
        if 'Feature_Col' in df.columns and 'Phenotype_Col' in df.columns:
            feature_col_name = df['Feature_Col'].iloc[0]
            phenotype_col_name = df['Phenotype_Col'].iloc[0]
        else:
            # Try to detect columns
            feature_col = [col for col in df.columns if 'Gene_Family' in col or 'Pathway_Feature' in col]
            phenotype_col = [col for col in df.columns if 'Phenotype_Feature' in col]
            
            if not feature_col or not phenotype_col:
                print("Error: Could not find feature or phenotype columns")
                print(f"Available columns: {list(df.columns)}")
                return None
            
            feature_col_name = feature_col[0]
            phenotype_col_name = phenotype_col[0]
            df['Feature_Col'] = feature_col_name
            df['Phenotype_Col'] = phenotype_col_name
        
        # Add system information
        df['system'] = df[phenotype_col_name].map(phenotype_to_system)
        df = df.dropna(subset=['system'])
        
        if df.empty:
            print("Error: No valid associations after mapping to systems")
            return None
        
        return df
    
    except Exception as e:
        print(f"Error loading pruned results: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# 3. ASSOCIATION BREADTH CALCULATION
# ==============================================================================
def calculate_association_breadth(df):
    """Calculate association breadth for each feature within each system."""
    feature_col = df['Feature_Col'].iloc[0]
    phenotype_col = df['Phenotype_Col'].iloc[0]
    
    breadth_df = df.groupby([feature_col, 'system'])[phenotype_col].nunique().reset_index()
    breadth_df = breadth_df.rename(columns={feature_col: 'Gene_Family_Feature'})
    breadth_df.columns = ['Gene_Family_Feature', 'system', 'association_count']
    return breadth_df


def rank_features_by_breadth(breadth_df):
    """Rank features within each system by association count (descending)."""
    return breadth_df.sort_values(['system', 'association_count'], ascending=[True, False])


def select_top_features(ranked_df, top_n=5):
    """Select top N features per system."""
    top_features = []
    for system in ranked_df['system'].unique():
        system_df = ranked_df[ranked_df['system'] == system]
        top_system_features = system_df.head(top_n)['Gene_Family_Feature'].tolist()
        top_features.extend(top_system_features)
    return top_features


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
def identify_key_gene_family_features():
    """Main function to identify key gene family features."""
    print("=" * 80)
    print("Identifying Key Gene Family Features")
    print("=" * 80)
    
    print("\nStep 1: Loading phenotype domain mapping...")
    phenotype_to_system = load_domain_mapping(DOMAIN_MAP_FILE)
    print(f"  Mapped {len(phenotype_to_system)} phenotypes to systems")
    
    print("\nStep 2: Loading post-association pruned gene family data...")
    print("  Note: Using pruned set obtained after correlation-based clumping")
    gene_family_df = load_pruned_results(INPUT_FILE, phenotype_to_system)
    
    if gene_family_df is None or gene_family_df.empty:
        print("  No gene family data available. Exiting.")
        return
    
    print(f"  Loaded {len(gene_family_df)} significant gene family associations")
    
    print("\nStep 3: Calculating association breadth...")
    gene_family_breadth = calculate_association_breadth(gene_family_df)
    print(f"  Found {len(gene_family_breadth)} feature-system combinations")
    
    print("\nStep 4: Ranking features by association count...")
    gene_family_ranked = rank_features_by_breadth(gene_family_breadth)
    
    print("\nStep 5: Selecting top features per system...")
    gene_family_top = select_top_features(gene_family_ranked, top_n=TOP_N_FEATURES_PER_SYSTEM)
    
    key_features = []
    for feature in gene_family_top:
        feature_info = gene_family_ranked[gene_family_ranked['Gene_Family_Feature'] == feature].iloc[0]
        key_features.append({
            'Gene_Family_Feature': feature,
            'System': feature_info['system'],
            'Association_Count': feature_info['association_count']
        })
    
    print(f"  Selected {len(gene_family_top)} top gene family features")
    
    print("\nStep 6: De-duplicating features across systems...")
    key_features_df = pd.DataFrame(key_features)
    
    deduplicated = key_features_df.groupby('Gene_Family_Feature').agg({
        'System': lambda x: ', '.join(sorted(set(x))),
        'Association_Count': 'max'
    }).reset_index()
    
    deduplicated = deduplicated.sort_values('Association_Count', ascending=False)
    
    print(f"\n--- Summary ---")
    print(f"Total key gene family features: {len(deduplicated)}")
    print(f"  - Body: {len(deduplicated[deduplicated['System'].str.contains('body')])}")
    print(f"  - CGM: {len(deduplicated[deduplicated['System'].str.contains('cgm')])}")
    print(f"  - Liver: {len(deduplicated[deduplicated['System'].str.contains('liver')])}")
    
    deduplicated.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to '{OUTPUT_FILE}'")
    
    return deduplicated


if __name__ == '__main__':
    identify_key_gene_family_features()


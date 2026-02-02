import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import spearmanr
from tqdm import tqdm
import pyarrow.feather as feather


# ==============================================================================
# 1. CONFIGURATION SECTION
# ==============================================================================
CORRECTED_RESULTS_DIR = "home/ec2-user/Studies/Oral_HPP/oral_data/regression_result/gene_family_phenotype_regression_results_corrected"
GENE_FAMILY_DATA_DIR = "home/ec2-user/Studies/Oral_HPP/oral_data/gene_family_processed"
PHENOTYPE_DATA_FILE = "home/ec2-user/Studies/Oral_HPP/oral_data/phenotype_cleaned.csv"
OUTPUT_DIR = "home/ec2-user/Studies/Oral_HPP/oral_data/gene_family_pruned"

P_VALUE_THRESHOLD = 0.05
CORRELATION_THRESHOLD = 0.30


# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
def load_corrected_results(results_dir):
    """Load all Bonferroni-corrected regression results."""
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files found in {results_dir}")
        return None
    
    all_results = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            if 'p_corrected_bonferroni' not in df.columns:
                continue
            
            # Find feature and phenotype columns
            # Gene family results use 'Pathway_Feature' (which contains UniRef90 names)
            feature_col = [col for col in df.columns if 'Pathway_Feature' in col or ('UniRef90' in str(col) and 'Feature' in str(col))]
            phenotype_col = [col for col in df.columns if 'Phenotype_Feature' in col]
            
            if not feature_col or not phenotype_col:
                continue
            
            # Check for required columns
            if 'Predictor_Coeff' not in df.columns or 'p_corrected_bonferroni' not in df.columns:
                continue
            
            df['Feature_Col'] = feature_col[0]
            df['Phenotype_Col'] = phenotype_col[0]
            all_results.append(df)
        except Exception as e:
            print(f"Warning: Error loading {csv_file}: {e}")
            continue
    
    if not all_results:
        return None
    
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df


def load_gene_family_abundances(data_dir):
    """Load gene family abundance data."""
    arrow_files = glob.glob(os.path.join(data_dir, "*.arrow"))
    
    if not arrow_files:
        print(f"Error: No Arrow files found in {data_dir}")
        return None
    
    all_data = {}
    for arrow_file in tqdm(arrow_files, desc="Loading gene family data"):
        try:
            table = feather.read_table(arrow_file)
            df = table.to_pandas()
            
            if df.empty:
                continue
            
            # First column is participant_id, second is collection_data
            id_col = df.columns[0]
            for feature_col in df.columns[2:]:
                # Store as DataFrame with ID and feature column
                feature_data = df[[id_col, feature_col]].copy()
                all_data[feature_col] = feature_data
        except Exception as e:
            print(f"Warning: Error loading {arrow_file}: {e}")
            continue
    
    print(f"Loaded {len(all_data)} gene families")
    return all_data


def load_phenotype_data(phenotype_file):
    """Load phenotype data."""
    try:
        df = pd.read_csv(phenotype_file)
        return df
    except Exception as e:
        print(f"Error loading phenotype data: {e}")
        return None


# ==============================================================================
# 3. PRUNING FUNCTIONS
# ==============================================================================
def compute_standardized_effect_size(beta, gene_family_abundance, phenotype_values):
    """Compute standardized effect size: S = |beta| * SD(X_g) / SD(Y_k)."""
    sd_gene = np.std(gene_family_abundance, ddof=1)
    sd_phenotype = np.std(phenotype_values, ddof=1)
    
    if sd_phenotype == 0:
        return 0.0
    
    return abs(beta) * sd_gene / sd_phenotype


def compute_joint_rank(p_values, effect_sizes):
    """Compute joint rank: R = rank(p_adj) + rank(-S)."""
    rank_p = pd.Series(p_values).rank(method='min')
    rank_s = pd.Series(-effect_sizes).rank(method='min')
    return rank_p + rank_s


def prune_phenotype_associations(df_phenotype, gene_family_data, phenotype_data, phenotype_name):
    """Prune associations for a single phenotype."""
    # Filter significant associations
    df_sig = df_phenotype[df_phenotype['p_corrected_bonferroni'] < P_VALUE_THRESHOLD].copy()
    
    if df_sig.empty:
        return []
    
    feature_col = df_sig['Feature_Col'].iloc[0]
    
    # Compute standardized effect sizes
    effect_sizes = []
    valid_indices = []
    
    for idx, row in df_sig.iterrows():
        feature_name = row[feature_col]
        
        if feature_name not in gene_family_data:
            continue
        
        # Get gene family abundance
        gene_df = gene_family_data[feature_name]
        id_col = gene_df.columns[0]
        
        # Merge with phenotype data (phenotype data should have participant_id column)
        # Check if phenotype data has the same ID column name
        if id_col not in phenotype_data.columns:
            # Try common ID column names
            if 'participant_id' in phenotype_data.columns:
                gene_df = gene_df.rename(columns={id_col: 'participant_id'})
                id_col = 'participant_id'
            else:
                continue
        
        merged = pd.merge(gene_df, phenotype_data, on=id_col, how='inner')
        
        if len(merged) < 10:  # Need minimum samples
            continue
        
        gene_abundance = merged[feature_name].values
        phenotype_values = merged[phenotype_name].values
        
        # Remove NaN
        valid_mask = ~(np.isnan(gene_abundance) | np.isnan(phenotype_values))
        if valid_mask.sum() < 10:
            continue
        
        gene_abundance = gene_abundance[valid_mask]
        phenotype_values = phenotype_values[valid_mask]
        
        # Compute effect size
        beta = row.get('Predictor_Coeff', 0.0)
        if np.isnan(beta):
            continue
        
        effect_size = compute_standardized_effect_size(beta, gene_abundance, phenotype_values)
        effect_sizes.append(effect_size)
        valid_indices.append(idx)
    
    if not valid_indices:
        return []
    
    df_valid = df_sig.loc[valid_indices].copy()
    df_valid['Effect_Size'] = effect_sizes
    
    # Compute joint ranks
    df_valid['Joint_Rank'] = compute_joint_rank(
        df_valid['p_corrected_bonferroni'].values,
        df_valid['Effect_Size'].values
    )
    
    # Iterative selection
    selected_features = []
    remaining_df = df_valid.copy()
    
    while not remaining_df.empty:
        # Select top-ranked feature
        top_idx = remaining_df['Joint_Rank'].idxmin()
        top_feature = remaining_df.loc[top_idx, feature_col]
        selected_features.append(top_idx)
        
        # Remove selected feature
        remaining_df = remaining_df.drop(top_idx)
        
        if remaining_df.empty:
            break
        
        # Compute correlations with selected feature
        top_feature_name = top_feature
        if top_feature_name not in gene_family_data:
            continue
        
        top_gene_df = gene_family_data[top_feature_name]
        id_col_top = top_gene_df.columns[0]
        
        # Normalize ID column name if needed
        if id_col_top not in phenotype_data.columns and 'participant_id' in phenotype_data.columns:
            top_gene_df = top_gene_df.rename(columns={id_col_top: 'participant_id'})
            id_col_top = 'participant_id'
        
        top_abundance = top_gene_df.set_index(id_col_top)[top_feature_name]
        
        # Check correlations with remaining features
        to_remove = []
        for idx, row in remaining_df.iterrows():
            feature_name = row[feature_col]
            
            if feature_name not in gene_family_data:
                continue
            
            gene_df = gene_family_data[feature_name]
            id_col_gene = gene_df.columns[0]
            
            # Normalize ID column name if needed
            if id_col_gene not in phenotype_data.columns and 'participant_id' in phenotype_data.columns:
                gene_df = gene_df.rename(columns={id_col_gene: 'participant_id'})
                id_col_gene = 'participant_id'
            
            gene_abundance = gene_df.set_index(id_col_gene)[feature_name]
            
            # Find common samples
            common_ids = top_abundance.index.intersection(gene_abundance.index)
            if len(common_ids) < 10:
                continue
            
            top_vals = top_abundance.loc[common_ids].values
            gene_vals = gene_abundance.loc[common_ids].values
            
            # Remove NaN
            valid_mask = ~(np.isnan(top_vals) | np.isnan(gene_vals))
            if valid_mask.sum() < 10:
                continue
            
            top_vals = top_vals[valid_mask]
            gene_vals = gene_vals[valid_mask]
            
            # Compute Spearman correlation
            try:
                corr, _ = spearmanr(top_vals, gene_vals)
                if not np.isnan(corr) and abs(corr) >= CORRELATION_THRESHOLD:
                    to_remove.append(idx)
            except:
                pass
        
        # Remove correlated features
        if to_remove:
            remaining_df = remaining_df.drop(to_remove)
    
    return df_valid.loc[selected_features]


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
def main():
    """Main execution function."""
    print("=" * 80)
    print("Post-association Pruning for Gene Families")
    print("=" * 80)
    
    # Load data
    print("\nStep 1: Loading corrected regression results...")
    df_results = load_corrected_results(CORRECTED_RESULTS_DIR)
    
    if df_results is None or df_results.empty:
        print("Error: Could not load corrected results.")
        return
    
    print(f"Loaded {len(df_results)} association results")
    
    print("\nStep 2: Loading gene family abundance data...")
    gene_family_data = load_gene_family_abundances(GENE_FAMILY_DATA_DIR)
    
    if not gene_family_data:
        print("Error: Could not load gene family data.")
        return
    
    print(f"Loaded {len(gene_family_data)} gene families")
    
    print("\nStep 3: Loading phenotype data...")
    phenotype_df = load_phenotype_data(PHENOTYPE_DATA_FILE)
    
    if phenotype_df is None:
        print("Error: Could not load phenotype data.")
        return
    
    print(f"Loaded phenotype data with {len(phenotype_df)} samples")
    
    # Get unique phenotypes
    print("\nStep 4: Pruning associations for each phenotype...")
    feature_col = df_results['Feature_Col'].iloc[0]
    phenotype_col = df_results['Phenotype_Col'].iloc[0]
    unique_phenotypes = df_results[phenotype_col].unique()
    
    all_pruned_results = []
    
    for phenotype in tqdm(unique_phenotypes, desc="Processing phenotypes"):
        df_pheno = df_results[df_results[phenotype_col] == phenotype].copy()
        
        if df_pheno.empty:
            continue
        
        pruned = prune_phenotype_associations(
            df_pheno, gene_family_data, phenotype_df, phenotype
        )
        
        if not pruned.empty:
            all_pruned_results.append(pruned)
    
    if not all_pruned_results:
        print("No pruned associations found.")
        return
    
    # Combine results
    df_final = pd.concat(all_pruned_results, ignore_index=True)
    
    # Save results
    print("\nStep 5: Saving pruned results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_file = os.path.join(OUTPUT_DIR, "gene_family_pruned_associations.csv")
    df_final.to_csv(output_file, index=False)
    
    print(f"\n--- Summary ---")
    print(f"Original significant associations: {len(df_results[df_results['p_corrected_bonferroni'] < P_VALUE_THRESHOLD])}")
    print(f"Pruned associations: {len(df_final)}")
    print(f"Reduction: {100 * (1 - len(df_final) / len(df_results[df_results['p_corrected_bonferroni'] < P_VALUE_THRESHOLD])):.2f}%")
    print(f"Unique phenotypes: {df_final[phenotype_col].nunique()}")
    print(f"Unique gene families: {df_final[feature_col].nunique()}")
    
    print(f"\nResults saved to '{output_file}'")


if __name__ == '__main__':
    main()


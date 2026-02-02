import pandas as pd
import numpy as np
import os
import pyarrow.feather as feather
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from tqdm import tqdm
import random


# ==============================================================================
# 1. CONFIGURATION SECTION
# ==============================================================================
INPUT_DIR = "home/ec2-user/Studies/Oral_HPP/oral_data/gene_family_processed"
OUTPUT_DIR = "home/ec2-user/Studies/Oral_HPP/oral_data/gene_family_decorrelated"

BATCH_SIZE = 3000
CORRELATION_CUTOFF = 0.3
TOP_PERCENT = 0.05
N_ITERATIONS = 3
RANDOM_SEED = 42


# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
def load_gene_family_data(input_dir):
    """Load all gene family Arrow files and combine them."""
    arrow_files = [f for f in os.listdir(input_dir) if f.endswith('.arrow')]
    
    if not arrow_files:
        print(f"Error: No Arrow files found in {input_dir}")
        return None
    
    all_data = []
    for arrow_file in tqdm(arrow_files, desc="Loading Arrow files"):
        file_path = os.path.join(input_dir, arrow_file)
        try:
            table = feather.read_table(file_path)
            df = table.to_pandas()
            
            if df.empty:
                continue
            
            # First two columns are ID and collection_data
            if df.shape[1] < 3:
                continue
            
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Error loading {arrow_file}: {e}")
            continue
    
    if not all_data:
        return None
    
    # Combine all dataframes
    # All Arrow files should have the same ID and collection_data column names
    if not all_data:
        return None
    
    combined_df = all_data[0]
    id_cols = combined_df.columns[:2].tolist()  # participant_id and collection_data
    
    for df in all_data[1:]:
        # Ensure same ID column names
        if df.columns[:2].tolist() != id_cols:
            # Rename to match
            df = df.rename(columns={df.columns[0]: id_cols[0], df.columns[1]: id_cols[1]})
        
        # Merge on ID columns, keeping all features
        combined_df = pd.merge(combined_df, df, on=id_cols, how='outer', suffixes=('', '_dup'))
    
    return combined_df


# ==============================================================================
# 3. DECORRELATION FUNCTIONS
# ==============================================================================
def compute_spearman_correlation_matrix(df_features):
    """Compute pairwise Spearman correlation matrix."""
    # Use pandas for efficiency
    corr_matrix = df_features.corr(method='spearman').values
    return corr_matrix


def hierarchical_clustering(corr_matrix, threshold=0.3):
    """Perform average-linkage hierarchical clustering with correlation threshold."""
    # Convert correlation to distance: d = 1 - |rho|
    distance_matrix = 1 - np.abs(corr_matrix)
    
    # Convert to condensed form for linkage
    condensed_distances = squareform(distance_matrix, checks=False)
    
    # Average linkage clustering
    linkage_matrix = linkage(condensed_distances, method='average')
    
    # Form clusters at threshold
    clusters = fcluster(linkage_matrix, 1 - threshold, criterion='distance')
    
    return clusters


def select_cluster_representative(df_features, cluster_labels, cluster_id):
    """Select representative feature for a cluster."""
    cluster_mask = cluster_labels == cluster_id
    cluster_features = df_features.columns[cluster_mask]
    
    if len(cluster_features) == 0:
        return None
    
    # Rank by mean abundance and select top 5%
    mean_abundances = df_features[cluster_features].mean()
    n_top = max(1, int(len(cluster_features) * TOP_PERCENT))
    top_features = mean_abundances.nlargest(n_top).index.tolist()
    
    if len(top_features) == 1:
        return top_features[0]
    
    # Compute pairwise distances among top features
    top_df = df_features[top_features]
    corr_matrix = top_df.corr(method='spearman').abs()
    distance_matrix = 1 - corr_matrix.values
    
    # Select feature with smallest mean distance to all others
    mean_distances = distance_matrix.mean(axis=1)
    representative_idx = np.argmin(mean_distances)
    
    return top_features[representative_idx]


def process_batch(df_batch, batch_id):
    """Process a single batch of features."""
    print(f"  Processing batch {batch_id} ({len(df_batch.columns)} features)...")
    
    # Compute correlation matrix
    corr_matrix = compute_spearman_correlation_matrix(df_batch)
    
    # Hierarchical clustering
    cluster_labels = hierarchical_clustering(corr_matrix, threshold=CORRELATION_CUTOFF)
    
    # Select representatives from each cluster
    representatives = []
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        rep = select_cluster_representative(df_batch, cluster_labels, cluster_id)
        if rep is not None:
            representatives.append(rep)
    
    print(f"    Selected {len(representatives)} representatives from {len(unique_clusters)} clusters")
    return representatives


def decorrelate_features(df_features, n_iterations=3):
    """Main decorrelation function with iterative refinement."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    current_features = df_features.copy()
    
    for iteration in range(n_iterations):
        print(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")
        print(f"Starting with {len(current_features.columns)} features")
        
        # Randomly partition into batches
        feature_list = list(current_features.columns)
        random.shuffle(feature_list)
        
        batches = []
        for i in range(0, len(feature_list), BATCH_SIZE):
            batches.append(feature_list[i:i + BATCH_SIZE])
        
        print(f"Partitioned into {len(batches)} batches")
        
        # Process each batch
        all_representatives = []
        for batch_id, batch_features in enumerate(batches):
            df_batch = current_features[batch_features]
            representatives = process_batch(df_batch, batch_id + 1)
            all_representatives.extend(representatives)
        
        # Update current features
        if len(all_representatives) == 0:
            print("Warning: No representatives selected. Stopping.")
            break
        
        # Use original df_features to get the selected representatives
        current_features = df_features[all_representatives].copy()
        print(f"After iteration {iteration + 1}: {len(current_features.columns)} features retained")
        
        # Check convergence
        if iteration < n_iterations - 1:
            # Compute final correlation statistics
            corr_matrix = current_features.corr(method='spearman').abs()
            np.fill_diagonal(corr_matrix.values, np.nan)
            mean_corr = corr_matrix.values[~np.isnan(corr_matrix.values)].mean()
            std_corr = corr_matrix.values[~np.isnan(corr_matrix.values)].std()
            print(f"  Mean absolute correlation: {mean_corr:.4f} (SD: {std_corr:.4f})")
    
    return current_features


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
def main():
    """Main execution function."""
    print("=" * 80)
    print("Pre-association Decorrelation Filter for Gene Families")
    print("=" * 80)
    
    # Load data
    print("\nStep 1: Loading gene family data...")
    df_all = load_gene_family_data(INPUT_DIR)
    
    if df_all is None or df_all.empty:
        print("Error: Could not load gene family data.")
        return
    
    print(f"Loaded data with shape: {df_all.shape}")
    
    # Separate ID columns from feature columns
    id_cols = df_all.columns[:2].tolist()
    df_ids = df_all[id_cols].copy()
    df_features = df_all.iloc[:, 2:].copy()
    
    print(f"Number of gene family features: {len(df_features.columns)}")
    
    # Decorrelate
    print("\nStep 2: Applying decorrelation filter...")
    df_decorrelated = decorrelate_features(df_features, n_iterations=N_ITERATIONS)
    
    # Combine with ID columns
    df_final = pd.concat([df_ids, df_decorrelated], axis=1)
    
    # Save results
    print("\nStep 3: Saving decorrelated features...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_file = os.path.join(OUTPUT_DIR, "gene_family_decorrelated.arrow")
    table = pyarrow.Table.from_pandas(df_final, preserve_index=False)
    feather.write_feather(table, output_file)
    
    print(f"\n--- Summary ---")
    print(f"Original features: {len(df_features.columns)}")
    print(f"Retained features: {len(df_decorrelated.columns)}")
    print(f"Reduction: {100 * (1 - len(df_decorrelated.columns) / len(df_features.columns)):.2f}%")
    
    # Final correlation statistics
    final_corr = df_decorrelated.corr(method='spearman').abs()
    np.fill_diagonal(final_corr.values, np.nan)
    mean_corr = final_corr.values[~np.isnan(final_corr.values)].mean()
    std_corr = final_corr.values[~np.isnan(final_corr.values)].std()
    print(f"Final mean absolute correlation: {mean_corr:.4f} (SD: {std_corr:.4f})")
    
    print(f"\nResults saved to '{output_file}'")


if __name__ == '__main__':
    import pyarrow
    main()


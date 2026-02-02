# Mining Gene Family

Two-step approach to handle high inter-correlation and functional redundancy in gene family data.

## Scripts

### `pre_association_decorrelation.py`
Pre-association decorrelation filter applied before association testing.

**Process:**
1. Randomly partition gene families into batches (up to 3,000 features per batch)
2. Compute pairwise Spearman correlations within each batch
3. Apply average-linkage hierarchical clustering (correlation cutoff = 0.3)
4. For each cluster: rank by mean abundance, select top 5%, choose representative
5. Iterate for 3 rounds to further suppress redundancy

**Input:**
- `gene_family_processed/*.arrow` files

**Output:**
- `gene_family_decorrelated.arrow` (reduced, uncorrelated feature set)

---

### `post_association_pruning.py`
Post-association pruning applied after Bonferroni correction to identify independent associations.

**Process:**
1. For each phenotype, retain gene families with p_adj < 0.05
2. Compute standardized effect size: S = |β| * SD(X_g) / SD(Y_k)
3. Compute joint rank: R = rank(p_adj) + rank(-S)
4. Iteratively select top-ranked feature, remove correlated features (|ρ| ≥ 0.30)

**Input:**
- Bonferroni-corrected regression results (directory)
- `gene_family_processed/*.arrow` files
- `phenotype_cleaned.csv`

**Output:**
- `gene_family_pruned_associations.csv` (independent associations per phenotype)

---

## Configuration

Update file paths in each script before running.


# Association Analysis

This module performs regression analysis between oral microbiome features and metabolic phenotypes.

## Scripts

### `analyse_strain.py`
Runs OLS regression between strain-level features and phenotypes, controlling for age, sex, and smoking. 

**Input:**
- `strain_processed.csv`
- `phenotype_cleaned.csv`

**Output:**
- `strain_phenotype_regression_results.csv`

---

### `analyse_pathway.py`
Runs OLS regression between pathway features and phenotypes, controlling for age, sex, and smoking. Merges data with 180-day time window.

**Input:**
- `pathway_processed.csv`
- `phenotype_cleaned.csv`

**Output:**
- `pathway_phenotype_regression_results.csv`

---

### `analyse_gene_family.py`
Runs OLS regression between UniRef90 gene family features and phenotypes, controlling for age, sex, and smoking. Processes all `.arrow` files in the input directory.

**Input:**
- `gene_family_processed/*.arrow`
- `phenotype_cleaned.csv`

**Output:**
- `gene_family_phenotype_regression_results.csv`

---

### `correct_P_value_strain_pathway.py`
Applies Bonferroni correction to regression p-values for strain and pathway results. Auto-detects feature column name.

**Input:**
- `strain_phenotype_regression_results.csv` (default)
- `pathway_phenotype_regression_results.csv` (use `-i` to specify)

**Output:**
- `*_corrected.csv`

---

### `correct_P_value_gene_family.py`
Applies Bonferroni correction to regression p-values for gene family results.

**Input:**
- `gene_family_phenotype_regression_results.csv`

**Output:**
- `gene_family_phenotype_regression_results_corrected.csv`

---

## Workflow

1. Run regression analysis: `analyse_strain.py`, `analyse_pathway.py`, or `analyse_gene_family.py`
2. Apply p-value correction: `correct_P_value_*.py`
3. Visualize results: See `visulizaion/` directory

---

## Configuration

Update hardcoded file paths in each script before running.

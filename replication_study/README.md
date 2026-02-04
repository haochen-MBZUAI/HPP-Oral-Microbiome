# Replication Study

This module validates BMI and waist circumference associations with oral microbial strains from the main analysis in an independent replication dataset.

## Analysis Methods

The replication study uses **the same preprocessing and analysis methods** as the main analysis:

- **Preprocessing**: Same transformation pipeline (zero-replacement → normalization → PPM → Log₁₀) for microbiome data; same outlier removal (>8 SD) and clipping (>5 SD) for phenotype data
- **Regression analysis**: Same OLS regression model controlling for age, sex, and smoking status
- **Mapping**: Extract and normalize genus names from both datasets, then match by normalized genus name

## Running Instructions

Run scripts in order:

1. **`00_convert_xpt_merge_and_preprocess_genus.py`** – Convert SAS XPT to CSV, merge G/F files (BMX, DEMO, SMQ), merge phenotype CSVs, preprocess oral genus data, and preprocess phenotype data (outlier removal and clipping).
2. **`01_run_association.py`** – Run association analysis (BMI and waist circumference) with age, sex, and smoking as covariates.
3. **`02_calculate_mapping.py`** – Map main (strain-level) results to replication (genus-level) and compare replication rate and direction consistency.
replication_study

# Metabolic Diseases Prediction Module

This module identifies oral microbiome features associated with metabolic diseases and builds predictive models.

## Scripts

### `select_diseases_associated_oral_features.py`
Identifies phenotypes and pathways associated with metabolic diseases using ANOVA F-test with 5-fold cross-validation.

**Input:**
- `phenotype_cleaned.csv` (from `preprocess` module)
- `metabolic_diseases.csv` (disease labels)
- `pathway_phenotype_regression_results_corrected.csv` (from `association_analyse` module)

**Output:**
- `{disease}_pheno_statistic.csv` (phenotype importance scores)
- `significant_pathways_from_statistic.csv` (disease-associated pathways)

---

### `predict.py`
Builds machine learning models to predict metabolic diseases using pathway features.

**Input:**
- `pathway_processed.csv` (from `preprocess` module)
- `metabolic_diseases.csv` (disease labels)
- `significant_pathways_from_statistic.csv` (optional, from `select_diseases_associated_oral_features.py`)

**Output:**
- Console output: Accuracy, AUC, classification reports, confusion matrices (5-fold CV results)

---

## Workflow

1. Run `select_diseases_associated_oral_features.py` to identify disease-associated features
2. Run `predict.py` to build and evaluate prediction models



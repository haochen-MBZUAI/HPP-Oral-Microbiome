# Identification of Key Oral Microbial Features

Identifies key oral microbial features by ranking based on association breadth (number of significant phenotypes) within each metabolic system.

## Scripts

### `identify_key_strain.py`
Ranks strain features by association count within each system and selects top 5 per system.

**Input:**
- Bonferroni-corrected strain regression results
- `../oral_features_classfication/feature.csv` (phenotype domain mapping)

**Output:**
- `key_strain_features.csv`

---

### `identify_key_pathway.py`
Ranks pathway features by association count within each system and selects top 5 per system.

**Input:**
- Bonferroni-corrected pathway regression results
- `../oral_features_classfication/feature.csv` (phenotype domain mapping)

**Output:**
- `key_pathway_features.csv`

---

### `identify_key_gene_family.py`
Ranks gene family features by association count within each system and selects top 5 per system.

**Input:**
- `../oral_features_classfication/feature.csv` (phenotype domain mapping)

**Output:**
- `key_gene_family_features.csv`



## Configuration

Update file paths in each script before running.


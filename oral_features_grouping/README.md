
# Oral Features Classification

## Scripts

### `classification.py`
Classifies features into Favourable, Adverse, or Mixed based on association direction.

**Input:**
- Bonferroni-corrected regression results (must contain `p_corrected_bonferroni` column)
- `feature.csv` (phenotype domain mapping)

**Output:**
- `{prefix}_Favourable.csv`
- `{prefix}_Adverse.csv`
- `{prefix}_Mixed.csv`

---

## Configuration

Update file paths in each script before running.

# Data Preprocessing

## Scripts

### `phenotype.py`
Cleans phenotype data by removing outliers (>8 SD) and clipping extreme values (>5 SD).


---

### `strain_pathway.py`
Standardizes strain or pathway abundance data: zero-replacement → normalization → PPM → Log₁₀.


image.png
---

### `gene_family.py`
Processes UniRef90 gene family abundance files (Arrow format). Same standardization pipeline as `strain_pathway.py`.



---

## Configuration

Update hardcoded file paths in each script before running.


# Decoding the Oral Microbiome: Metagenomic Insights into Host Metabolic Health

> Code to reproduce the analyses in **“Decoding the Oral Microbiome: Metagenomic Insights into Host Metabolic Health”**.  

## Our Contributions:

1. **Fine granularity:** Strain‑level oral metagenomes with matched gene families and pathways (MetaPhlAn 4 + HUMAnN 3.6).
2. **Deep multi‑system phenotyping:** 44 metabolic measures across liver ultrasonography, CGM, and DXA in 9,431 adults.
3. **Tight design and rigor:** ±180‑day temporal alignment, covariate‑adjusted OLS models, and Bonferroni control per feature layer.
4. **Biological insight:** Strain signatures align with adiposity, while functional signatures track glycemic control; community contributions are quantified.
5. **Translational value:** Phenotype‑selected oral features improve disease‑risk prediction and show directionally consistent replication at the genus level.
<div align=center><img src="visulization/Fig1.png" width="80%" height="80%" /></div>

## Environment Setup
```
conda create -n oral_hpp python==3.11
pip install -r requirements.txt
```

## Pipeline 

1. **Preprocess (`preprocess/`)**  
   Harmonize time windows (±180 days), clean phenotypes, and transform microbiome features  
   (zero‑replacement thresholds, total‑sum normalization → PPM, log₁₀ transform).

2. **Association analysis (`association_analyse/`)**  
   OLS models adjusted for age, sex, smoking; Bonferroni control per layer (strain / gene family / pathway).

3. **Oral feature classification (`oral_features_classfication/`)**  
   Take the **significant** features and assign **beneficial / detrimental / mixed** labels by synthesizing directions across the three systems (liver, CGM, body).

4. **Metabolic disease prediction (`metabolic_diseases/`)**  
   Compare **baseline (all features)** vs **phenotype‑selected features** for classification of common metabolic risk states with cross‑validated evaluation.

---


## Data Access

### HPP (Human Phenotype Project)
- **Controlled Access**: Due to ethical and IRB requirements, HPP data is available through a controlled-access portal.
- **Access Portal**: <https://humanphenotypeproject.org/data-access>
- **Process**: Researchers must submit a statement of purpose and sign a data use agreement. Upon approval, data can be accessed in a secure environment.
- **Ethics Approval**: Weizmann Institute IRB **#1719-1**.
- **Data Scope**: This study utilizes oral swab metagenomes (MetaPhlAn 4 / HUMAnN 3.6 outputs) and 44 metabolic phenotypes, including liver ultrasound, CGM, and DXA.
- **We do not provide any controlled data in this repository**. You can run the code in this repository within the approved environment to reproduce the experiments.

# Decoding the Oral Microbiome: Metagenomic Insights into Host Metabolic Health

> Code to reproduce the analyses in **“Decoding the Oral Microbiome: Metagenomic Insights into Host Metabolic Health”**.  

## Our Contributions:

1. **Population-scale, high-resolution metagenomics with deep metabolic phenotyping:** We profile standardized bilateral buccal-swab whole-metagenome data in 9,431 HPP adults, paired with 44 metabolic measures spanning liver ultrasound, CGM, and DXA.
2. **A unified, rigorous multi-layer MWAS framework:** We systematically test associations across strain, gene-family, and pathway layers using covariate-adjusted regression and layer-wise multiple-testing control, enabling direct comparison of signals across metabolic systems.
3. **Actionable outputs with translational and external support:** We deliver a multi-system oral–metabolic association atlas with prioritized cross-phenotype markers, demonstrate proof-of-concept metabolic disease classification using phenotype-selected oral features, and provide independent directional replication at genus resolution.
<div align=center><img src="visulization/Fig1.png" width="80%" height="80%" /></div>

## Data Access

### HPP (Human Phenotype Project)
- **Controlled Access**: Due to ethical and IRB requirements, HPP data is available through a controlled-access portal.
- **Access Portal**: <https://humanphenotypeproject.org/data-access>
- **Process**: Researchers must submit a statement of purpose and sign a data use agreement. Upon approval, data can be accessed in a secure environment.
- **TRE Tutorial**: After obtaining access, please refer to [`User-guide-for-TRE.pdf`](./User-guide-for-TRE.pdf) for a detailed guide on how to use the Trusted Research Environment (TRE).
- **Ethics Approval**: Weizmann Institute IRB **#1719-1**.


## Environment Setup
```
conda create -n oral_hpp python==3.11
pip install -r requirements.txt
```

## Running Pipeline 

The analysis follows a sequential workflow where inputs and outputs are chained. While the high-level steps are outlined below, please refer to: * **Subdirectory READMEs**: Each folder contains a local `README.md` with detailed execution instructions and script-level documentation.

1. **Preprocess** (`preprocess/`)  
   Clean phenotypes; standardize strain/pathway/gene-family abundance (zero-replacement → normalization → PPM → log₁₀). 

2. **Association analysis** (`association_analyse/`)  
   OLS (age, sex, smoking); Then Bonferroni correction (`correct_P_value_*`).

3. **Key oral features** (`Identification_key_oral_features/`)  
   Rank by association breadth and take top features per system. 
4. **Oral feature grouping** (`oral_features_classfication/`)  
   Classify significant strain/pathway into **Favourable / Adverse / Mixed** from association directions across liver, CGM, body.

5. **Metabolic disease classfication** (`metabolic_diseases/`)  
   Select pathways linked to disease-related phenotypes (5-fold CV); train classifiers (LightGBM) on strain/pathway abundance; evaluate with cross-validation.

6. **Replication** (`replication_study/`)  
   In an independent cohort (NHANES): preprocess genus and phenotype, run association (BMI, waist circumference), then compare direction.

---




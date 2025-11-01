import pandas as pd
import statsmodels.formula.api as smf
from tqdm import tqdm
import sys

def run_gene_family_liver_regression(df: pd.DataFrame):
    """
    Performs linear regression analysis for gene family features, liver features, controlling for age, sex, and smoking.

    Returns:
        pd.DataFrame: A DataFrame containing all regression results.
                      Returns an empty DataFrame if critical errors occur.
    """
    print("Starting regression analysis...")
    print(f"Input DataFrame shape: {df.shape}")

    # --- 1. Define column names ---
    # !! IMPORTANT !!: Replace these placeholder strings with your actual column names.
    covariate_age_col_name = "age_at_collection"
    covariate_sex_col_name = "sex"
    covariate_smoking_col_name = "smoking"

    # Define the start and end column names for the ranges of features.
    # The code will select all columns between these two (inclusive).
    pathway_start_col_name = "1CWET2-PWY: folate transformations III (E. coli)"
    pathway_end_col_name = "VALSYN-PWY: L-valine biosynthesis"

    liver_feature_start_col_name = "attenuation_coefficient_qbox"
    liver_feature_end_col_name = "body_comp_trunk_lean_mass"

    # --- 2. Check if all specified column names exist in the DataFrame ---
    required_col_names = [
        covariate_age_col_name,
        covariate_sex_col_name,
        covariate_smoking_col_name,
        pathway_start_col_name,
        pathway_end_col_name,
        liver_feature_start_col_name,
        liver_feature_end_col_name
    ]

    missing_cols = [name for name in required_col_names if name not in df.columns]
    if missing_cols:
        print(f"Error: The following required column names were not found in the DataFrame: {missing_cols}")
        return pd.DataFrame()

    # --- 3. Get lists of column names for pathway and liver features ---
    try:
        # Get the integer index location of the start and end columns
        pathway_start_idx = df.columns.get_loc(pathway_start_col_name)
        pathway_end_idx = df.columns.get_loc(pathway_end_col_name)

        liver_feature_start_idx = df.columns.get_loc(liver_feature_start_col_name)
        liver_feature_end_idx = df.columns.get_loc(liver_feature_end_col_name)

        # Extract the full list of column names from the ranges
        original_pathway_col_names = df.columns[pathway_start_idx: pathway_end_idx + 1].tolist()
        original_liver_feature_col_names = df.columns[liver_feature_start_idx: liver_feature_end_idx + 1].tolist()

        print(f"Using '{covariate_age_col_name}' as age covariate.")
        print(f"Using '{covariate_sex_col_name}' as sex covariate.")
        print(f"Using '{covariate_smoking_col_name}' as smoking covariate.")
        print(f"Found {len(original_pathway_col_names)} gene family (pathway) features.")
        print(f"Found {len(original_liver_feature_col_names)} liver features.")

    except KeyError as e:
        print(f"Error: A specified column name could not be found during index lookup: {e}")
        return pd.DataFrame()

    if not original_pathway_col_names or not original_liver_feature_col_names:
        print("Error: No gene family (pathway) or liver feature columns found based on the provided start/end names.")
        return pd.DataFrame()

    # --- 4. Define safe column names for use in the regression formula ---
    safe_target_feature_name = 'TARGET_FEATURE'
    safe_predictor_feature_name = 'PREDICTOR_FEATURE'
    safe_age_name = 'COVARIATE_AGE'
    safe_sex_name = 'COVARIATE_SEX'
    safe_smoking_name = 'COVARIATE_SMOKING'

    results_list = []

    total_iterations = len(original_pathway_col_names) * len(original_liver_feature_col_names)
    print(f"A total of {total_iterations} regression analyses will be performed...")

    # --- 5. Iterate and perform regression analysis ---
    with tqdm(total=total_iterations, desc="Running Regressions") as pbar:
        for pathway_name_original in original_pathway_col_names:
            for liver_feature_name_original in original_liver_feature_col_names:
                try:
                    cols_to_select = [
                        liver_feature_name_original,
                        pathway_name_original,
                        covariate_age_col_name,
                        covariate_sex_col_name,
                        covariate_smoking_col_name
                    ]

                    temp_df_for_regression = df[cols_to_select].copy()

                    temp_df_for_regression.columns = [
                        safe_target_feature_name,
                        safe_predictor_feature_name,
                        safe_age_name,
                        safe_sex_name,
                        safe_smoking_name
                    ]

                    for col in temp_df_for_regression.columns:
                        temp_df_for_regression[col] = pd.to_numeric(temp_df_for_regression[col], errors='coerce')

                    temp_df_for_regression.dropna(inplace=True)

                    if len(temp_df_for_regression) >= 30:
                        formula = f'{safe_target_feature_name} ~ {safe_predictor_feature_name} + {safe_age_name} + C({safe_sex_name}) + C({safe_smoking_name})'
                        model = smf.ols(formula, data=temp_df_for_regression).fit()

                        param_predictor = model.params.get(safe_predictor_feature_name, float('nan'))
                        pval_predictor = model.pvalues.get(safe_predictor_feature_name, float('nan'))
                        param_age = model.params.get(safe_age_name, float('nan'))
                        pval_age = model.pvalues.get(safe_age_name, float('nan'))

                        sex_param_name = next((p for p in model.params.index if p.startswith(f'C({safe_sex_name})[T.')),
                                              None)
                        param_sex = model.params.get(sex_param_name, float('nan')) if sex_param_name else float('nan')
                        pval_sex = model.pvalues.get(sex_param_name, float('nan')) if sex_param_name else float('nan')

                        smoking_param_name = next(
                            (p for p in model.params.index if p.startswith(f'C({safe_smoking_name})[T.')), None)
                        param_smoking = model.params.get(smoking_param_name,
                                                         float('nan')) if smoking_param_name else float('nan')
                        pval_smoking = model.pvalues.get(smoking_param_name,
                                                         float('nan')) if smoking_param_name else float('nan')

                        results_list.append({
                            'Pathway_Feature': pathway_name_original,
                            'Liver_Feature': liver_feature_name_original,
                            'N_Observations': model.nobs,
                            'R_Squared': model.rsquared,
                            'Predictor_Coeff': param_predictor,
                            'Predictor_PValue': pval_predictor,
                            'Age_Coeff': param_age,
                            'Age_PValue': pval_age,
                            'Sex_Coeff': param_sex,
                            'Sex_PValue': pval_sex,
                            'Smoking_Coeff': param_smoking,
                            'Smoking_PValue': pval_smoking,
                            'Sex_Param_Name_Used': sex_param_name if sex_param_name else "N/A",
                            'Smoking_Param_Name_Used': smoking_param_name if smoking_param_name else "N/A",
                            'Error_Message': None
                        })
                    else:
                        results_list.append({
                            'Pathway_Feature': pathway_name_original,
                            'Liver_Feature': liver_feature_name_original,
                            'N_Observations': len(temp_df_for_regression),
                            'R_Squared': float('nan'),
                            'Predictor_Coeff': float('nan'),
                            'Predictor_PValue': float('nan'),
                            'Age_Coeff': float('nan'),
                            'Age_PValue': float('nan'),
                            'Sex_Coeff': float('nan'),
                            'Sex_PValue': float('nan'),
                            'Smoking_Coeff': float('nan'),
                            'Smoking_PValue': float('nan'),
                            'Sex_Param_Name_Used': "N/A",
                            'Smoking_Param_Name_Used': "N/A",
                            'Error_Message': 'Insufficient data after NaN drop (less than 30 observations)'
                        })

                except Exception as e_regression:
                    print(
                        f"\nRegression error: Pathway={pathway_name_original}, LiverFeature={liver_feature_name_original}. Error: {e_regression}")
                    results_list.append({
                        'Pathway_Feature': pathway_name_original,
                        'Liver_Feature': liver_feature_name_original,
                        'N_Observations': float('nan'),
                        'R_Squared': float('nan'),
                        'Predictor_Coeff': float('nan'),
                        'Predictor_PValue': float('nan'),
                        'Age_Coeff': float('nan'),
                        'Age_PValue': float('nan'),
                        'Sex_Coeff': float('nan'),
                        'Sex_PValue': float('nan'),
                        'Smoking_Coeff': float('nan'),
                        'Smoking_PValue': float('nan'),
                        'Sex_Param_Name_Used': "N/A",
                        'Smoking_Param_Name_Used': "N/A",
                        'Error_Message': str(e_regression)
                    })
                finally:
                    pbar.update(1)

    print("\nRegression analysis completed.")
    return pd.DataFrame(results_list)


pathway_phenotype_df = load_csv("home/ec2-user/Stidies/Oral_HPP/oral_data/pathway_phenotype_processed.csv")

regression_results_df = run_gene_family_liver_regression(pathway_phenotype_df)


if not regression_results_df.empty:
    print("\nRegression results summary:")
    print(regression_results_df.head())
    # You might want to save the results to a CSV file:
    regression_results_df.to_csv('/home/ec2-user/Stidies/Oral_HPP/oral_data/regression_result/pathway_phenotype_regression_results.csv')
    # print("\nResults saved to 'gene_family_liver_regression_results.csv'")
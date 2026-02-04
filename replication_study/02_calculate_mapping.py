#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Calculate Mapping and Compare Main vs Replication Results

Maps strain-level associations from main analysis to genus-level associations
in replication study, and calculates replication rate and direction consistency.
"""

import pandas as pd
import numpy as np
import os
import re

# =============================================================================
# Configuration
# =============================================================================

_BASE_DIR = os.environ.get("REPLICATION_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

# Main analysis results (strain-level, Bonferroni-corrected); set MAIN_RESULTS_FILE env or path
MAIN_RESULTS_FILE = os.environ.get(
    "MAIN_RESULTS_FILE",
    os.path.join(os.path.dirname(__file__), "..", "oral_data", "regression_result", "strain_phenotype_regression_results_corrected.csv")
)

REPLICATION_RESULTS_FILE = os.path.join(_BASE_DIR, "regression_result", "replication_association_results.csv")
TAXONOMY_MAPPING_FILE = None

OUTPUT_DIR = os.path.join(_BASE_DIR, "regression_result")
MAPPING_FILE = os.path.join(OUTPUT_DIR, "strain_to_genus_mapping.csv")
COMPARISON_FILE = os.path.join(OUTPUT_DIR, "main_vs_replication_comparison.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "replication_summary.txt")

P_VALUE_THRESHOLD = 0.05

# =============================================================================
# Functions
# =============================================================================

def extract_genus_from_strain(strain_name):
    if '|g__' in str(strain_name):
        try:
            genus = strain_name.split('|g__')[1].split('|')[0]
            return genus
        except Exception:
            return None
    return None


def extract_genus_from_taxonomy(taxonomy_str):
    if pd.isna(taxonomy_str):
        return None
    try:
        parts = str(taxonomy_str).split(';')
        genus = parts[-1] if parts else None
        return genus if genus and genus.upper() != 'NA' else None
    except Exception:
        return None


def normalize_genus(genus_name):
    if pd.isna(genus_name):
        return None
    name = str(genus_name).strip().lower()
    if name == 'na' or not name:
        return None
    if '/' in name:
        name = name.split('/')[0]
    if ('incertae_sedis' in name or '_ucg-' in name or name.endswith('_group') or
            re.match(r'^(ggb|fgb)\d*$', name)):
        return None
    if name.endswith('_unclassified'):
        name = name.rsplit('_unclassified', 1)[0]
    name = name.replace('[', '').replace(']', '')
    name = re.sub(r'_\d+[a-z]?$', '', name)
    return name if name else None


def create_strain_to_genus_mapping(main_results_df):
    print("Creating strain-to-genus mapping...")
    unique_strains = main_results_df['Strain_Feature'].unique()
    mapping = {}
    for strain in unique_strains:
        genus = extract_genus_from_strain(strain)
        if genus:
            if genus not in mapping:
                mapping[genus] = []
            mapping[genus].append(strain)
    print(f"Mapped {len(unique_strains)} strains to {len(mapping)} unique genera")
    mapping_list = []
    for genus, strains in mapping.items():
        for strain in strains:
            mapping_list.append({
                'Strain_Feature': strain,
                'Genus_Name': genus,
                'Num_Strains_Per_Genus': len(strains)
            })
    return pd.DataFrame(mapping_list)


def map_main_to_replication(main_results_df, replication_results_df, mapping_df):
    print("Mapping main analysis results to replication results...")
    main_filtered = main_results_df[
        main_results_df['Phenotype_Feature'].str.lower().str.contains('|'.join(['bmi', 'waist']), case=False, na=False)
    ].copy()
    main_filtered = main_filtered.merge(mapping_df[['Strain_Feature', 'Genus_Name']], on='Strain_Feature', how='left')

    has_taxonomy = 'Taxonomy' in replication_results_df.columns
    if not has_taxonomy:
        print("  Warning: Replication results do not have 'Taxonomy' column.")
    print(f"  Main analysis: {len(main_filtered)} strain-phenotype associations")
    print(f"  Replication dataset: {len(replication_results_df)} genus-phenotype associations")

    main_by_genus = main_filtered.groupby(['Genus_Name', 'Phenotype_Feature']).agg({
        'Predictor_Coeff': 'mean',
        'Predictor_PValue': lambda x: x.min(),
        'p_corrected_bonferroni': lambda x: x.min() if x.notna().any() else x.min()
    }).reset_index()
    main_by_genus['Genus_Normalized'] = main_by_genus['Genus_Name'].apply(normalize_genus)

    if has_taxonomy:
        replication_results_df = replication_results_df.copy()
        replication_results_df['Genus_From_Taxonomy'] = replication_results_df['Taxonomy'].apply(extract_genus_from_taxonomy)
        replication_results_df['Genus_Normalized'] = replication_results_df['Genus_From_Taxonomy'].apply(normalize_genus)
    else:
        replication_results_df = replication_results_df.copy()
        replication_results_df['Genus_Normalized'] = None

    comparison_list = []
    for idx, main_row in main_by_genus.iterrows():
        main_genus_normalized = main_row['Genus_Normalized']
        main_genus_raw = main_row['Genus_Name']
        main_phenotype = main_row['Phenotype_Feature']
        rep_phenotype = None
        if 'bmi' in main_phenotype.lower():
            rep_phenotype = 'BMXBMI'
        elif 'waist' in main_phenotype.lower():
            rep_phenotype = 'BMXWAIST'
        if not rep_phenotype or pd.isna(main_genus_normalized):
            continue
        rep_matching = replication_results_df[
            (replication_results_df['Phenotype_Feature'] == rep_phenotype) &
            (replication_results_df['Genus_Normalized'] == main_genus_normalized)
        ]
        if rep_matching.empty:
            comparison_list.append({
                'Genus_Name': main_genus_raw,
                'Genus_Normalized': main_genus_normalized,
                'Main_Phenotype': main_phenotype,
                'Replication_Phenotype': rep_phenotype,
                'Main_Coeff': main_row['Predictor_Coeff'],
                'Main_PValue': main_row['Predictor_PValue'],
                'Main_PValue_Corrected': main_row.get('p_corrected_bonferroni', main_row['Predictor_PValue']),
                'Replication_Genus_Feature': None,
                'Replication_Coeff': float('nan'),
                'Replication_PValue': float('nan'),
                'Status': 'Genus not found in replication dataset'
            })
        else:
            rep_row = rep_matching.iloc[0]
            main_coeff = main_row['Predictor_Coeff']
            main_pval = main_row.get('p_corrected_bonferroni', main_row['Predictor_PValue'])
            rep_coeff = rep_row['Predictor_Coeff']
            rep_pval = rep_row['Predictor_PValue']
            main_sig = main_pval < P_VALUE_THRESHOLD
            rep_sig = rep_pval < P_VALUE_THRESHOLD
            same_direction = (main_coeff > 0 and rep_coeff > 0) or (main_coeff < 0 and rep_coeff < 0)
            if main_sig and rep_sig and same_direction:
                status = 'Replicated'
            elif main_sig and rep_sig and not same_direction:
                status = 'Opposite direction'
            elif main_sig and not rep_sig:
                status = 'Not replicated (non-significant)'
            else:
                status = 'Not significant in main analysis'
            comparison_list.append({
                'Genus_Name': main_genus_raw,
                'Genus_Normalized': main_genus_normalized,
                'Main_Phenotype': main_phenotype,
                'Replication_Phenotype': rep_phenotype,
                'Main_Coeff': main_coeff,
                'Main_PValue': main_row['Predictor_PValue'],
                'Main_PValue_Corrected': main_pval,
                'Replication_Genus_Feature': rep_row['Genus_Feature'],
                'Replication_Coeff': rep_coeff,
                'Replication_PValue': rep_pval,
                'Direction_Consistent': same_direction,
                'Status': status
            })
    return pd.DataFrame(comparison_list)


def calculate_summary_statistics(comparison_df):
    print("Calculating summary statistics...")
    main_sig = comparison_df[comparison_df['Main_PValue_Corrected'] < P_VALUE_THRESHOLD].copy()
    if len(main_sig) == 0:
        print("  No significant associations found in main analysis")
        return {}
    replicated = main_sig[main_sig['Status'] == 'Replicated']
    opposite_dir = main_sig[main_sig['Status'] == 'Opposite direction']
    not_replicated = main_sig[main_sig['Status'] == 'Not replicated (non-significant)']
    replication_rate = len(replicated) / len(main_sig) * 100 if len(main_sig) > 0 else 0
    valid_replication = main_sig[main_sig['Direction_Consistent'].notna()]
    direction_consistency = valid_replication['Direction_Consistent'].sum() / len(valid_replication) * 100 if len(valid_replication) > 0 else 0
    summary = {
        'Total_significant_in_main': len(main_sig),
        'Replicated': len(replicated),
        'Opposite_direction': len(opposite_dir),
        'Not_replicated': len(not_replicated),
        'Replication_rate_percent': replication_rate,
        'Direction_consistency_percent': direction_consistency,
        'Total_mapped': len(comparison_df),
        'Not_found_in_replication': len(comparison_df[comparison_df['Status'].str.contains('not found', case=False, na=False)])
    }
    return summary


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Replication Study: Mapping and Comparison")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nStep 1: Loading main analysis results...")
    if not os.path.exists(MAIN_RESULTS_FILE):
        print(f"Error: Main results file not found: {MAIN_RESULTS_FILE}")
        exit(1)
    main_results_df = pd.read_csv(MAIN_RESULTS_FILE)
    print(f"Loaded {len(main_results_df)} associations from main analysis")

    print("\nStep 2: Loading replication results...")
    if not os.path.exists(REPLICATION_RESULTS_FILE):
        print(f"Error: Replication results file not found: {REPLICATION_RESULTS_FILE}")
        print("Please run 01_run_association.py first.")
        exit(1)
    replication_results_df = pd.read_csv(REPLICATION_RESULTS_FILE)
    print(f"Loaded {len(replication_results_df)} associations from replication study")

    print("\nStep 3: Creating strain-to-genus mapping...")
    mapping_df = create_strain_to_genus_mapping(main_results_df)
    mapping_df.to_csv(MAPPING_FILE, index=False)
    print(f"Mapping saved to: {MAPPING_FILE}")

    if TAXONOMY_MAPPING_FILE and os.path.exists(TAXONOMY_MAPPING_FILE) and 'Taxonomy' not in replication_results_df.columns:
        print("\nStep 4a: Adding Taxonomy column to replication results...")
        try:
            taxonomy_map = pd.read_csv(TAXONOMY_MAPPING_FILE)
            if 'Taxonomy' in taxonomy_map.columns:
                id_col = 'Genus_Feature' if 'Genus_Feature' in taxonomy_map.columns else taxonomy_map.columns[0]
                taxonomy_dict = dict(zip(taxonomy_map[id_col], taxonomy_map['Taxonomy']))
                replication_results_df['Taxonomy'] = replication_results_df['Genus_Feature'].map(taxonomy_dict)
        except Exception as e:
            print(f"  Warning: Could not add Taxonomy column: {e}")

    print("\nStep 4: Mapping and comparing results...")
    comparison_df = map_main_to_replication(main_results_df, replication_results_df, mapping_df)
    comparison_df.to_csv(COMPARISON_FILE, index=False)
    print(f"Comparison results saved to: {COMPARISON_FILE}")

    print("\nStep 5: Calculating summary statistics...")
    summary = calculate_summary_statistics(comparison_df)

    with open(SUMMARY_FILE, 'w') as f:
        f.write("Replication Study Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total associations in main analysis: {len(main_results_df)}\n")
        f.write(f"Total associations mapped: {summary.get('Total_mapped', 0)}\n")
        f.write(f"Total significant in main analysis: {summary.get('Total_significant_in_main', 0)}\n")
        f.write(f"Successfully replicated: {summary.get('Replicated', 0)}\n")
        f.write(f"Opposite direction: {summary.get('Opposite_direction', 0)}\n")
        f.write(f"Not replicated (non-significant): {summary.get('Not_replicated', 0)}\n")
        f.write(f"Not found in replication dataset: {summary.get('Not_found_in_replication', 0)}\n")
        f.write(f"\nReplication rate: {summary.get('Replication_rate_percent', 0):.2f}%\n")
        f.write(f"Direction consistency: {summary.get('Direction_consistency_percent', 0):.2f}%\n")
    print(f"\nSummary saved to: {SUMMARY_FILE}")

    print("\n" + "=" * 80)
    print("Replication Summary")
    print("=" * 80)
    for key, value in summary.items():
        print(f"{key}: {value:.2f}%" if 'percent' in key else f"{key}: {value}")
    print("\n" + "=" * 80)
    print("Mapping and comparison completed!")
    print("=" * 80)

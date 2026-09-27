"""TCGA-STAD molecular subtype lookup from cBioPortal."""

from pathlib import Path
import json
import pandas as pd

from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT  # central data store: BASE/'data/...'
JSON_PATH = BASE / 'data/clinical/gastric/GEO/TCGA_STAD_pancan_clinical.json'
OUR_CLIN  = BASE / 'data/clinical/gastric/processed/TCGA_clinical.csv'
OUT_LOOKUP = BASE / 'data/clinical/gastric/processed/TCGA_subtypes.csv'
OUT_MERGED = BASE / 'data/clinical/gastric/processed/TCGA_clinical_with_subtypes.csv'

# 1) Parse the PanCanAtlas JSON
data = json.load(open(JSON_PATH))
df = pd.DataFrame(data)
wide = df.pivot_table(index='patientId', columns='clinicalAttributeId',
                      values='value', aggfunc='first').reset_index()
print(f'PanCanAtlas patients in JSON: {len(wide)}')

# Pull subtype + auxiliary attributes that may be useful later
keep_cols = ['patientId']
for c in ['SUBTYPE', 'MSI_SCORE_MANTIS', 'MSI_SENSOR_SCORE',
          'TMB_NONSYNONYMOUS', 'ANEUPLOIDY_SCORE',
          'GENETIC_ANCESTRY_LABEL', 'SUBTYPE_ABBREVIATION']:
    if c in wide.columns:
        keep_cols.append(c)
lookup = wide[keep_cols].rename(columns={'patientId': 'patient_id'})

# Strip the STAD_ prefix from SUBTYPE for readability
if 'SUBTYPE' in lookup.columns:
    lookup['molecular_subtype'] = lookup['SUBTYPE'].str.replace('STAD_', '', regex=False)
    lookup = lookup.drop(columns=['SUBTYPE'])

# Lowercase column names
lookup.columns = [c.lower() for c in lookup.columns]
print(f'Subtype lookup columns: {lookup.columns.tolist()}')
print(f'Subtype distribution: {lookup["molecular_subtype"].value_counts(dropna=False).to_dict()}')

lookup.to_csv(OUT_LOOKUP, index=False)
print(f'Saved lookup: {OUT_LOOKUP} ({len(lookup)} patients)')

# 2) Merge with our existing TCGA_clinical.csv
our_clin = pd.read_csv(OUR_CLIN)
print(f'\nOur TCGA_clinical.csv: {len(our_clin)} patients')

merged = our_clin.merge(lookup, on='patient_id', how='left')
matched = merged['molecular_subtype'].notna().sum()
print(f'Matched with subtype: {matched} / {len(our_clin)} ({100*matched/len(our_clin):.0f}%)')
print('Matched subtype distribution:')
print(merged['molecular_subtype'].value_counts(dropna=False).to_dict())

merged.to_csv(OUT_MERGED, index=False)
print(f'Saved merged: {OUT_MERGED}')

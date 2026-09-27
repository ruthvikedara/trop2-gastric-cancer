"""Cohort helpers (tumor-only filtering)."""

from pathlib import Path
import pandas as pd
from genolib import PROJECT_ROOT

BASE = PROJECT_ROOT
CLIN = BASE / 'data/clinical/gastric/processed'

# cohort -> (clinical_csv, id_col, column, value, match) for selecting TUMOR samples.
# Cohorts not listed here are tumor-only already.
_TUMOR_RULES = {
    'GSE66229':  ('GSE66229_clinical.csv',  'gsm_id', 'tissue_type',  'tumor', 'exact'),
    'GSE54129':  ('GSE54129_clinical.csv',  'gsm_id', 'tissue',       'tumor', 'contains'),
    'GSE118916': ('GSE118916_clinical.csv', 'gsm_id', 'sample_title', 'tumor', 'contains'),
}

PROJECT_EXCLUDE = {'GSE26253'}

# MSI calls non-credible on these platforms (see run notes). GSE26253 remains
# listed through PROJECT_EXCLUDE so older callers cannot accidentally restore it.
MSI_EXCLUDE = PROJECT_EXCLUDE | {'GSE118916'}

# Subtype (GCclassifier TCGA/ACRG) failed on GSE26253 (>10% signature genes missing).
SUBTYPE_TCGA_ACRG_EXCLUDE = PROJECT_EXCLUDE.copy()


def tumor_ids(cohort):
    """Set of tumor sample IDs for a cohort, or None if the cohort is all-tumor."""
    rule = _TUMOR_RULES.get(cohort)
    if rule is None:
        return None
    csv, id_col, col, val, match = rule
    clin = pd.read_csv(CLIN / csv)
    clin[id_col] = clin[id_col].astype(str).str.strip()
    s = clin[col].astype(str).str.strip().str.lower()
    mask = (s == val.lower()) if match == 'exact' else s.str.contains(val.lower(), na=False)
    return set(clin.loc[mask, id_col])


def restrict_to_tumor(series_or_df, cohort):
    """Drop normal samples (by index) when a cohort carries them; else passthrough."""
    keep = tumor_ids(cohort)
    if keep is None:
        return series_or_df
    idx = series_or_df.index.astype(str)
    return series_or_df[idx.isin(keep)]

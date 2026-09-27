"""Survival loaders and statistics."""

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from statsmodels.stats.meta_analysis import combine_effects

from genolib import PROJECT_ROOT


BASE = PROJECT_ROOT
CLIN = BASE / 'data' / 'clinical' / 'gastric' / 'processed'
PROC = BASE / 'data' / 'processed' / 'gastric'
GENES = ('TACSTD2', 'CLDN18', 'CD274', 'ERBB2')

DATASETS = {
    'TCGA-STAD': dict(
        clinical='TCGA_clinical_with_subtypes.csv',
        expr='TCGA_STAD_for_xcell.txt',
        id_col='patient_id',
        filt=None,
    ),
    'GSE66229': dict(
        clinical='GSE66229_clinical.csv',
        expr='ACRG_GSE66229_for_xcell.txt',
        id_col='gsm_id',
        filt=('tissue_type', 'Tumor'),
    ),
    'GSE15459': dict(
        clinical='GSE15459_clinical.csv',
        expr='ACRG_GSE15459_for_xcell.txt',
        id_col='gsm_id',
        filt=None,
    ),
    'GSE34942': dict(
        clinical='GSE34942_clinical.csv',
        expr='ACRG_GSE34942_for_xcell.txt',
        id_col='gsm_id',
        filt=None,
    ),
}


def normalize_id(value):
    value = str(value).strip()
    return '-'.join(value.split('-')[:3]) if value.startswith('TCGA') else value


def load_gene_rows(path, genes=GENES):
    found = {}
    with open(path) as handle:
        samples = [normalize_id(x) for x in handle.readline().rstrip('\n').split('\t')[1:]]
        for line in handle:
            tab = line.index('\t')
            gene = line[:tab]
            if gene not in genes:
                continue
            values = line[tab + 1:].rstrip('\n').split('\t')
            found[gene] = pd.Series(
                [float(x) if x.strip() else np.nan for x in values],
                index=samples,
                dtype=float,
            )
    return found


def stage_group(value):
    if pd.isna(value):
        return np.nan
    value = str(value).upper().replace('STAGE', '').strip()
    if value.startswith(('III', 'IV', '3', '4')):
        return 'III-IV'
    if value.startswith(('I', 'II', '1', '2')):
        return 'I-II'
    return np.nan


def clean_sex(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if value.startswith('m'):
        return 'Male'
    if value.startswith('f'):
        return 'Female'
    return np.nan


def clean_lauren(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if 'intestinal' in value:
        return 'Intestinal'
    if 'diffuse' in value:
        return 'Diffuse'
    if 'mixed' in value:
        return 'Mixed'
    return np.nan


def subtype_table(cohort):
    if cohort == 'TCGA-STAD':
        data = pd.read_csv(CLIN / 'TCGA_clinical_with_subtypes.csv')
        data['sample_id'] = data['patient_id'].map(normalize_id)
        return data.set_index('sample_id')[['molecular_subtype']].rename(
            columns={'molecular_subtype': 'TCGA_subtype'}
        )
    path = CLIN / f'{cohort}_subtypes_predicted.csv'
    data = pd.read_csv(path)
    data['sample_id'] = data['sample_id'].map(normalize_id)
    return data.set_index('sample_id')[['TCGA_subtype', 'EMP_subtype']]


def load_cohort(cohort, cfg):
    clinical = pd.read_csv(CLIN / cfg['clinical'])
    if cfg['filt']:
        col, val = cfg['filt']
        clinical = clinical[clinical[col].astype(str).str.lower() == val.lower()]
    clinical['sample_id'] = clinical[cfg['id_col']].map(normalize_id)
    clinical = clinical.drop_duplicates('sample_id').set_index('sample_id')

    expr = load_gene_rows(PROC / cfg['expr'])
    # Eligibility for the OS/subgroup analyses requires TROP2, not complete
    # expression for every optional biomarker.  Preserve biomarker missingness
    # and let each biomarker-specific model drop only the rows it actually uses.
    common = clinical.index.intersection(expr['TACSTD2'].dropna().index)
    clinical = clinical.loc[common].copy()
    out = pd.DataFrame(index=common)
    out['duration'] = pd.to_numeric(clinical['os_months'], errors='coerce')
    out['event'] = pd.to_numeric(clinical['os_status'], errors='coerce')
    out['age'] = pd.to_numeric(clinical.get('age'), errors='coerce')
    out['sex'] = clinical.get('sex', pd.Series(index=common, dtype=object)).map(clean_sex)
    out['stage_group'] = clinical.get(
        'stage', pd.Series(index=common, dtype=object)
    ).map(stage_group)
    out['lauren'] = clinical.get(
        'lauren', pd.Series(index=common, dtype=object)
    ).map(clean_lauren)
    out['cohort'] = cohort

    for gene in GENES:
        values = expr[gene].reindex(common).astype(float)
        out[gene] = values
        sd = values.std(ddof=0)
        out[f'{gene}_z'] = (values - values.mean()) / sd if sd > 0 else np.nan
        out[f'{gene}_high'] = (
            (values >= values.median()).astype(float).where(values.notna())
        )

    subtypes = subtype_table(cohort)
    out = out.join(subtypes, how='left')
    if 'EMP_subtype' not in out:
        # TCGA EMP calls come from the expression classifier output.
        predicted = pd.read_csv(CLIN / 'TCGA-STAD_subtypes_predicted.csv')
        predicted['sample_id'] = predicted['sample_id'].map(normalize_id)
        out = out.join(
            predicted.set_index('sample_id')[['EMP_subtype']],
            how='left',
        )
    out = out[(out['duration'] > 0) & out['event'].isin([0, 1])]
    return out


def load_all():
    return pd.concat(
        [load_cohort(name, cfg) for name, cfg in DATASETS.items()],
        axis=0,
    )


def fit_cox(data, covariates, target='TACSTD2_high', strata=True, penalizer=0.0):
    columns = ['duration', 'event', *covariates]
    if strata:
        columns.append('cohort')
    sub = data[columns].dropna().copy()
    result = {
        'n': len(sub),
        # Per-arm N (TROP2-high / TROP2-low) for the forest "N" column.
        'n_high': int((sub[target] == 1).sum()) if target in sub else 0,
        'n_low': int((sub[target] == 0).sum()) if target in sub else 0,
        'events': int(sub['event'].sum()),
        'HR': np.nan,
        'lo': np.nan,
        'hi': np.nan,
        'p': np.nan,
        'coef': np.nan,
        'se': np.nan,
    }
    # Permit small prespecified strata (notably TCGA EBV: n=38, 14 events)
    # when the model is estimable; the wide CI communicates the imprecision.
    if len(sub) < 30 or sub['event'].sum() < 10 or sub[target].nunique() < 2:
        return result
    model = CoxPHFitter(penalizer=penalizer)
    model.fit(
        sub,
        duration_col='duration',
        event_col='event',
        strata=['cohort'] if strata else None,
    )
    row = model.summary.loc[target]
    result.update({
        'HR': float(np.exp(row['coef'])),
        'lo': float(np.exp(row['coef lower 95%'])),
        'hi': float(np.exp(row['coef upper 95%'])),
        'p': float(row['p']),
        'coef': float(row['coef']),
        'se': float(row['se(coef)']),
    })
    return result


# Subgroup forest statistics (shared by the full S2A forest and the compact
# F2E inset so both panels report identical numbers).
#
# Multiplicity (Benjamini-Hochberg):
# there is NOT a single FDR for the whole figure, FDR is computed per test
# within a declared family, so every row carries its own value. Two families
# are reported, and both are always adjusted over the FULL nine-factor set
# below, never over whichever subset a given panel happens to display:
#   1. stratum-level FDR, BH across all estimable within-stratum TROP2 HR
#      tests (the per-row `p` column);
#   2. interaction FDR, BH across the nine factor-level interaction tests
#      (the factor header rows).
# Adjusting over the full family is what keeps the compact F2E inset from
# printing a different FDR than S2A for the very same test.

FULL_SUBGROUP_FACTORS = [
    ('Age', 'age_group', ['<50', '50-64', '>=65']),
    ('Sex', 'sex', ['Female', 'Male']),
    ('Stage', 'stage_group', ['I-II', 'III-IV']),
    ('Lauren type', 'lauren', ['Intestinal', 'Diffuse', 'Mixed']),
    ('TCGA subtype', 'TCGA_subtype', ['EBV', 'MSI', 'GS', 'CIN']),
    ('EMP subtype', 'EMP_subtype', ['EP', 'MP']),
    ('CLDN18 expression', 'CLDN18_group', ['Low', 'High']),
    ('PD-L1 (CD274) expression', 'CD274_group', ['Low', 'High']),
    ('HER2 (ERBB2) expression', 'ERBB2_group', ['Low', 'High']),
]


def prepare_subgroup_data(data):
    """Add the derived grouping columns the subgroup factors reference."""
    import numpy as np  # local: keeps the module import order unchanged
    data = data.copy()
    data['age_group'] = pd.cut(
        data['age'], [-np.inf, 50, 65, np.inf], labels=['<50', '50-64', '>=65'],
        right=False,
    )
    for gene in ('CLDN18', 'CD274', 'ERBB2'):
        data[f'{gene}_group'] = data[f'{gene}_high'].map({0: 'Low', 1: 'High'})
    return data


def interaction_p(data, column, levels):
    """Likelihood-ratio test for TROP2 x subgroup interaction (cohort-stratified)."""
    from scipy.stats import chi2
    sub = data[['duration', 'event', 'cohort', 'TACSTD2_high', column]].dropna().copy()
    sub = sub[sub[column].isin(levels)]
    dummies = pd.get_dummies(sub[column], prefix='group', drop_first=True, dtype=float)
    if dummies.shape[1] == 0:
        return np.nan
    design = pd.concat(
        [sub[['duration', 'event', 'cohort', 'TACSTD2_high']], dummies],
        axis=1,
    )
    base = CoxPHFitter(penalizer=0.005)
    base.fit(design, 'duration', 'event', strata=['cohort'])
    interaction_names = []
    for dummy in dummies:
        name = f'ix_{dummy}'
        design[name] = design['TACSTD2_high'] * design[dummy]
        interaction_names.append(name)
    full = CoxPHFitter(penalizer=0.005)
    full.fit(design, 'duration', 'event', strata=['cohort'])
    statistic = 2 * (full.log_likelihood_ - base.log_likelihood_)
    return float(chi2.sf(max(0, statistic), len(interaction_names)))


def _bh(values):
    """BH-adjusted p-values, propagating NaN for non-estimable tests."""
    from statsmodels.stats.multitest import multipletests
    values = pd.Series(values, dtype=float)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    ok = values.notna()
    if ok.any():
        out[ok] = multipletests(values[ok].to_numpy(), method='fdr_bh')[1]
    return out


def subgroup_stats(data, factors=FULL_SUBGROUP_FACTORS):
    """Per-stratum TROP2 HRs + BH FDR over the full nine-factor family.

    Always fits every factor in ``factors`` (default: the full family) so the
    BH denominators are fixed; callers displaying a subset filter the returned
    frame afterwards rather than re-adjusting.
    """
    data = prepare_subgroup_data(data)
    rows = []
    for factor, column, levels in factors:
        ip = interaction_p(data, column, levels)
        for level in levels:
            result = fit_cox(
                data[data[column] == level], ['TACSTD2_high'],
                target='TACSTD2_high', strata=True,
            )
            rows.append({'factor': factor, 'column': column, 'level': level,
                         'interaction_p': ip, **result})
    stats = pd.DataFrame(rows)
    stats['fdr'] = _bh(stats['p']).to_numpy()
    per_factor = stats.groupby('factor', sort=False)['interaction_p'].first()
    stats['interaction_fdr'] = stats['factor'].map(
        pd.Series(_bh(per_factor).to_numpy(), index=per_factor.index)
    )
    return stats


def random_effects_trop2(data, target='TACSTD2_high'):
    """Paule-Mandel random-effects synthesis of cohort-specific log-HRs."""
    rows = []
    for cohort in DATASETS:
        result = fit_cox(
            data[data['cohort'] == cohort],
            [target],
            target=target,
            strata=False,
        )
        rows.append({'cohort': cohort, **result})
    estimates = pd.DataFrame(rows).dropna(subset=['coef', 'se'])
    meta = combine_effects(
        estimates['coef'].to_numpy(),
        estimates['se'].to_numpy() ** 2,
        method_re='iterated',
        row_names=estimates['cohort'].tolist(),
        use_t=False,
    )
    effect = float(meta.mean_effect_re)
    se = float(np.sqrt(meta.var_eff_w_re))
    summary = {
        'HR': float(np.exp(effect)),
        'lo': float(np.exp(effect - 1.96 * se)),
        'hi': float(np.exp(effect + 1.96 * se)),
        'p': float(2 * (1 - __import__('scipy').stats.norm.cdf(abs(effect / se)))),
        'tau2': float(meta.tau2),
        'i2': float(max(0.0, meta.i2) * 100),
        'k': len(estimates),
    }
    return estimates, summary

"""Audit survival cohorts."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from genolib import PROJECT_ROOT

BASE = PROJECT_ROOT
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'survival'
OUT.mkdir(parents=True, exist_ok=True)

# cohort -> (clinical csv, id filter, descriptor). Mirrors run_survival_cdh17_trop2.py.
COHORTS = {
    'TCGA-STAD': dict(csv='TCGA_clinical.csv', filt=None,
                      desc='TCGA-STAD (US/multi-site, primary resection)'),
    'GSE66229':  dict(csv='GSE66229_clinical.csv', filt=('tissue_type', 'Tumor'),
                      desc='ACRG (Korea, primary resection)'),
    'GSE15459':  dict(csv='GSE15459_clinical.csv', filt=None,
                      desc='Singapore (primary resection)'),
    'GSE34942':  dict(csv='GSE34942_clinical.csv', filt=None,
                      desc='Singapore (primary resection)'),
}
CLIN = BASE / 'data' / 'clinical' / 'gastric' / 'processed'

# treatment / line-of-therapy column name candidates we look for
TREAT_KEYS = ('treat', 'therap', 'chemo', 'drug', 'regimen', 'line', 'adjuvant',
              'neoadjuvant', 'radiat', 'surg', 'immunother', 'modality')


def norm_stage(v):
    if pd.isna(v):
        return None
    s = str(v).strip().upper().replace('STAGE ', '')
    try:
        n = float(s)
        return 'I-II' if n <= 2 else 'III' if n == 3 else 'IV' if n == 4 else None
    except ValueError:
        pass
    if s.startswith('IV'):
        return 'IV'
    if s.startswith('III'):
        return 'III'
    if s.startswith('I'):
        return 'I-II'
    return None


def norm_lauren(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s == 'intestinal':
        return 'Intestinal'
    if s == 'diffuse':
        return 'Diffuse'
    if s == 'mixed':
        return 'Mixed'
    return None


def main():
    rows = []
    stage_comp = {}
    lauren_comp = {}
    STAGES = ['I-II', 'III', 'IV']
    LAURENS = ['Intestinal', 'Diffuse', 'Mixed']

    for cohort, cfg in COHORTS.items():
        clin = pd.read_csv(CLIN / cfg['csv'])
        if cfg['filt']:
            col, val = cfg['filt']
            clin = clin[clin[col] == val]
        clin = clin.dropna(subset=['os_months', 'os_status'])
        os_status = pd.to_numeric(clin['os_status'], errors='coerce')
        os_months = pd.to_numeric(clin['os_months'], errors='coerce')
        n = len(clin)
        events = int((os_status == 1).sum())

        stage = clin['stage'].map(norm_stage) if 'stage' in clin.columns else pd.Series(dtype=object)
        lauren = clin['lauren'].map(norm_lauren) if 'lauren' in clin.columns else pd.Series(dtype=object)
        stage_comp[cohort] = {s: int((stage == s).sum()) for s in STAGES}
        lauren_comp[cohort] = {l: int((lauren == l).sum()) for l in LAURENS}

        treat_cols = [c for c in clin.columns
                      if any(k in c.lower() for k in TREAT_KEYS)]
        subtype = ('molecular_subtype' in clin.columns)

        rows.append({
            'cohort': cohort, 'descriptor': cfg['desc'],
            'n_with_OS': n, 'events': events, 'event_rate': round(events / n, 2) if n else np.nan,
            'median_OS_months': round(float(os_months.median()), 1),
            'max_followup_months': round(float(os_months.max()), 1),
            'stage_annotated': int(stage.notna().sum()),
            'stage_I_II/III/IV': f"{stage_comp[cohort]['I-II']}/{stage_comp[cohort]['III']}/{stage_comp[cohort]['IV']}",
            'lauren_annotated': int(lauren.notna().sum()),
            'molecular_subtype_available': subtype,
            'treatment_columns_found': ';'.join(treat_cols) if treat_cols else 'NONE',
            'line_of_therapy': 'NOT RECORDED',
            'treatment_modality': 'NOT RECORDED (primary-resection cohort)',
        })

    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / 'F4_OS_cohort_audit.csv', index=False)
    print(audit.to_string(index=False))

    # figure: stage + Lauren composition + treatment-data flag
    cohorts = list(COHORTS)
    fig, (axS, axL, axT) = plt.subplots(1, 3, figsize=(15, 4.6),
                                        gridspec_kw={'width_ratios': [1, 1, 1.15]})
    fig.suptitle('OS cohort audit - composition & treatment-data availability '
                 '(read before interpreting KM)', fontsize=12.5, fontweight='bold', y=1.02)

    stage_colors = {'I-II': '#9ecae1', 'III': '#fdae6b', 'IV': '#de2d26'}
    lauren_colors = {'Intestinal': '#74c476', 'Diffuse': '#9e9ac8', 'Mixed': '#c7c7c7'}

    def stacked(ax, comp, cats, colors, title):
        x = np.arange(len(cohorts))
        bottom = np.zeros(len(cohorts))
        for cat in cats:
            vals = np.array([comp[c][cat] for c in cohorts], float)
            ax.bar(x, vals, bottom=bottom, color=colors[cat], label=cat, width=0.7,
                   edgecolor='white', linewidth=0.5)
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v >= 8:
                    ax.text(xi, b + v / 2, int(v), ha='center', va='center',
                            fontsize=7.5, color='black')
            bottom += vals
        ax.set_xticks(x); ax.set_xticklabels(cohorts, rotation=30, ha='right', fontsize=8.5)
        ax.set_ylabel('patients', fontsize=9)
        ax.set_title(title, fontsize=10.5, pad=6)
        ax.legend(fontsize=8, frameon=False, title_fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    stacked(axS, stage_comp, STAGES, stage_colors, 'Stage distribution')
    stacked(axL, lauren_comp, LAURENS, lauren_colors, 'Lauren distribution')

    # treatment-data flag panel
    axT.axis('off')
    axT.set_title('Treatment / line-of-therapy data', fontsize=10.5, pad=6)
    lines = [
        '⚠  Treatment modality: NOT RECORDED',
        '⚠  Line of therapy:     NOT RECORDED',
        '',
        'All four are PRIMARY-RESECTION cohorts',
        '(surgery ± unspecified adjuvant); none are',
        'IO- or ADC-treated, treatment-stratified trials.',
        '',
        'Available covariates: stage, Lauren, age, sex,',
        'molecular subtype (TCGA/ACRG), EBV.',
        '',
        'Interpretation: OS here = natural history of',
        'resected disease. It CANNOT capture a TROP2-ADC',
        'or ICI effect (no such treatment given) → a null',
        'TROP2-OS association is the expected, honest',
        'result, and supports the TME / patient-selection',
        'framing rather than a prognostic-biomarker claim.',
    ]
    axT.text(0.0, 0.96, '\n'.join(lines), transform=axT.transAxes, ha='left', va='top',
             fontsize=9, family='sans-serif',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff7e6',
                       edgecolor='#d9a066', linewidth=1.2))

    plt.tight_layout()
    out = OUT / 'F4_OS_cohort_audit'
    for ext in ('.pdf', '.png'):
        fig.savefig(out.with_suffix(ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out}.{{pdf,png}} + .csv')


if __name__ == '__main__':
    main()

"""Multivariable Cox model."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import PAL, apply_pub_style, clean_ax, fmt_p, savefig
from survival_revision_utils import load_all


OUT = Path(__file__).resolve().parent.parent / 'plots' / 'survival'
STEM = 'F2_TROP2_multivariable_OS_forest'


def main():
    apply_pub_style()
    data = load_all().copy()
    data['age_per_10y'] = data['age'] / 10
    data['male'] = (data['sex'] == 'Male').astype(float).where(data['sex'].notna())
    data['stage_III_IV'] = (
        (data['stage_group'] == 'III-IV').astype(float)
        .where(data['stage_group'].notna())
    )
    for level in ('Diffuse', 'Mixed'):
        data[f'lauren_{level.lower()}'] = (
            (data['lauren'] == level).astype(float).where(data['lauren'].notna())
        )
    for level in ('EBV', 'MSI', 'GS'):
        data[f'tcga_{level}'] = (
            (data['TCGA_subtype'] == level).astype(float)
            .where(data['TCGA_subtype'].notna())
        )

    covariates = [
        'TACSTD2_high', 'age_per_10y', 'male', 'stage_III_IV',
        'lauren_diffuse', 'lauren_mixed',
        'tcga_EBV', 'tcga_MSI', 'tcga_GS',
        'CLDN18_z', 'CD274_z', 'ERBB2_z',
    ]
    columns = ['duration', 'event', 'cohort', *covariates]
    complete = data[columns].dropna().copy()
    model = CoxPHFitter(penalizer=0.01)
    model.fit(
        complete, duration_col='duration', event_col='event',
        strata=['cohort'],
    )

    labels = {
        'TACSTD2_high': 'TROP2-high vs low',
        'age_per_10y': 'Age (per 10 years)',
        'male': 'Male vs female',
        'stage_III_IV': 'Stage III-IV vs I-II',
        'lauren_diffuse': 'Lauren diffuse vs intestinal',
        'lauren_mixed': 'Lauren mixed vs intestinal',
        'tcga_EBV': 'TCGA EBV vs CIN',
        'tcga_MSI': 'TCGA MSI vs CIN',
        'tcga_GS': 'TCGA GS vs CIN',
        'CLDN18_z': 'CLDN18 (per SD)',
        'CD274_z': 'CD274 / PD-L1 (per SD)',
        'ERBB2_z': 'ERBB2 / HER2 (per SD)',
    }
    rows = []
    for term in covariates:
        result = model.summary.loc[term]
        rows.append({
            'term': term,
            'label': labels[term],
            'HR': float(np.exp(result['coef'])),
            'lo': float(np.exp(result['coef lower 95%'])),
            'hi': float(np.exp(result['coef upper 95%'])),
            'p': float(result['p']),
            'n': len(complete),
            'events': int(complete['event'].sum()),
        })
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / f'{STEM}_stats.csv', index=False)

    # Three dedicated columns prevent labels/estimates from intruding into the
    # confidence-interval field after panel reduction.
    fig = plt.figure(figsize=(10.2, 7.25))
    grid = fig.add_gridspec(
        1, 3, width_ratios=[2.35, 2.35, 3.05], wspace=0.025,
        left=0.035, right=0.985, top=0.845, bottom=0.13,
    )
    label_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])
    estimate_ax = fig.add_subplot(grid[0, 2])
    y = np.arange(len(stats))[::-1]
    for yy, row in zip(y, stats.itertuples()):
        color = PAL['accent_high'] if row.term == 'TACSTD2_high' else '#555555'
        ax.plot([row.lo, row.hi], [yy, yy], color=color, lw=2.0)
        ax.scatter(row.HR, yy, s=70 if row.term == 'TACSTD2_high' else 55,
                   marker='D' if row.term == 'TACSTD2_high' else 'o',
                   color=color, edgecolor='black', lw=0.5, zorder=3)
        label_ax.text(
            0.98, yy, row.label,
            ha='right', va='center', fontsize=11.3,
            color=color if row.term == 'TACSTD2_high' else '#222222',
            fontweight='bold' if row.term == 'TACSTD2_high' else 'normal',
        )
        estimate_ax.text(
            0.02, yy,
            f'{row.HR:.2f} ({row.lo:.2f}-{row.hi:.2f}); {fmt_p(row.p)}',
            ha='left', va='center', fontsize=10.8, color=color,
        )
    ax.axvline(1, color='black', ls='--', lw=0.9)
    ax.set_xscale('log')
    ax.set_xlim(0.35, 5.0)
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xticklabels(['0.5', '1', '2', '4'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=10.5)
    ax.set_xlabel('Adjusted hazard ratio for overall survival', fontsize=12.5)
    ax.set_ylim(-0.65, len(stats) - 0.25)
    clean_ax(ax)

    for side_ax in (label_ax, estimate_ax):
        side_ax.set_xlim(0, 1)
        side_ax.set_ylim(ax.get_ylim())
        side_ax.set_xticks([])
        side_ax.set_yticks([])
        for spine in side_ax.spines.values():
            spine.set_visible(False)
    label_ax.set_title('Covariate', fontsize=11.5, fontweight='bold',
                       loc='right', pad=8)
    estimate_ax.set_title('HR (95% CI); p', fontsize=11.5,
                          fontweight='bold', loc='left', pad=8)
    fig.suptitle('Multivariable overall-survival model',
                 fontsize=15, fontweight='bold', y=0.975)
    fig.text(
        0.5, 0.92,
        f'n={len(complete)}, events={int(complete.event.sum())}; '
        'baseline stratified by cohort',
        ha='center', va='center', fontsize=11.2, color='#333333',
    )
    savefig(fig, OUT / STEM, pad_inches=0.04)
    print(model.summary[['coef', 'exp(coef)', 'p']].to_string())


if __name__ == '__main__':
    main()

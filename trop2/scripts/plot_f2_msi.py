"""TROP2 expression by MSI status."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from cohort_utils import restrict_to_tumor, MSI_EXCLUDE
from style import PAL, clean_ax, savefig, fmt_p
from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT
PROC = BASE / 'data/clinical/gastric/processed'
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
OUT.mkdir(parents=True, exist_ok=True)

TROP2 = 'TACSTD2'
MSS_COLOR = PAL['mss']
MSIH_COLOR = PAL['msi']

COHORTS = {
    'TCGA-STAD': 'data/processed/gastric/TCGA_STAD_for_xcell.txt',
    'GSE66229':  'data/processed/gastric/ACRG_GSE66229_for_xcell.txt',
    'GSE15459':  'data/processed/gastric/ACRG_GSE15459_for_xcell.txt',
    'GSE34942':  'data/processed/gastric/ACRG_GSE34942_for_xcell.txt',
    'GSE35809':  'data/processed/gastric/ACRG_GSE35809_for_xcell.txt',
    'GSE51105':  'data/processed/gastric/ACRG_GSE51105_for_xcell.txt',
    'GSE54129':  'data/processed/gastric/ACRG_GSE54129_for_xcell.txt',
    'GSE57303':  'data/processed/gastric/ACRG_GSE57303_for_xcell.txt',
    'GSE84437':  'data/processed/gastric/ACRG_GSE84437_for_xcell.txt',
    'GSE118916': 'data/processed/gastric/ACRG_GSE118916_for_xcell.txt',
}


def trop2_series(expr_file):
    with open(expr_file) as f:
        samples = f.readline().rstrip('\n').split('\t')[1:]
        for line in f:
            tab = line.index('\t')
            if line[:tab] == TROP2:
                vals = line[tab + 1:].rstrip('\n').split('\t')
                arr = [float(v) if v.strip() else np.nan for v in vals]
                return pd.Series(arr, index=samples)
    return None


def fmt_p(p):
    if p is None or np.isnan(p):
        return 'p = N/A'
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3g}'


def main():
    panels, stats_rows = [], []
    for cohort, rel in COHORTS.items():
        if cohort in MSI_EXCLUDE:
            print(f'[{cohort}] EXCLUDED - MSI calls non-credible on this platform')
            continue
        expr = trop2_series(BASE / rel)
        msi_f = PROC / f'{cohort}_msi_predicted.csv'
        if expr is None or not msi_f.exists():
            print(f'[{cohort}] missing expression or MSI labels - skipped')
            continue
        expr = restrict_to_tumor(expr, cohort)  # drop adjacent normals
        msi = pd.read_csv(msi_f).set_index('sample_id')['msi_premsim']
        df = pd.DataFrame({'trop2': expr}).join(pd.DataFrame({'msi': msi}), how='inner').dropna()
        mss = df.loc[df['msi'] == 'MSS', 'trop2'].values
        msih = df.loc[df['msi'] == 'MSI-H', 'trop2'].values
        if len(mss) < 5 or len(msih) < 5:
            print(f'[{cohort}] too few MSI-H (n={len(msih)}) - skipped')
            continue
        _, p = mannwhitneyu(msih, mss, alternative='two-sided')
        panels.append((cohort, mss, msih, p))
        stats_rows.append({'cohort': cohort, 'n_mss': len(mss), 'n_msih': len(msih),
                           'trop2_median_mss': float(np.median(mss)),
                           'trop2_median_msih': float(np.median(msih)),
                           'mannwhitney_p': p})
        print(f'[{cohort}] MSS med={np.median(mss):.2f} (n={len(mss)}) vs '
              f'MSI-H med={np.median(msih):.2f} (n={len(msih)}), {fmt_p(p)}')

    if not panels:
        print('Nothing to plot - run the classifier driver first.')
        return

    n = len(panels)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle('TROP2 by MSI status',
                 fontsize=12, fontweight='bold', y=1.005)

    rng = np.random.default_rng(0)
    for ax, (cohort, mss, msih, p) in zip(axes, panels):
        positions = [0, 1]
        parts = ax.violinplot([mss, msih], positions=positions, showmeans=False,
                              showmedians=False, showextrema=False, widths=0.7)
        for pc, color in zip(parts['bodies'], [MSS_COLOR, MSIH_COLOR]):
            pc.set_facecolor(color); pc.set_alpha(0.35); pc.set_edgecolor(color)
        ax.boxplot([mss, msih], positions=positions, widths=0.18, patch_artist=True,
                   medianprops={'color': 'black', 'linewidth': 1.2},
                   boxprops={'facecolor': 'white', 'linewidth': 1.0},
                   whiskerprops={'linewidth': 1.0}, capprops={'linewidth': 1.0},
                   flierprops={'marker': 'o', 'markersize': 2.5, 'markerfacecolor': 'gray',
                               'markeredgecolor': 'gray', 'alpha': 0.4})
        for x_pos, arr, color in zip(positions, [mss, msih], [MSS_COLOR, MSIH_COLOR]):
            ax.scatter(np.full(len(arr), x_pos) + rng.uniform(-0.07, 0.07, len(arr)), arr,
                       s=6, color=color, alpha=0.5, edgecolors='none', zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([f'MSS\n(n={len(mss)})', f'MSI-H\n(n={len(msih)})'], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(cohort, fontsize=10, fontweight='bold', pad=12)
        ax.text(0.5, 1.005, fmt_p(p), transform=ax.transAxes, ha='center', va='bottom',
                fontsize=7.5, color='#333')
        ax.set_ylabel('TROP2 (log2)', fontsize=8)
        clean_ax(ax)
    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    savefig(fig, OUT / 'F2_TROP2_by_MSI')
    pd.DataFrame(stats_rows).to_csv(OUT / 'F2_TROP2_by_MSI_stats.csv', index=False)


if __name__ == '__main__':
    main()

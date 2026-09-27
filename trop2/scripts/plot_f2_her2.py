"""TROP2 expression by HER2 status."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

from cohort_utils import restrict_to_tumor
from style import PAL, clean_ax, savefig, fmt_p
from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
OUT.mkdir(parents=True, exist_ok=True)

TROP2 = 'TACSTD2'
HER2 = 'ERBB2'

STRAT_METHOD = 'median'  # 'median' | 'tertile'; overridable via --strat
if '--strat' in sys.argv:
    STRAT_METHOD = sys.argv[sys.argv.index('--strat') + 1]

# All gastric cohorts with processed expression matrices.
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

HIGH_COLOR = PAL['high']
LOW_COLOR = PAL['low']


def load_gene_rows(expr_file, genes):
    target = set(genes)
    found = {}
    with open(expr_file) as f:
        header = f.readline().rstrip('\n').split('\t')
        samples = header[1:]
        for line in f:
            tab = line.index('\t')
            gene = line[:tab]
            if gene in target:
                vals = line[tab + 1:].rstrip('\n').split('\t')
                found[gene] = [float(v) if v.strip() else np.nan for v in vals]
                if len(found) == len(target):
                    break
    return pd.DataFrame(found, index=samples)  # samples x genes


def her2_groups(erbb2: pd.Series, method: str):
    """Return (high_mask, low_mask) over non-NA ERBB2. Tertile drops the middle."""
    valid = erbb2.notna()
    v = erbb2[valid]
    if method == 'tertile':
        lo, hi = v.quantile([1 / 3, 2 / 3])
        high = valid & (erbb2 >= hi)
        low = valid & (erbb2 <= lo)
    else:  # median
        med = v.median()
        high = valid & (erbb2 >= med)
        low = valid & (erbb2 < med)
    return high, low


def fmt_p(p):
    if p is None or np.isnan(p):
        return 'p = N/A'
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3g}'


def main():
    rng = np.random.default_rng(0)
    print(f'HER2 stratification: {STRAT_METHOD}')

    stats_rows = []
    panels = []  # (cohort, trop2_high, trop2_low, p, rho, rho_p)
    for cohort, rel in COHORTS.items():
        df = load_gene_rows(BASE / rel, [TROP2, HER2])
        if TROP2 not in df.columns or HER2 not in df.columns:
            print(f'[{cohort}] missing {TROP2 if TROP2 not in df.columns else HER2} - skipped')
            continue
        df = restrict_to_tumor(df, cohort)  # drop adjacent normals
        trop2 = pd.to_numeric(df[TROP2], errors='coerce')
        erbb2 = pd.to_numeric(df[HER2], errors='coerce')

        high, low = her2_groups(erbb2, STRAT_METHOD)
        t_high = trop2[high].dropna().values
        t_low = trop2[low].dropna().values
        if len(t_high) < 5 or len(t_low) < 5:
            print(f'[{cohort}] too few in a HER2 group - skipped')
            continue

        _, p = mannwhitneyu(t_high, t_low, alternative='two-sided')
        both = trop2.notna() & erbb2.notna()
        rho, rho_p = spearmanr(trop2[both], erbb2[both])

        panels.append((cohort, t_high, t_low, p, rho, rho_p))
        stats_rows.append({
            'cohort': cohort, 'strat': STRAT_METHOD,
            'n_her2_high': len(t_high), 'n_her2_low': len(t_low),
            'trop2_median_her2high': float(np.median(t_high)),
            'trop2_median_her2low': float(np.median(t_low)),
            'mannwhitney_p': p, 'spearman_rho': rho, 'spearman_p': rho_p,
        })
        print(f'[{cohort}] TROP2 HER2-high med={np.median(t_high):.2f} vs '
              f'low={np.median(t_low):.2f}, {fmt_p(p)}; Spearman rho={rho:+.2f} ({fmt_p(rho_p)})')

    if not panels:
        print('No cohorts plotted.')
        return

    # --- grid ---
    n = len(panels)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    hi_lbl = 'HER2-high' + ('\n(top third)' if STRAT_METHOD == 'tertile' else '\n(≥ median)')
    lo_lbl = 'HER2-low' + ('\n(bottom third)' if STRAT_METHOD == 'tertile' else '\n(< median)')
    fig.suptitle(f'TROP2 by HER2 (ERBB2) status',
                 fontsize=12, fontweight='bold', y=1.005)

    for ax, (cohort, t_high, t_low, p, rho, rho_p) in zip(axes, panels):
        positions = [0, 1]
        parts = ax.violinplot([t_low, t_high], positions=positions,
                              showmeans=False, showmedians=False, showextrema=False, widths=0.7)
        for pc, color in zip(parts['bodies'], [LOW_COLOR, HIGH_COLOR]):
            pc.set_facecolor(color); pc.set_alpha(0.35); pc.set_edgecolor(color)
        ax.boxplot([t_low, t_high], positions=positions, widths=0.18, patch_artist=True,
                   medianprops={'color': 'black', 'linewidth': 1.2},
                   boxprops={'facecolor': 'white', 'linewidth': 1.0},
                   whiskerprops={'linewidth': 1.0}, capprops={'linewidth': 1.0},
                   flierprops={'marker': 'o', 'markersize': 2.5, 'markerfacecolor': 'gray',
                               'markeredgecolor': 'gray', 'alpha': 0.4})
        for x_pos, arr, color in zip(positions, [t_low, t_high], [LOW_COLOR, HIGH_COLOR]):
            jitter = rng.uniform(-0.07, 0.07, size=len(arr))
            ax.scatter(np.full(len(arr), x_pos) + jitter, arr, s=6, color=color,
                       alpha=0.5, edgecolors='none', zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels([f'{lo_lbl}\n(n={len(t_low)})', f'{hi_lbl}\n(n={len(t_high)})'], fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_title(cohort, fontsize=10, fontweight='bold', pad=12)
        ax.set_ylabel('TROP2 (log2)', fontsize=8)
        clean_ax(ax)

        ax.text(0.5, 1.005, f'{fmt_p(p)}  •  ρ={rho:+.2f}', transform=ax.transAxes,
                ha='center', va='bottom', fontsize=7.5, color='#333333')

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    savefig(fig, OUT / 'F2_TROP2_by_HER2')

    stats_df = pd.DataFrame(stats_rows)
    stats_csv = OUT / 'F2_TROP2_HER2_stats.csv'
    stats_df.to_csv(stats_csv, index=False)


if __name__ == '__main__':
    main()

"""TROP2 vs HER2 and Claudin-18, adjusted for epithelial content."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata, t as tdist

from cohort_utils import restrict_to_tumor
from style import PAL, clean_ax, savefig
from genolib import PROJECT_ROOT

BASE = PROJECT_ROOT
PROC = BASE / 'data/clinical/gastric/processed'
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
OUT.mkdir(parents=True, exist_ok=True)

# (gene_x, gene_y, display, role)
PAIRS = [
    ('TACSTD2', 'EPCAM',  'TROP2 ~ EPCAM',   'pan-epithelial control'),
    ('TACSTD2', 'KRT19',  'TROP2 ~ KRT19',   'pan-epithelial control'),
    ('TACSTD2', 'ERBB2',  'TROP2 ~ HER2',    'CIN/intestinal niche'),
    ('TACSTD2', 'CLDN18', 'TROP2 ~ CLDN18',  'foveolar-lineage state'),
    ('ERBB2',   'CLDN18', 'HER2 ~ CLDN18',   'context'),
]
GENES = sorted({g for p in PAIRS for g in p[:2]})

COHORTS = {
    'TCGA-STAD': 'TCGA_STAD_for_xcell.txt',
    'GSE66229':  'ACRG_GSE66229_for_xcell.txt',
    'GSE15459':  'ACRG_GSE15459_for_xcell.txt',
    'GSE34942':  'ACRG_GSE34942_for_xcell.txt',
    'GSE35809':  'ACRG_GSE35809_for_xcell.txt',
    'GSE51105':  'ACRG_GSE51105_for_xcell.txt',
    'GSE54129':  'ACRG_GSE54129_for_xcell.txt',
    'GSE57303':  'ACRG_GSE57303_for_xcell.txt',
    'GSE84437':  'ACRG_GSE84437_for_xcell.txt',
}


def gene_series(expr_file, wanted):
    out = {}
    with open(expr_file) as f:
        samples = f.readline().rstrip('\n').split('\t')[1:]
        for line in f:
            tab = line.index('\t')
            gene = line[:tab]
            if gene in wanted:
                vals = line[tab + 1:].rstrip('\n').split('\t')
                out[gene] = pd.Series([float(v) if v.strip() else np.nan for v in vals],
                                      index=samples)
    return out


def partial_spearman(x, y, Z):
    rx, ry = rankdata(x), rankdata(y)
    A = np.column_stack([np.ones(len(rx))] + [rankdata(z) for z in Z])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]
    df = len(rx) - 2 - len(Z)
    tt = r * np.sqrt(df / max(1e-12, 1 - r ** 2))
    return r, 2 * tdist.sf(abs(tt), df)


def main():
    rows = []
    for cohort, fname in COHORTS.items():
        expr = gene_series(BASE / 'data/processed/gastric' / fname, set(GENES))
        est_f = PROC / f'{cohort}_estimate.csv'
        if not est_f.exists():
            continue
        norm = lambda s: str(s).replace('.', '-')
        est = pd.read_csv(est_f).set_index('sample_id')
        est.index = est.index.map(norm)
        for g in GENES:
            if g in expr:
                expr[g].index = expr[g].index.map(norm)
                expr[g] = restrict_to_tumor(expr[g], cohort)
        present = [g for g in GENES if g in expr]
        common = est.index
        for g in present:
            common = common.intersection(expr[g].dropna().index)
        if len(common) < 40:
            print(f'[{cohort}] n<40 - skipped')
            continue
        sv = est.loc[common, 'stromal_score'].astype(float).values
        iv = est.loc[common, 'immune_score'].astype(float).values
        for gx, gy, disp, role in PAIRS:
            if gx not in present or gy not in present:
                rows.append({'cohort': cohort, 'pair': disp, 'role': role, 'n': 0})
                continue
            x = expr[gx].loc[common].astype(float).values
            y = expr[gy].loc[common].astype(float).values
            r0, p0 = spearmanr(x, y)
            rs, ps = partial_spearman(x, y, [sv])
            ri, pi = partial_spearman(x, y, [iv])
            rb, pb = partial_spearman(x, y, [sv, iv])
            rows.append({'cohort': cohort, 'pair': disp, 'role': role, 'n': len(common),
                         'raw_r': r0, 'raw_p': p0,
                         'adj_stroma_r': rs, 'adj_stroma_p': ps,
                         'adj_immune_r': ri, 'adj_immune_p': pi,
                         'adj_both_r': rb, 'adj_both_p': pb})
        print(f'[{cohort}] n={len(common)} done')

    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / 'F2_epithelial_targets_adjusted_stats.csv', index=False)
    ok = stats.dropna(subset=['raw_r'])

    print('\n=== Median across cohorts: raw -> |stroma ===')
    for _, _, disp, role in PAIRS:
        s = ok[ok['pair'] == disp]
        if len(s):
            print(f"  {disp:>16} ({role:>24}): raw {s['raw_r'].median():+.3f} -> "
                  f"|stroma {s['adj_stroma_r'].median():+.3f} "
                  f"(pos {int((s['adj_stroma_r'] > 0).sum())}/{len(s)}) -> "
                  f"|both {s['adj_both_r'].median():+.3f}")

    # --- figure: 1x5 per-cohort dumbbells raw -> stroma-adjusted ---
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 4.6), sharey=True)
    for ax, (_, _, disp, role) in zip(axes, PAIRS):
        s = ok[ok['pair'] == disp].sort_values('adj_stroma_r')
        y = np.arange(len(s))
        for yi, (_, r) in zip(y, s.iterrows()):
            ax.plot([r['raw_r'], r['adj_stroma_r']], [yi, yi], color=PAL['gray'],
                    lw=1.6, zorder=1)
        ax.scatter(s['raw_r'], y, s=50, facecolor='none', edgecolor=PAL['low'],
                   lw=1.3, label='raw', zorder=2)
        sig = s['adj_stroma_p'] < 0.05
        ax.scatter(s.loc[sig, 'adj_stroma_r'], y[sig.values], s=54, color=PAL['tumor'],
                   label='adj. for stroma (purity)', zorder=3)
        ax.scatter(s.loc[~sig, 'adj_stroma_r'], y[(~sig).values], s=54, facecolor='none',
                   edgecolor=PAL['tumor'], lw=1.3, zorder=3)
        ax.axvline(0, color='black', lw=0.9, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(s['cohort'], fontsize=8.5)
        ax.set_title(f'{disp}\n({role})\nmedian {s["raw_r"].median():+.2f} → '
                     f'{s["adj_stroma_r"].median():+.2f}', fontsize=9.5, pad=6)
        ax.set_xlabel('Spearman ρ', fontsize=9)
        clean_ax(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.text(0.5, 0.978, 'Epithelial-target co-expression with TROP2: raw vs '
             'purity-adjusted (filled = adj. p<0.05)',
             ha='center', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.928, 'Adjustment REMOVES the positive shared-purity component '
             '(high-purity tubes carry more of every epithelial gene) - ',
             ha='center', fontsize=9, style='italic', color='#333333')
    fig.text(0.5, 0.893, 'so medians shift DOWN; what survives is genuine '
             'tumor-intrinsic co-expression, not dilution.',
             ha='center', fontsize=9, style='italic', color='#333333')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.855),
               ncol=2, frameon=False, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.82])
    savefig(fig, OUT / 'F2_epithelial_targets_adjusted')


if __name__ == '__main__':
    main()

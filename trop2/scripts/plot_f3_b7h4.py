"""TROP2 and B7-H4 (VTCN1) co-expression."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu, fisher_exact

from cohort_utils import restrict_to_tumor
from style import PAL, clean_ax, savefig, fmt_p, stars
from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
OUT.mkdir(parents=True, exist_ok=True)

TROP2 = 'TACSTD2'
B7H4 = 'VTCN1'

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
}


def load_two_genes(expr_file, g1, g2):
    found = {}
    with open(expr_file) as f:
        samples = f.readline().rstrip('\n').split('\t')[1:]
        for line in f:
            tab = line.index('\t')
            gene = line[:tab]
            if gene in (g1, g2):
                vals = line[tab + 1:].rstrip('\n').split('\t')
                found[gene] = [float(v) if v.strip() else np.nan for v in vals]
                if len(found) == 2:
                    break
    if g1 not in found or g2 not in found:
        return None
    return pd.DataFrame(found, index=samples)


def main():
    rows = []
    for cohort, rel in COHORTS.items():
        df = load_two_genes(BASE / rel, TROP2, B7H4)
        if df is None:
            print(f'[{cohort}] missing {TROP2} or {B7H4} - skipped')
            continue
        df = restrict_to_tumor(df, cohort).dropna()
        if len(df) < 20:
            print(f'[{cohort}] n<20 - skipped')
            continue
        rho, p = spearmanr(df[TROP2], df[B7H4])

        t_med, b_med = df[TROP2].median(), df[B7H4].median()
        t_hi = df[TROP2] >= t_med
        b_hi = df[B7H4] >= b_med
        hh = int((t_hi & b_hi).sum()); hl = int((t_hi & ~b_hi).sum())
        lh = int((~t_hi & b_hi).sum()); ll = int((~t_hi & ~b_hi).sum())
        pct_combo = hh / (hh + hl) if (hh + hl) else np.nan  # B7H4-hi among TROP2-hi
        try:
            or_, fp = fisher_exact([[hh, hl], [lh, ll]])
        except ValueError:
            or_, fp = np.nan, np.nan

        b_in_thi = df.loc[t_hi, B7H4].values
        b_in_tlo = df.loc[~t_hi, B7H4].values
        _, mw_p = mannwhitneyu(b_in_thi, b_in_tlo, alternative='two-sided')
        delta = float(np.median(b_in_thi) - np.median(b_in_tlo))

        rows.append({'cohort': cohort, 'n': len(df), 'spearman_rho': rho, 'spearman_p': p,
                     'pct_b7h4hi_among_trop2hi': pct_combo, 'fisher_or': or_, 'fisher_p': fp,
                     'vtcn1_delta_trop2hi_minus_lo': delta, 'vtcn1_mw_p': mw_p})
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f'[{cohort}] n={len(df)} rho={rho:+.2f}{sig} | B7H4-hi among TROP2-hi={pct_combo:.0%} '
              f'(OR={or_:.2f}) | VTCN1 delta(hi-lo)={delta:+.2f} (MW p={mw_p:.2g})')

    if not rows:
        print('Nothing to plot.')
        return
    stats = pd.DataFrame(rows).sort_values('spearman_rho', ascending=True)
    stats.to_csv(OUT / 'F3_B7H4_TROP2_coexpr_stats.csv', index=False)

    # --- figure: (left) Spearman forest, (right) TROP2-high split into B7H4 hi/lo ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 0.42 * len(stats) + 2.2))
    fig.suptitle('TROP2 × B7-H4 (VTCN1) co-expression',
                 fontsize=12, fontweight='bold', y=1.02)

    y = np.arange(len(stats))
    colors = [PAL['combo'] if r > 0 else PAL['tumor'] for r in stats['spearman_rho']]
    axL.barh(y, stats['spearman_rho'], color=colors, alpha=0.8)
    for yi, (rho, p) in enumerate(zip(stats['spearman_rho'], stats['spearman_p'])):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        axL.text(rho + (0.01 if rho >= 0 else -0.01), yi, f'{rho:+.2f} {sig}',
                 va='center', ha='left' if rho >= 0 else 'right', fontsize=8)
    axL.set_yticks(y); axL.set_yticklabels(stats['cohort'], fontsize=9)
    axL.axvline(0, color='black', lw=0.8)
    axL.set_xlabel('Spearman ρ (VTCN1 vs TACSTD2)', fontsize=9)
    axL.set_title('(a) Co-expression', fontsize=10, pad=6)
    clean_ax(axL)
    xmax = max(0.05, stats['spearman_rho'].abs().max())
    axL.set_xlim(-xmax * 1.3, xmax * 1.3)

    # right: among TROP2-high tumors, fraction that are B7-H4-high (combo-eligible)
    combo = stats['pct_b7h4hi_among_trop2hi'].values
    axR.barh(y, combo, color=PAL['b7h4'], alpha=0.85)
    axR.barh(y, 1 - combo, left=combo, color=PAL['gray'], alpha=0.8)
    for yi, c in enumerate(combo):
        axR.text(c / 2, yi, f'{c:.0%}', va='center', ha='center', fontsize=8,
                 color='white', fontweight='bold')
    axR.set_yticks(y); axR.set_yticklabels(stats['cohort'], fontsize=9)
    axR.set_xlim(0, 1)
    axR.set_xlabel('Fraction of TROP2-high tumors', fontsize=10)
    axR.set_title('(b) Combo-eligible', fontsize=10, pad=6)
    clean_ax(axR)

    plt.tight_layout()
    savefig(fig, OUT / 'F3_B7H4_TROP2_coexpr')
    print(f'  stats: {OUT / "F3_B7H4_TROP2_coexpr_stats.csv"}')


if __name__ == '__main__':
    main()

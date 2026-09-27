"""TROP2 vs immune cells, adjusted for tumor purity."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

from cohort_utils import restrict_to_tumor
from style import (PAL, clean_ax, savefig, apply_pub_style, axis_label,
                   LABEL_FS, TITLE_FS, TICK_FS, LEGEND_FS, ANNOT_FS)
from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT
PROC = BASE / 'data/clinical/gastric/processed'
XCELL = BASE / 'data/xcell_output/gastric'
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
OUT.mkdir(parents=True, exist_ok=True)
LARGE_TEXT = '--large-text' in sys.argv
OUTPUT_STEM = ('F3_purity_adjusted_immune_large_text'
               if LARGE_TEXT else 'F3_purity_adjusted_immune')

TROP2 = 'TACSTD2'

PURITY = 'xcell'
if '--purity' in sys.argv:
    PURITY = sys.argv[sys.argv.index('--purity') + 1]

COHORTS = {
    'TCGA-STAD': ('TCGA_STAD_for_xcell.txt', 'TCGA_STAD_for__*.txt'),
    'GSE66229':  ('ACRG_GSE66229_for_xcell.txt', 'ACRG_GSE66229_for__*.txt'),
    'GSE15459':  ('ACRG_GSE15459_for_xcell.txt', 'ACRG_GSE15459_for__*.txt'),
    'GSE34942':  ('ACRG_GSE34942_for_xcell.txt', 'ACRG_GSE34942_for__*.txt'),
    'GSE35809':  ('ACRG_GSE35809_for_xcell.txt', 'ACRG_GSE35809_for__*.txt'),
    'GSE51105':  ('ACRG_GSE51105_for_xcell.txt', 'ACRG_GSE51105_for__*.txt'),
    'GSE54129':  ('ACRG_GSE54129_for_xcell.txt', 'ACRG_GSE54129_for__*.txt'),
    'GSE57303':  ('ACRG_GSE57303_for_xcell.txt', 'ACRG_GSE57303_for__*.txt'),
    'GSE84437':  ('ACRG_GSE84437_for_xcell.txt', 'ACRG_GSE84437_for__*.txt'),
}

ADAPTIVE_CELLS = ['CD8+ T-cells', 'CD4+ T-cells', 'B-cells', 'Tregs',
                  'CD8+ Tem', 'CD4+ Tem', 'Th1 cells']


def trop2_series(expr_file):
    with open(expr_file) as f:
        samples = f.readline().rstrip('\n').split('\t')[1:]
        for line in f:
            tab = line.index('\t')
            if line[:tab] == TROP2:
                vals = line[tab + 1:].rstrip('\n').split('\t')
                return pd.Series([float(v) if v.strip() else np.nan for v in vals], index=samples)
    return None


def load_xcell(glob_pat):
    files = sorted(XCELL.glob(glob_pat))
    if not files:
        return None
    return pd.read_csv(files[0], sep='\t', index_col=0)


def partial_spearman(x, y, z):
    """First-order partial Spearman r(x,y | z) via rank correlations."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return (rxy - rxz * ryz) / denom if denom > 0 else np.nan


def stromal_proxy(cohort, xcell_df):
    if PURITY == 'estimate':
        f = PROC / f'{cohort}_estimate.csv'
        if not f.exists():
            return None
        return pd.read_csv(f).set_index('sample_id')['stromal_score']
    if 'StromaScore' not in xcell_df.index:
        return None
    return xcell_df.loc['StromaScore']


def main():
    apply_pub_style()
    print(f'Purity proxy: {PURITY} (controlling for STROMAL content)')
    rows = []
    confound = []
    for cohort, (expr_rel, glob_pat) in COHORTS.items():
        trop2 = trop2_series(BASE / f'data/processed/gastric/{expr_rel}')
        xc = load_xcell(glob_pat)
        if trop2 is None or xc is None:
            print(f'[{cohort}] missing expression or xCell - skipped')
            continue
        norm = lambda s: str(s).replace('.', '-')
        trop2.index = trop2.index.map(norm)
        xc.columns = [norm(c) for c in xc.columns]
        stroma = stromal_proxy(cohort, xc)
        if stroma is None:
            print(f'[{cohort}] no stromal proxy ({PURITY}) - skipped')
            continue
        stroma.index = stroma.index.map(norm)

        common = trop2.index.intersection(xc.columns).intersection(stroma.index)
        trop2c = restrict_to_tumor(trop2.loc[common], cohort)
        common = trop2c.dropna().index
        if len(common) < 25:
            print(f'[{cohort}] n<25 common tumor samples - skipped')
            continue
        t = trop2.loc[common].astype(float)
        s = stroma.loc[common].astype(float)

        crho, _ = spearmanr(t, s)
        confound.append((cohort, crho))

        for cell in ADAPTIVE_CELLS:
            if cell not in xc.index:
                continue
            c = pd.to_numeric(xc.loc[cell, common], errors='coerce').astype(float)
            ok = c.notna() & t.notna() & s.notna()
            if ok.sum() < 25:
                continue
            raw_r, raw_p = spearmanr(t[ok], c[ok])
            part_r = partial_spearman(c[ok].values, t[ok].values, s[ok].values)
            rows.append({'cohort': cohort, 'cell_type': cell, 'n': int(ok.sum()),
                         'raw_r': raw_r, 'raw_p': raw_p, 'partial_r': part_r})
        print(f'[{cohort}] TROP2~stroma rho={crho:+.2f}  (n={len(common)})')

    if not rows:
        print('Nothing computed.')
        return
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / f'{OUTPUT_STEM}_stats.csv', index=False)

    summ = stats.groupby('cell_type').agg(
        raw=('raw_r', 'median'), partial=('partial_r', 'median'),
        raw_lo=('raw_r', lambda x: x.quantile(.25)),
        raw_hi=('raw_r', lambda x: x.quantile(.75)),
        n_cohorts=('cohort', 'nunique')
    ).reindex([c for c in ADAPTIVE_CELLS if c in stats['cell_type'].values])

    # Sort by raw effect (most depleted at bottom, natural reading for cold TME)
    summ = summ.sort_values('raw', ascending=True)

    n_cells = len(summ)
    n_coh = len(confound)
    bump = 2.3 if LARGE_TEXT else 1.8
    # Tight canvas: left dumbbell dominates, right bar is a slim inset
    fig_h = max(3.2, 0.38 * n_cells + 1.4)
    fig, (ax, axc) = plt.subplots(
        1, 2, figsize=(8.4, fig_h),
        gridspec_kw={'width_ratios': [2.6, 1.0], 'wspace': 0.28},
    )

    y = np.arange(n_cells)
    raw_c = PAL['tumor']
    adj_c = PAL['low']
    for yi, (cell, r) in enumerate(summ.iterrows()):
        ax.plot([r['raw'], r['partial']], [yi, yi], color='#B0B0B0', lw=1.8, zorder=1)
        ax.scatter(r['raw'], yi, s=60, color=raw_c, zorder=3, edgecolors='white',
                   linewidths=0.4, label='raw' if yi == 0 else None)
        ax.scatter(r['partial'], yi, s=60, color=adj_c, zorder=3, edgecolors='white',
                   linewidths=0.4, label='purity-adjusted' if yi == 0 else None)
        # end-cap value labels (outside the segment, no collision)
        left_v, right_v = (r['raw'], r['partial']) if r['raw'] <= r['partial'] else (r['partial'], r['raw'])
        left_c = raw_c if r['raw'] <= r['partial'] else adj_c
        right_c = adj_c if r['raw'] <= r['partial'] else raw_c
        ax.text(left_v - 0.012, yi, f'{left_v:+.2f}',
                ha='right', va='center',
                fontsize=ANNOT_FS - 1 + bump, color=left_c)
        ax.text(right_v + 0.012, yi, f'{right_v:+.2f}',
                ha='left', va='center',
                fontsize=ANNOT_FS - 1 + bump, color=right_c)

    ax.axvline(0, color='black', lw=0.9, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(summ.index, fontsize=TICK_FS + 1 + bump)
    ax.set_xlabel(axis_label('Spearman r with TROP2', unit='median across cohorts'),
                  fontsize=LABEL_FS + bump)
    ax.set_title('Adaptive immunity (T/B): raw vs purity-adjusted',
                 fontsize=TITLE_FS - 1 + bump, pad=4)
    # tight x-limits around the data (kill the empty right half)
    xmin = min(summ['raw'].min(), summ['partial'].min())
    xmax = max(summ['raw'].max(), summ['partial'].max())
    pad = 0.09  # room for end-cap value labels
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(-0.55, n_cells - 0.45)
    ax.legend(fontsize=LEGEND_FS + bump, loc='lower right', frameon=False,
              handletextpad=0.3, borderaxespad=0.2)
    clean_ax(ax)
    ax.tick_params(axis='x', labelsize=TICK_FS + bump)

    cf = pd.DataFrame(confound, columns=['cohort', 'rho']).sort_values('rho')
    colors = [PAL['stroma'] if r < 0 else PAL['gray'] for r in cf['rho']]
    axc.barh(np.arange(n_coh), cf['rho'], color=colors, height=0.72, edgecolor='none')
    axc.set_yticks(np.arange(n_coh))
    axc.set_yticklabels(cf['cohort'], fontsize=TICK_FS - 0.5 + bump)
    axc.axvline(0, color='black', lw=0.9)
    axc.set_xlabel(axis_label('ρ(TROP2, stroma)'),
                   fontsize=LABEL_FS - 1 + bump)
    axc.set_title('TROP2-high = less stroma',
                  fontsize=TITLE_FS - 2 + bump, pad=4)
    axc.set_ylim(-0.6, n_coh - 0.4)
    # tight x around data
    rx = cf['rho']
    axc.set_xlim(min(rx.min() - 0.05, -0.05), max(rx.max() + 0.05, 0.05))
    clean_ax(axc)
    axc.tick_params(axis='x', labelsize=TICK_FS - 0.5 + bump)

    fig.subplots_adjust(left=0.18, right=0.98, top=0.90, bottom=0.14, wspace=0.32)
    savefig(fig, OUT / OUTPUT_STEM, pad_inches=0.04)
    print('\nSummary (median across cohorts):')
    print(summ[['raw', 'partial', 'n_cohorts']].round(3).to_string())


if __name__ == '__main__':
    main()

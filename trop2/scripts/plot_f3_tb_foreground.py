"""T- and B-cell scores in TROP2-high vs low."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind

from cohort_utils import restrict_to_tumor
from style import (PAL, clean_ax, savefig, apply_pub_style,
                   LABEL_FS, TITLE_FS, TICK_FS, LEGEND_FS)
from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT
XCELL = BASE / 'data/xcell_output/gastric'
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
OUT.mkdir(parents=True, exist_ok=True)
LARGE_TEXT = '--large-text' in sys.argv
OUTPUT_STEM = 'F3_TB_foreground_large_text' if LARGE_TEXT else 'F3_TB_foreground'

TROP2 = 'TACSTD2'

# Functional order within each compartment (mirrors Fig2 ordering).
T_CELLS = ['CD4+ T-cells', 'CD4+ naive T-cells', 'CD4+ memory T-cells', 'CD4+ Tcm', 'CD4+ Tem',
           'Th1 cells', 'Th2 cells', 'CD8+ T-cells', 'CD8+ naive T-cells', 'CD8+ Tcm', 'CD8+ Tem',
           'Tgd cells', 'Tregs']
B_CELLS = ['B-cells', 'naive B-cells', 'Memory B-cells', 'Class-switched memory B-cells',
           'pro B-cells', 'Plasma cells']
CELLS = T_CELLS + B_CELLS
GAP = 1.0  # x-gap between T-cell and B-cell blocks

RED, BLUE = PAL['increased'], PAL['decreased']

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


def trop2_series(expr_file):
    with open(expr_file) as f:
        samples = f.readline().rstrip('\n').split('\t')[1:]
        for line in f:
            tab = line.index('\t')
            if line[:tab] == TROP2:
                vals = line[tab + 1:].rstrip('\n').split('\t')
                return pd.Series([float(v) if v.strip() else np.nan for v in vals], index=samples)
    return None


def main():
    norm = lambda s: str(s).replace('.', '-')
    tmat = pd.DataFrame(index=CELLS, columns=list(COHORTS.keys()), dtype=float)
    pmat = pd.DataFrame(index=CELLS, columns=list(COHORTS.keys()), dtype=float)
    n_by_cohort, rows = {}, []

    for cohort, (expr_rel, glob_pat) in COHORTS.items():
        trop2 = trop2_series(BASE / f'data/processed/gastric/{expr_rel}')
        files = sorted(XCELL.glob(glob_pat))
        if trop2 is None or not files:
            print(f'[{cohort}] missing - skipped'); continue
        xc = pd.read_csv(files[0], sep='\t', index_col=0)
        trop2.index = trop2.index.map(norm)
        xc.columns = [norm(c) for c in xc.columns]
        common = restrict_to_tumor(trop2, cohort).dropna().index.intersection(xc.columns)
        if len(common) < 25:
            print(f'[{cohort}] n<25 - skipped'); continue
        t = trop2.loc[common].astype(float)
        med = t.median()
        hi, lo = t[t >= med].index, t[t < med].index
        n_by_cohort[cohort] = len(common)
        for cell in CELLS:
            if cell not in xc.index:
                continue
            vh = pd.to_numeric(xc.loc[cell, hi], errors='coerce').dropna()
            vl = pd.to_numeric(xc.loc[cell, lo], errors='coerce').dropna()
            if len(vh) < 5 or len(vl) < 5:
                continue
            tstat, p = ttest_ind(vh, vl, equal_var=False)
            tmat.loc[cell, cohort] = tstat
            pmat.loc[cell, cohort] = p
            rows.append({'cohort': cohort, 'cell_type': cell, 't_stat': tstat, 'p_value': p,
                         'compartment': 'T' if cell in T_CELLS else 'B'})
        print(f'[{cohort}] n={len(common)}')

    pd.DataFrame(rows).to_csv(OUT / f'{OUTPUT_STEM}_stats.csv', index=False)

    # cohort rows largest-first (Fig2 convention)
    cohort_order = sorted(n_by_cohort, key=lambda c: -n_by_cohort[c])
    # x positions with a gap between T and B blocks
    xpos = {cell: i for i, cell in enumerate(T_CELLS)}
    for j, cell in enumerate(B_CELLS):
        xpos[cell] = len(T_CELLS) + GAP + j
    xmax = max(xpos.values())

    apply_pub_style()
    bump = 2.3 if LARGE_TEXT else 1.8
    fig, ax = plt.subplots(
        figsize=(12.15, 8.85)
    )
    for ci, cohort in enumerate(cohort_order):
        for cell in CELLS:
            t, p = tmat.loc[cell, cohort], pmat.loc[cell, cohort]
            if pd.isna(t):
                continue
            ax.plot(xpos[cell], ci, 'o', markersize=np.sqrt(abs(t) * 80),
                    color=(RED if t > 0 else BLUE), alpha=(1.0 if p < 0.05 else 0.2),
                    markeredgewidth=0)

    ax.set_xticks([xpos[c] for c in CELLS])
    ax.set_xticklabels(CELLS, rotation=90, ha='center', fontsize=TICK_FS + bump)
    ax.set_yticks(range(len(cohort_order)))
    ax.set_yticklabels([f'{c} (n={n_by_cohort[c]})' for c in cohort_order],
                       fontsize=TICK_FS + 1 + bump)
    ax.set_xlim(-0.6, xmax + 0.4)
    ax.set_ylim(-0.6, len(cohort_order) - 0.4)
    ax.invert_yaxis()
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color('black'); sp.set_linewidth(0.6)

    xt = ax.get_xaxis_transform()
    ax.text((len(T_CELLS) - 1) / 2, 1.015, 'T cells', transform=xt, ha='center', va='bottom',
            fontsize=LABEL_FS + bump, fontweight='bold', color='#333333')
    ax.text(len(T_CELLS) + GAP + (len(B_CELLS) - 1) / 2, 1.015, 'B cells', transform=xt,
            ha='center', va='bottom', fontsize=LABEL_FS + bump,
            fontweight='bold', color='#333333')

    color_handles = [mpatches.Patch(facecolor=RED, label='increased in TROP2-high'),
                     mpatches.Patch(facecolor=BLUE, label='decreased in TROP2-high')]
    sig_handles = [mpatches.Patch(facecolor=PAL['gray'], alpha=1.0, label='p < 0.05'),
                   mpatches.Patch(facecolor=PAL['gray'], alpha=0.2, label='n.s.')]
    size_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                               markersize=np.sqrt(tv * 80), markeredgewidth=0, label=f'|t|={tv}')
                    for tv in [1, 2, 3, 5]]
    leg1 = ax.legend(handles=color_handles, loc='upper left', bbox_to_anchor=(1.005, 1.00),
                     frameon=False, fontsize=LEGEND_FS + bump)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=sig_handles, loc='upper left', bbox_to_anchor=(1.005, 0.78),
                     frameon=False, fontsize=LEGEND_FS + bump)
    ax.add_artist(leg2)
    leg3 = ax.legend(handles=size_handles, loc='upper left', bbox_to_anchor=(1.005, 0.58),
                     frameon=False, title='|t-stat| (Welch)',
                     fontsize=LEGEND_FS + bump)
    leg3.get_title().set_fontsize(LEGEND_FS + bump)

    ax.set_title('TROP2-high vs low: T/B-cell compartments',
                 fontsize=TITLE_FS + bump, fontweight='bold',
                 pad=31)
    ax.set_xlabel('xCell cell type', fontsize=LABEL_FS + bump)
    ax.set_ylabel('Cohort', fontsize=LABEL_FS + bump)
    plt.tight_layout()
    savefig(fig, OUT / OUTPUT_STEM, extra_artists=[leg1, leg2, leg3])


if __name__ == '__main__':
    main()

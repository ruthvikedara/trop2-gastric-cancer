"""TROP2 vs immune cells within subtype."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from cohort_utils import restrict_to_tumor
from style import PAL, clean_ax, savefig
from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT
PROC = BASE / 'data/clinical/gastric/processed'
XCELL = BASE / 'data/xcell_output/gastric'
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
OUT.mkdir(parents=True, exist_ok=True)

TROP2 = 'TACSTD2'
CELLS = ['CD8+ T-cells', 'CD4+ T-cells', 'B-cells', 'NK cells']

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
    rows = []
    for cohort, (expr_rel, glob_pat) in COHORTS.items():
        trop2 = trop2_series(BASE / f'data/processed/gastric/{expr_rel}')
        files = sorted(XCELL.glob(glob_pat))
        sub_f = PROC / f'{cohort}_subtypes_predicted.csv'
        if trop2 is None or not files or not sub_f.exists():
            print(f'[{cohort}] missing input - skipped')
            continue
        xc = pd.read_csv(files[0], sep='\t', index_col=0)
        sub = pd.read_csv(sub_f)
        if 'EMP_subtype' not in sub.columns:
            print(f'[{cohort}] no EMP label - skipped')
            continue
        trop2.index = trop2.index.map(norm)
        xc.columns = [norm(c) for c in xc.columns]
        emp = sub.set_index(sub['sample_id'].map(norm))['EMP_subtype']

        common = restrict_to_tumor(trop2, cohort).dropna().index.intersection(xc.columns).intersection(emp.index)
        ep = [s for s in common if emp.get(s) == 'EP']
        if len(common) < 25 or len(ep) < 25:
            print(f'[{cohort}] n<25 (all={len(common)}, EP={len(ep)}) - skipped')
            continue
        t_all = trop2.loc[common].astype(float)
        t_ep = trop2.loc[ep].astype(float)

        for cell in CELLS:
            if cell not in xc.index:
                continue
            c_all = pd.to_numeric(xc.loc[cell, common], errors='coerce').astype(float)
            c_ep = pd.to_numeric(xc.loc[cell, ep], errors='coerce').astype(float)
            r_all, _ = spearmanr(t_all, c_all, nan_policy='omit')
            r_ep, p_ep = spearmanr(t_ep, c_ep, nan_policy='omit')
            rows.append({'cohort': cohort, 'cell_type': cell, 'r_overall': r_all,
                         'r_ep_only': r_ep, 'p_ep_only': p_ep, 'n_ep': len(ep)})
        print(f'[{cohort}] all={len(common)}, EP={len(ep)}')

    if not rows:
        print('Nothing computed.')
        return
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / 'F3_subtype_stratified_stats.csv', index=False)

    summ = stats.groupby('cell_type').agg(overall=('r_overall', 'median'),
                                          ep_only=('r_ep_only', 'median')).reindex(CELLS)

    fig, ax = plt.subplots(figsize=(8, 0.7 * len(CELLS) + 2))
    y = np.arange(len(summ))
    for yi, (cell, r) in enumerate(summ.iterrows()):
        ax.plot([r['overall'], r['ep_only']], [yi, yi], color='#bbbbbb', lw=2, zorder=1)
        ax.scatter(r['overall'], yi, s=90, color=PAL['gray'], zorder=3,
                   label='all tumors' if yi == 0 else None)
        ax.scatter(r['ep_only'], yi, s=90, color=PAL['ep'], zorder=3,
                   label='epithelial (EP) only' if yi == 0 else None)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(summ.index, fontsize=10)
    ax.set_xlabel('Spearman r with TROP2 (median across cohorts)', fontsize=10)
    ax.set_title('Immune-cold within epithelial subtype', fontsize=11, pad=8)
    ax.legend(fontsize=9, loc='lower right')
    clean_ax(ax)

    plt.tight_layout()
    savefig(fig, OUT / 'F3_subtype_stratified')
    print('\nMedian across cohorts (overall vs EP-only):')
    print(summ.round(3).to_string())


if __name__ == '__main__':
    main()

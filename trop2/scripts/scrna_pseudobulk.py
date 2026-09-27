"""Pseudobulk TROP2 vs B7-H4."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import PAL, clean_ax, savefig

from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT
PROC = BASE / 'data/processed/scrna/gastric'
OUT = BASE / 'trop2/plots/scrna'

DATASETS = {'Kumar_GSE183904': PROC / 'kumar_tumor.h5ad',
            'Sathe_GSE150290': PROC / 'sathe_tumor.h5ad'}
MIN_EPI = 20  # min epithelial cells per tumor to include


def pseudobulk(name, path):
    a = sc.read_h5ad(path)
    epi = a[a.obs.lineage == 'Epithelial']
    counts = epi.layers['counts']                      # raw counts, cells x genes
    gi = {g: epi.var_names.get_loc(g) for g in ('TACSTD2', 'VTCN1')}
    tot = np.asarray(counts.sum(axis=1)).ravel()       # per-cell library size
    df = pd.DataFrame({
        'sample': epi.obs['sample'].values,
        'TACSTD2': np.asarray(counts[:, gi['TACSTD2']].todense()).ravel(),
        'VTCN1':   np.asarray(counts[:, gi['VTCN1']].todense()).ravel(),
        'tot': tot,
    })
    g = df.groupby('sample').agg(TACSTD2=('TACSTD2', 'sum'), VTCN1=('VTCN1', 'sum'),
                                 tot=('tot', 'sum'), n_epi=('tot', 'size'))
    g = g[g.n_epi >= MIN_EPI].copy()
    g['TROP2'] = np.log1p(g.TACSTD2 / g.tot * 1e6)     # logCPM pseudobulk
    g['B7H4'] = np.log1p(g.VTCN1 / g.tot * 1e6)
    g['dataset'] = name
    return g.reset_index()


def main():
    frames = [pseudobulk(n, p) for n, p in DATASETS.items() if p.exists()]
    pb = pd.concat(frames, ignore_index=True)
    pb.to_csv(OUT / 'pseudobulk_B7H4_TROP2_stats.csv', index=False)

    per = {}
    for ds, sub in pb.groupby('dataset'):
        rho, p = spearmanr(sub.TROP2, sub.B7H4)
        per[ds] = (rho, p, len(sub))
        print(f'{ds}: pseudobulk Spearman rho={rho:+.2f}, p={p:.2g} (n={len(sub)} tumors)')
    # pooled (z-score within dataset to remove platform offset)
    pb['zT'] = pb.groupby('dataset')['TROP2'].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    pb['zB'] = pb.groupby('dataset')['B7H4'].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    rho_all, p_all = spearmanr(pb.zT, pb.zB)
    print(f'POOLED: rho={rho_all:+.2f}, p={p_all:.2g} (n={len(pb)} tumors)')

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    colors = {'Kumar_GSE183904': PAL['Kumar'], 'Sathe_GSE150290': PAL['Sathe']}
    for ds, sub in pb.groupby('dataset'):
        rho, p, n = per[ds]
        ax.scatter(sub.TROP2, sub.B7H4, s=70, alpha=0.8, color=colors.get(ds, 'gray'),
                   edgecolors='white', linewidths=0.5,
                   label=f"{ds.replace('_', ' ')}  (ρ={rho:+.2f}, n={n})")
    ax.set_xlabel('Pseudobulk TROP2 (TACSTD2), logCPM', fontsize=11)
    ax.set_ylabel('Pseudobulk B7-H4 (VTCN1), logCPM', fontsize=11)
    ax.set_title('B7-H4 vs TROP2 per tumor (pseudobulk)\n'
                 f'(pooled Spearman ρ={rho_all:+.2f}, p={p_all:.1e}, n={len(pb)} tumors)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='best', frameon=True)
    clean_ax(ax)
    fig.tight_layout()
    savefig(fig, OUT / 'pseudobulk_B7H4_TROP2')
    plt.close(fig)
    print(f'\nSaved {OUT}/pseudobulk_B7H4_TROP2.{{pdf,png}} + stats')


if __name__ == '__main__':
    main()

"""Single-cell validation of bulk findings."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import scanpy as sc
from scipy.stats import fisher_exact, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (
    PAL, clean_ax, savefig, fmt_p, _tint, apply_pub_style,
    TICK_FS, LABEL_FS, TITLE_FS,
)

from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT
PROC = BASE / 'data/processed/scrna/gastric'
OUT = BASE / 'trop2/plots/scrna'
OUT.mkdir(parents=True, exist_ok=True)

TROP2, B7H4 = 'TACSTD2', 'VTCN1'
IMMUNE = ['T/NK', 'B/Plasma', 'Myeloid', 'Mast']
LINEAGE_COLORS = {
    'B/Plasma': '#1f77b4',
    'Endothelial': '#ff7f0e',
    'Epithelial': '#2ca02c',
    'Fibroblast': '#d62728',
    'Mast': '#9467bd',
    'Myeloid': '#8c564b',
    'T/NK': '#e377c2',
    'Unassigned': '#7f7f7f',
}

DATASETS = {
    'Kumar_GSE183904': PROC / 'kumar_tumor.h5ad',
    'Sathe_GSE150290': PROC / 'sathe_tumor.h5ad',
}


def expr_vec(adata, gene, layer=None):
    x = adata[:, gene].layers[layer] if layer else adata[:, gene].X
    return np.asarray(x.todense()).ravel() if hasattr(x, 'todense') else np.asarray(x).ravel()


def positive_vmax(values, percentile):
    """Robust upper color limit based only on detected cells."""
    positive = np.asarray(values)[np.asarray(values) > 0]
    return float(np.percentile(positive, percentile)) if len(positive) else None


def plot_expression_umap(fig, ax, coords, values, title, vmax, point_size=0.45):
    """Draw detected cells over a quiet gray background so dropout does not
    dominate the color scale."""
    values = np.asarray(values)
    detected = values > 0
    ax.scatter(
        coords[~detected, 0], coords[~detected, 1], s=point_size * 0.7,
        color='#D5D5D5', alpha=0.35, linewidths=0, rasterized=True)
    order = np.where(detected)[0]
    order = order[np.argsort(values[order])]
    sca = ax.scatter(
        coords[order, 0], coords[order, 1], c=values[order],
        s=point_size, cmap='magma', vmin=0, vmax=vmax, linewidths=0,
        rasterized=True)
    cb = fig.colorbar(sca, ax=ax, fraction=0.035, pad=0.015)
    cb.set_label('Expression (log1p normalized)', fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold')
    ax.set_axis_off()


def run_dataset(name, path):
    adata = sc.read_h5ad(path)
    print(f'\n===== {name}: {adata.n_obs} cells =====')
    adata.obs['TROP2'] = expr_vec(adata, TROP2)
    adata.obs['TROP2_pos'] = expr_vec(adata, TROP2, layer='counts') > 0
    adata.obs['B7H4_pos'] = expr_vec(adata, B7H4, layer='counts') > 0
    adata.obs['B7H4'] = expr_vec(adata, B7H4)

    # V1: TROP2 by lineage
    by_lin = adata.obs.groupby('lineage', observed=True).agg(
        n=('TROP2', 'size'), mean_TROP2=('TROP2', 'mean'), pct_pos=('TROP2_pos', 'mean'))
    by_lin['pct_pos'] *= 100
    frac_epi_among_pos = 100 * (adata.obs.loc[adata.obs.TROP2_pos, 'lineage'] == 'Epithelial').mean()
    order = by_lin.sort_values('mean_TROP2', ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(7, 4))
    sc.pl.violin(adata, [TROP2], groupby='lineage', order=order, ax=ax, show=False,
                 rotation=40, stripplot=False)
    ax.set_title(f'V1 [{name}]: TROP2 epithelial-restricted', fontsize=11, fontweight='bold')
    fig.savefig(OUT / f'{name}_V1_TROP2_by_lineage.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    trop2_vmax = positive_vmax(adata.obs['TROP2'].values, 99)
    b7h4_vmax = positive_vmax(adata.obs['B7H4'].values, 95)
    print(f'  UMAP color limits: TROP2 positive-cell q99={trop2_vmax:.2f}; '
          f'B7-H4 positive-cell q95={b7h4_vmax:.2f}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    # lineage legend ON the clusters (avoids the side-legend overlapping the next panel)
    sc.pl.umap(adata, color='lineage', ax=axes[0], show=False, frameon=False,
               title=f'{name.replace("_", " ")}: cell lineage',
               legend_loc='on data', legend_fontsize=TICK_FS,
               legend_fontoutline=2)
    coords = np.asarray(adata.obsm['X_umap'])
    plot_expression_umap(
        fig, axes[1], coords, adata.obs['TROP2'].values,
        'TROP2', trop2_vmax)
    plot_expression_umap(
        fig, axes[2], coords, adata.obs['B7H4'].values,
        'B7-H4 (VTCN1)', b7h4_vmax)
    fig.subplots_adjust(wspace=0.30)
    savefig(fig, OUT / f'{name}_UMAP')

    # V2: co-expression in epithelial cells
    epi = adata[adata.obs.lineage == 'Epithelial']
    tp, bp = epi.obs.TROP2_pos.values, epi.obs.B7H4_pos.values
    a_ = int((tp & bp).sum()); b_ = int((tp & ~bp).sum())
    c_ = int((~tp & bp).sum()); d_ = int((~tp & ~bp).sum())
    or_, fp = fisher_exact([[a_, b_], [c_, d_]])
    pct_in_pos = 100 * a_ / (a_ + b_) if (a_ + b_) else np.nan
    pct_in_neg = 100 * c_ / (c_ + d_) if (c_ + d_) else np.nan
    rho, rp = spearmanr(expr_vec(epi, TROP2), expr_vec(epi, B7H4))

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.bar(['among\nTROP2+', 'among\nTROP2-'], [pct_in_pos, pct_in_neg], color=[PAL['gold'], _tint(PAL['gold'], 0.5)])
    for i, v in enumerate([pct_in_pos, pct_in_neg]):
        ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('% B7-H4 (VTCN1)+ epithelial cells')
    ax.set_title(f'V2 [{name}]: B7-H4+ enriched in TROP2+\nOR={or_:.1f}, p={fp:.1e} (n_epi={epi.n_obs})',
                 fontsize=10, fontweight='bold')
    clean_ax(ax)
    fig.savefig(OUT / f'{name}_V2_coexpression.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # V3 per-sample points
    per = []
    for s, sub in adata.obs.groupby('sample', observed=True):
        em = sub.lineage == 'Epithelial'
        if em.sum() < 20:
            continue
        per.append({'dataset': name, 'sample': s,
                    'mean_epi_TROP2': sub.loc[em, 'TROP2'].mean(),
                    'immune_frac': sub.lineage.isin(IMMUNE).mean()})

    metrics = {
        'dataset': name, 'n_cells': adata.n_obs, 'n_epithelial': int(epi.n_obs),
        'V1_pct_TROP2pos_that_are_epithelial': round(frac_epi_among_pos, 1),
        'V1_epi_mean_TROP2': round(by_lin.loc['Epithelial', 'mean_TROP2'], 3),
        'V1_epi_pct_pos': round(by_lin.loc['Epithelial', 'pct_pos'], 1),
        'V2_pct_double_pos': round(100 * a_ / epi.n_obs, 2),
        'V2_pct_B7H4_among_TROP2pos': round(pct_in_pos, 1),
        'V2_pct_B7H4_among_TROP2neg': round(pct_in_neg, 1),
        'V2_fisher_OR': round(or_, 2), 'V2_fisher_p': fp,
        'V2_singlecell_rho': round(rho, 3), 'V2_rho_p': rp,
    }
    print(f'  V1: {frac_epi_among_pos:.0f}% of TROP2+ cells are epithelial; epi pct_pos={by_lin.loc["Epithelial","pct_pos"]:.0f}%')
    print(f'  V2 (n_epi={epi.n_obs}): B7H4+ among TROP2+ {pct_in_pos:.1f}% vs TROP2- {pct_in_neg:.1f}% '
          f'(OR={or_:.2f}, p={fp:.1e}); single-cell rho={rho:+.3f}')
    umap_data = {
        'name': name.replace('_', ' '),
        'coords': np.asarray(adata.obsm['X_umap']).copy(),
        'lineage': adata.obs['lineage'].astype(str).to_numpy(),
        'TROP2': adata.obs['TROP2'].to_numpy(copy=True),
        'B7H4': adata.obs['B7H4'].to_numpy(copy=True),
        'TROP2_vmax': trop2_vmax,
        'B7H4_vmax': b7h4_vmax,
    }
    return metrics, pd.DataFrame(per), umap_data


def combined_umap_panel(plot_rows):
    """Two-dataset UMAP validation with rasterized points and vector text."""
    n_rows = len(plot_rows)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(14.5, 4.15 * n_rows), squeeze=False)
    for ri, row in enumerate(plot_rows):
        xy = row['coords']
        lineage = row['lineage']
        ax = axes[ri, 0]
        for label in LINEAGE_COLORS:
            keep = lineage == label
            if not keep.any():
                continue
            ax.scatter(
                xy[keep, 0], xy[keep, 1], s=0.35,
                color=LINEAGE_COLORS[label], linewidths=0,
                rasterized=True)
            center = np.median(xy[keep], axis=0)
            ax.text(
                center[0], center[1], label, ha='center', va='center',
                fontsize=TICK_FS, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
        ax.set_title(f'{row["name"]}\nCell lineage', fontsize=TITLE_FS,
                     fontweight='bold')
        ax.set_axis_off()

        for ci, (key, title, vmax) in enumerate((
                ('TROP2', 'TROP2', row['TROP2_vmax']),
                ('B7H4', 'B7-H4 (VTCN1)', row['B7H4_vmax'])), start=1):
            ax = axes[ri, ci]
            plot_expression_umap(
                fig, ax, xy, row[key], title, vmax, point_size=0.35)

    fig.suptitle(
        'TROP2 and B7-H4 signals are enriched in the epithelial compartment',
        fontsize=13, fontweight='bold', y=0.995)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.93, bottom=0.02,
                        wspace=0.16, hspace=0.18)
    savefig(fig, OUT / 'UMAP_lineage_TROP2_B7H4')


def main():
    apply_pub_style()
    rows, v3_frames, umap_rows = [], [], []
    for name, path in DATASETS.items():
        if not path.exists():
            print(f'[{name}] h5ad not found - skipped')
            continue
        m, v3, umap_data = run_dataset(name, path)
        rows.append(m); v3_frames.append(v3); umap_rows.append(umap_data)

    if not rows:
        print('No datasets processed.')
        return
    summ = pd.DataFrame(rows).set_index('dataset')
    summ.to_csv(OUT / 'scrna_validation_summary.csv')
    print('\n=== PER-DATASET SUMMARY ===')
    print(summ.T.to_string())
    combined_umap_panel(umap_rows)

    # pooled V3 (z-score TROP2 within dataset, then pool)
    v3 = pd.concat(v3_frames, ignore_index=True)
    v3['z_TROP2'] = v3.groupby('dataset')['mean_epi_TROP2'].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else x * 0)
    if len(v3) >= 8:
        rho_all, p_all = spearmanr(v3['z_TROP2'], v3['immune_frac'])
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        colors = {'Kumar_GSE183904': PAL['Kumar'], 'Sathe_GSE150290': PAL['Sathe']}
        for ds, sub in v3.groupby('dataset'):
            ax.scatter(sub.z_TROP2, 100 * sub.immune_frac, s=55,
                       color=colors.get(ds, 'gray'), label=ds.replace('_', ' '), alpha=0.8)
        # trend line
        b, a = np.polyfit(v3.z_TROP2, 100 * v3.immune_frac, 1)
        xs = np.linspace(v3.z_TROP2.min(), v3.z_TROP2.max(), 50)
        ax.plot(xs, a + b * xs, color='black', lw=1.2, ls='--')
        ax.set_xlabel('Per-tumor epithelial TROP2 (z-scored within dataset)', fontsize=9.5)
        ax.set_ylabel('% immune cells per tumor', fontsize=9.5)
        ax.set_title(
            f'Per-tumor epithelial TROP2 vs immune-cell fraction (n={len(v3)})\n'
            f'Spearman ρ={rho_all:+.2f}, {fmt_p(p_all)}',
            fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        clean_ax(ax)
        savefig(fig, OUT / 'combined_V3_TROP2_vs_immune')
        plt.close(fig)
        # per-dataset V3 too
        per_ds = {ds: spearmanr(s.mean_epi_TROP2, s.immune_frac)[0] for ds, s in v3.groupby('dataset')}
        print(f'\nV3 pooled (n={len(v3)} tumors): Spearman rho={rho_all:+.2f}, p={p_all:.2g}')
        for ds, r in per_ds.items():
            print(f'   {ds}: rho={r:+.2f} (n={ (v3.dataset==ds).sum() })')
        v3.to_csv(OUT / 'combined_V3_persample.csv', index=False)

    print(f'\nSaved outputs to {OUT}')


if __name__ == '__main__':
    main()

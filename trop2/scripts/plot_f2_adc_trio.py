"""TROP2 correlation with HER2, Claudin-18 and PD-L1."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, chi2, norm, t as t_dist

from cohort_utils import restrict_to_tumor
from style import PAL, apply_pub_style, clean_ax, savefig, fmt_p
from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
OUT.mkdir(parents=True, exist_ok=True)


def re_meta_correlation(rhos, ns):
    """Random-effects (DerSimonian-Laird) meta-analysis of Spearman correlations.

    Per-cohort rho -> Fisher z (var = 1/(n-3)) -> DL pooling -> back-transform.
    Returns a dict with pooled rho + CI, tau2, I2, Q, Q p-value, and a 95%
    prediction interval (back-transformed to the rho scale).
    """
    rhos = np.asarray(rhos, dtype=float)
    ns = np.asarray(ns, dtype=float)
    k = len(rhos)
    z = np.arctanh(np.clip(rhos, -0.999999, 0.999999))
    v = 1.0 / (ns - 3.0)
    w = 1.0 / v

    z_fixed = np.sum(w * z) / np.sum(w)
    Q = np.sum(w * (z - z_fixed) ** 2)
    df = k - 1
    Q_p = chi2.sf(Q, df) if df > 0 else np.nan
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0

    w_star = 1.0 / (v + tau2)
    z_pooled = np.sum(w_star * z) / np.sum(w_star)
    var_pooled = 1.0 / np.sum(w_star)
    se_pooled = np.sqrt(var_pooled)

    z_lo, z_hi = z_pooled - 1.96 * se_pooled, z_pooled + 1.96 * se_pooled
    rho_pooled = np.tanh(z_pooled)
    rho_lo, rho_hi = np.tanh(z_lo), np.tanh(z_hi)

    if k > 2:
        t_crit = t_dist.ppf(0.975, df=k - 2)
        pi_half = t_crit * np.sqrt(var_pooled + tau2)
        pi_lo, pi_hi = np.tanh(z_pooled - pi_half), np.tanh(z_pooled + pi_half)
    else:
        pi_lo, pi_hi = np.nan, np.nan

    return {
        'rho_pooled': rho_pooled, 'rho_lo': rho_lo, 'rho_hi': rho_hi,
        'z_pooled': z_pooled, 'se_pooled': se_pooled,
        'tau2': tau2, 'I2': I2, 'Q': Q, 'Q_df': df, 'Q_p': Q_p,
        'pi_lo': pi_lo, 'pi_hi': pi_hi,
        'meta_p': float(2 * norm.sf(abs(z_pooled / se_pooled))),
    }

GENES = ['TACSTD2', 'ERBB2', 'CLDN18', 'CD274']
LABELS = {
    'TACSTD2': 'TROP2 (TACSTD2)\nz-score within cohort',
    'ERBB2': 'HER2 (ERBB2)\nz-score within cohort',
    'CLDN18': 'Claudin-18 (CLDN18)\nz-score within cohort',
    'CD274': 'PD-L1 (CD274)\nz-score within cohort',
}

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

# Categorical palette for 10 retained cohorts
COHORT_COLORS = [
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
    '#8491B4', '#91D1C2', '#B09C85', '#7E6148', '#6A3D9A', '#B15928',
]


def load_gene_rows(expr_file, genes):
    """Read specific gene rows from expression matrix -> DataFrame (samples x genes)."""
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
    return pd.DataFrame(found, index=samples)


def main():
    apply_pub_style()
    # load per-cohort tumor-only expression, keep raw (un-z-scored) copies for
    # per-cohort correlations, and a z-scored-and-pooled copy for the scatter panels
    per_cohort = {}
    all_dfs = []
    cohort_labels = []
    for i, (cohort, rel) in enumerate(COHORTS.items()):
        df = load_gene_rows(BASE / rel, GENES)
        df = restrict_to_tumor(df, cohort)
        df = df.dropna(subset=['TACSTD2'])
        df['_cohort'] = cohort
        df['_cohort_idx'] = i
        per_cohort[cohort] = df
        all_dfs.append(df)
        cohort_labels.append(cohort)
        print(f'  [{cohort}] n={len(df)}')

    pooled = pd.concat(all_dfs, axis=0)
    print(f'\n  Pooled: {len(pooled)} tumor samples across {len(COHORTS)} cohorts')

    # Within-cohort z-score BEFORE pooling for the SCATTER display only (Spearman
    # rho itself is invariant to this monotonic per-cohort transform, so per-cohort
    # rho values below are identical whether computed pre- or post-z-scoring).
    # Raw pooled values mix platform offsets, pooled raw rho was composition/
    # platform-dependent (TROP2~CLDN18: -0.15 with normals -> +0.47 tumor-only;
    # HER2~CLDN18: +0.55 pooled vs +0.09 per-cohort median).
    pooled[GENES] = pooled.groupby('_cohort')[GENES].transform(
        lambda s: (s - s.mean()) / s.std())

    pairs = [
        ('TACSTD2', 'ERBB2'),
        ('TACSTD2', 'CLDN18'),
        ('TACSTD2', 'CD274'),
    ]
    pair_labels = {
        ('TACSTD2', 'ERBB2'): 'TROP2 vs HER2',
        ('TACSTD2', 'CLDN18'): 'TROP2 vs Claudin-18',
        ('TACSTD2', 'CD274'): 'TROP2 vs PD-L1',
    }

    # -- per-cohort Spearman rho (naive z-scored-pooled rho kept for comparison) --
    stats_rows = []
    meta_rows = []
    pair_meta = {}
    pair_cohort_stats = {}
    for gx, gy in pairs:
        cohort_rows = []
        for cohort in cohort_labels:
            df = per_cohort[cohort].dropna(subset=[gx, gy]) \
                if {gx, gy}.issubset(per_cohort[cohort].columns) else pd.DataFrame()
            if len(df) < 5:
                continue
            r, p = spearmanr(df[gx].values, df[gy].values)
            n = len(df)
            z = np.arctanh(np.clip(r, -0.999999, 0.999999))
            se = np.sqrt(1.0 / (n - 3))
            cohort_rows.append({
                'gene_x': gx, 'gene_y': gy, 'cohort': cohort, 'n': n,
                'spearman_rho': r, 'spearman_p': p, 'fisher_z': z, 'fisher_z_var': se ** 2,
                'ci_lo': np.tanh(z - 1.96 * se), 'ci_hi': np.tanh(z + 1.96 * se),
            })
        cohort_df = pd.DataFrame(cohort_rows)
        pair_cohort_stats[(gx, gy)] = cohort_df
        stats_rows.extend(cohort_rows)

        meta = re_meta_correlation(cohort_df['spearman_rho'].values, cohort_df['n'].values)
        # naive z-scored-pooled rho (old canonical approach) for comparison
        pair_pool = pooled.dropna(subset=[gx, gy])
        x_pool = pair_pool[gx].values
        y_pool = pair_pool[gy].values
        rho_naive, p_naive = spearmanr(x_pool, y_pool)
        pair_meta[(gx, gy)] = meta
        meta_rows.append({
            'gene_x': gx, 'gene_y': gy, 'k_cohorts': len(cohort_df),
            'rho_naive_zscored_pooled': rho_naive, 'p_naive_zscored_pooled': p_naive,
            'rho_RE_pooled': meta['rho_pooled'], 'rho_RE_ci_lo': meta['rho_lo'],
            'rho_RE_ci_hi': meta['rho_hi'], 'tau2': meta['tau2'], 'I2_pct': meta['I2'],
            'meta_p': meta['meta_p'],
            'Q': meta['Q'], 'Q_df': meta['Q_df'], 'Q_p': meta['Q_p'],
            'pred_interval_lo': meta['pi_lo'], 'pred_interval_hi': meta['pi_hi'],
        })
        # sanity check: RE-pooled rho within range of per-cohort rhos; I2 in [0,100]
        lo, hi = cohort_df['spearman_rho'].min(), cohort_df['spearman_rho'].max()
        assert lo - 1e-6 <= meta['rho_pooled'] <= hi + 1e-6, \
            f'{gx}~{gy}: RE-pooled rho {meta["rho_pooled"]:.3f} outside per-cohort range [{lo:.3f}, {hi:.3f}]'
        assert 0 <= meta['I2'] <= 100

    # figure: top row scatters, bottom row forest strips
    fig, axes = plt.subplots(
        2, 3, figsize=(7.2, 4.1),
        gridspec_kw={'height_ratios': [1.0, 1.15], 'hspace': 0.58, 'wspace': 0.38},
    )
    fig.suptitle(
        'TROP2 correlations with treatment biomarkers across gastric cohorts '
        '(random-effects Spearman correlation)',
        fontsize=9.5, fontweight='bold', y=0.99,
    )

    for col, (gx, gy) in enumerate(pairs):
        ax = axes[0, col]
        pair_pool = pooled.dropna(subset=[gx, gy])
        x = pair_pool[gx].values
        y = pair_pool[gy].values
        cidx = pair_pool['_cohort_idx'].values.astype(int)
        meta = pair_meta[(gx, gy)]

        for ci, clbl in enumerate(cohort_labels):
            mask = cidx == ci
            if mask.sum() == 0:
                continue
            ax.scatter(x[mask], y[mask], s=3.5, alpha=0.38,
                       color=COHORT_COLORS[ci % len(COHORT_COLORS)],
                       edgecolors='none', label=clbl, zorder=2)

        z = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, np.polyval(z, xline), color='#333333', lw=1.5,
                ls='--', alpha=0.7, zorder=5)

        ax.set_xlabel(LABELS[gx], fontsize=6.3)
        ax.set_ylabel(LABELS[gy], fontsize=6.3)
        ax.tick_params(labelsize=5.8)
        clean_ax(ax)

        ax.text(0.0, 1.035,
                f'RE pooled $\\rho$ = {meta["rho_pooled"]:+.3f} '
                f'[{meta["rho_lo"]:+.3f}, {meta["rho_hi"]:+.3f}]\n'
                f'Association {fmt_p(meta["meta_p"])}\n'
                f'$I^2$ = {meta["I2"]:.0f}%; Q-test {fmt_p(meta["Q_p"])}',
                transform=ax.transAxes, fontsize=6.1, va='bottom', ha='left',
                fontweight='bold', color='#222222', clip_on=False)

    # forest strip per pair (bottom row)
    for col, (gx, gy) in enumerate(pairs):
        ax = axes[1, col]
        cdf = pair_cohort_stats[(gx, gy)].sort_values('spearman_rho').reset_index(drop=True)
        meta = pair_meta[(gx, gy)]
        yy = np.arange(len(cdf))

        ax.errorbar(cdf['spearman_rho'], yy,
                    xerr=[cdf['spearman_rho'] - cdf['ci_lo'], cdf['ci_hi'] - cdf['spearman_rho']],
                    fmt='o', ms=4, color=PAL['low'], ecolor='#888888',
                    elinewidth=1, capsize=2, zorder=3)

        y_diamond = len(cdf) + 0.8
        diamond = plt.Polygon([
            (meta['rho_lo'], y_diamond),
            (meta['rho_pooled'], y_diamond + 0.35),
            (meta['rho_hi'], y_diamond),
            (meta['rho_pooled'], y_diamond - 0.35),
        ], closed=True, facecolor=PAL['gold'], edgecolor='#222222', lw=0.8, zorder=4)
        ax.add_patch(diamond)

        # prediction interval whisker under the diamond, if available
        if np.isfinite(meta['pi_lo']):
            ax.plot([meta['pi_lo'], meta['pi_hi']], [y_diamond, y_diamond],
                    color='#222222', lw=0.9, ls=':', zorder=2)

        ax.axvline(0, color='#CCCCCC', lw=0.8, zorder=1)
        ax.set_yticks(list(yy) + [y_diamond])
        ax.set_yticklabels(list(cdf['cohort']) + ['RE pooled'], fontsize=5.8)
        for tick, lbl in zip(ax.get_yticklabels(), list(cdf['cohort']) + ['RE pooled']):
            if lbl == 'RE pooled':
                tick.set_fontweight('bold')
        ax.set_xlabel('Spearman r (95% CI)', fontsize=6.2)
        ax.set_xlim(-1, 1)
        ax.tick_params(labelsize=5.5)
        clean_ax(ax)

    fig.text(
        0.5, 0.01,
        'Per-cohort estimates are shown below each scatter; gold diamond = '
        'random-effects pooled correlation; dotted line = 95% prediction interval.',
        ha='center', fontsize=5.6, color='#444444',
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.80, bottom=0.14)
    savefig(fig, OUT / 'F2_ADC_trio_scatter')

    # stats CSV
    per_cohort_df = pd.DataFrame(stats_rows)
    per_cohort_df.insert(0, 'row_type', 'per_cohort')
    meta_df = pd.DataFrame(meta_rows)
    meta_df.insert(0, 'row_type', 'pooled_meta')
    stats_csv = OUT / 'F2_ADC_trio_stats.csv'
    with open(stats_csv, 'w') as f:
        f.write('# per-cohort Spearman rho (Fisher z, DL random-effects meta-analysis)\n')
        per_cohort_df.to_csv(f, index=False)
        f.write('\n# pooled: RE (DerSimonian-Laird) vs within-cohort-z-scored naive pooling\n')
        meta_df.to_csv(f, index=False)
    print(f'\n  Stats saved to {stats_csv}')
    for row in meta_rows:
        print(f"  {row['gene_x']} vs {row['gene_y']}: "
              f"RE rho={row['rho_RE_pooled']:+.3f} "
            f"[{row['rho_RE_ci_lo']:+.3f}, {row['rho_RE_ci_hi']:+.3f}], "
            f"meta_p={row['meta_p']:.3g}, I2={row['I2_pct']:.0f}%, "
            f"Q_p={row['Q_p']:.3f} "
              f"(naive z-pooled rho={row['rho_naive_zscored_pooled']:+.3f})")


if __name__ == '__main__':
    main()

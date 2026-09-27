"""Proliferation and RB pathway scores vs TROP2."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, norm
from statsmodels.stats.meta_analysis import combine_effects
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort_utils import restrict_to_tumor
from style import PAL, apply_pub_style, clean_ax, fmt_p, savefig
from genolib import PROJECT_ROOT


BASE = PROJECT_ROOT
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'pathways'
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = Path(__file__).resolve().parent.parent / 'results' / 'pathways'
RESULTS.mkdir(parents=True, exist_ok=True)
STEM = 'F2_TROP2_proliferation_RB_pathways'
GMT = BASE / 'data' / 'caris' / 'h.all.v2023.2.Hs.symbols.gmt'

COHORTS = {
    'TCGA-STAD': 'data/processed/gastric/TCGA_STAD_for_xcell.txt',
    'GSE66229': 'data/processed/gastric/ACRG_GSE66229_for_xcell.txt',
    'GSE15459': 'data/processed/gastric/ACRG_GSE15459_for_xcell.txt',
    'GSE34942': 'data/processed/gastric/ACRG_GSE34942_for_xcell.txt',
    'GSE35809': 'data/processed/gastric/ACRG_GSE35809_for_xcell.txt',
    'GSE51105': 'data/processed/gastric/ACRG_GSE51105_for_xcell.txt',
    'GSE54129': 'data/processed/gastric/ACRG_GSE54129_for_xcell.txt',
    'GSE57303': 'data/processed/gastric/ACRG_GSE57303_for_xcell.txt',
    'GSE84437': 'data/processed/gastric/ACRG_GSE84437_for_xcell.txt',
}

HALLMARKS = {
    'HALLMARK_E2F_TARGETS': 'E2F targets',
    'HALLMARK_G2M_CHECKPOINT': 'G2/M checkpoint',
    'HALLMARK_MYC_TARGETS_V1': 'MYC targets',
    'HALLMARK_MITOTIC_SPINDLE': 'Mitotic spindle',
    'HALLMARK_DNA_REPAIR': 'DNA repair',
    'HALLMARK_PI3K_AKT_MTOR_SIGNALING': 'PI3K-AKT-mTOR',
    'HALLMARK_P53_PATHWAY': 'p53 pathway',
    'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION': 'EMT',
}

# Malorni et al., Oncotarget 2016, Table 1 (87-gene RB loss-of-function signature).
RBSIG = [
    'CDC45', 'RAD51', 'PIF1', 'TPX2', 'CCNB2', 'NDC80', 'CDCA5', 'CENPO',
    'ASPM', 'CCNA2', 'CDT1', 'DEPDC1', 'MYBL2', 'DEPDC1B', 'NEIL3', 'UBE2C',
    'NCAPH', 'SGOL1', 'CDCA3', 'BUB1B', 'KIFC1', 'MCM10', 'CENPE', 'MCM7',
    'FOXM1', 'KIF4B', 'MKI67', 'BIRC5', 'PKMYT1', 'TTK', 'CENPA', 'DLGAP5',
    'KIF14', 'AURKB', 'MELK', 'OIP5', 'KIF2C', 'KIF20A', 'NUSAP1', 'CDCA8',
    'PTTG1', 'ASF1B', 'TICRR', 'TRIP13', 'FAM64A', 'ORC1', 'GTSE1', 'MND1',
    'PLK1', 'PTTG3P', 'STIL', 'EXO1', 'SPC25', 'RRM2', 'RAD54L', 'CDC25A',
    'PRC1', 'CEP55', 'CENPM', 'ANLN', 'CDC20', 'PTTG2', 'FANCI', 'CENPI',
    'FAM83D', 'SKA1', 'AURKA', 'CENPN', 'SKA3', 'TROAP', 'ORC6', 'MCM4',
    'POLQ', 'CHEK1', 'ARHGAP11A', 'KIF4A', 'CDKN3', 'KIF15', 'CLSPN',
    'KIF11', 'AUNIP', 'BUB1', 'MTFR2', 'CENPW', 'BLM', 'KIF23', 'NUF2',
]


def read_gmt():
    selected = {}
    with open(GMT) as handle:
        for line in handle:
            fields = line.rstrip('\n').split('\t')
            if fields[0] in HALLMARKS:
                selected[HALLMARKS[fields[0]]] = fields[2:]
    selected['RB loss-of-function (RBsig)'] = RBSIG
    return selected


def read_expression(path):
    data = pd.read_csv(path, sep='\t', index_col=0)
    data = data.apply(pd.to_numeric, errors='coerce')
    return data.loc[~data.index.duplicated()]


def hedges_g(high, low):
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    n1, n0 = len(high), len(low)
    pooled_var = ((n1 - 1) * np.var(high, ddof=1) +
                  (n0 - 1) * np.var(low, ddof=1)) / (n1 + n0 - 2)
    if pooled_var <= 0:
        return np.nan, np.nan
    d = (np.mean(high) - np.mean(low)) / np.sqrt(pooled_var)
    correction = 1 - 3 / (4 * (n1 + n0) - 9)
    g = correction * d
    variance = ((n1 + n0) / (n1 * n0) +
                g ** 2 / (2 * (n1 + n0 - 2)))
    return float(g), float(variance)


def main():
    apply_pub_style()
    gene_sets = read_gmt()
    rows = []
    score_rows = []
    for cohort, rel in COHORTS.items():
        expression = read_expression(BASE / rel).T
        expression = restrict_to_tumor(expression, cohort)
        expression = expression.dropna(axis=1, how='all')
        trop2 = expression['TACSTD2'].astype(float)
        high_mask = trop2 >= trop2.median()
        # Standardize each gene across samples before forming pathway means.
        means = expression.mean(axis=0)
        sds = expression.std(axis=0, ddof=0).replace(0, np.nan)
        z = (expression - means) / sds
        for pathway, genes in gene_sets.items():
            present = [gene for gene in genes if gene in z.columns and gene != 'TACSTD2']
            if len(present) < 10:
                continue
            score = z[present].mean(axis=1)
            high = score[high_mask].dropna()
            low = score[~high_mask].dropna()
            g, variance = hedges_g(high, low)
            p = float(ttest_ind(high, low, equal_var=False).pvalue)
            rows.append({
                'cohort': cohort, 'pathway': pathway,
                'genes_present': len(present), 'n_high': len(high), 'n_low': len(low),
                'hedges_g': g, 'variance': variance, 'welch_p': p,
            })
            for sample, value in score.items():
                score_rows.append({
                    'cohort': cohort, 'sample_id': sample, 'pathway': pathway,
                    'score': value, 'TROP2_group': 'High' if high_mask.loc[sample] else 'Low',
                })
        print(f'[{cohort}] n={len(expression)}')

    stats = pd.DataFrame(rows)
    summaries = []
    for pathway, group in stats.groupby('pathway', sort=False):
        group = group.dropna(subset=['hedges_g', 'variance'])
        meta = combine_effects(
            group['hedges_g'].to_numpy(),
            group['variance'].to_numpy(),
            method_re='iterated', use_t=False,
        )
        effect = float(meta.mean_effect_re)
        se = float(np.sqrt(meta.var_eff_w_re))
        summaries.append({
            'cohort': 'RANDOM EFFECTS', 'pathway': pathway,
            'genes_present': int(group['genes_present'].min()),
            'n_high': int(group['n_high'].sum()),
            'n_low': int(group['n_low'].sum()),
            'hedges_g': effect, 'variance': se ** 2,
            'ci_lo': effect - 1.96 * se, 'ci_hi': effect + 1.96 * se,
            'welch_p': float(2 * (1 - norm.cdf(abs(effect / se)))),
            'i2': float(max(0, meta.i2) * 100),
        })
    summary = pd.DataFrame(summaries)
    summary['fdr'] = multipletests(summary['welch_p'], method='fdr_bh')[1]
    stats = pd.concat([stats, summary], ignore_index=True, sort=False)
    stats.to_csv(OUT / f'{STEM}_stats.csv', index=False)
    pd.DataFrame(score_rows).to_csv(RESULTS / f'{STEM}_scores.csv', index=False)

    pathway_order = (
        summary.sort_values('hedges_g', ascending=False)['pathway'].tolist()
    )
    cohort_order = list(COHORTS)
    matrix = (
        stats[stats['cohort'].isin(cohort_order)]
        .pivot(index='pathway', columns='cohort', values='hedges_g')
        .reindex(index=pathway_order, columns=cohort_order)
    )
    pmat = (
        stats[stats['cohort'].isin(cohort_order)]
        .pivot(index='pathway', columns='cohort', values='welch_p')
        .reindex(index=pathway_order, columns=cohort_order)
    )
    summary = summary.set_index('pathway').reindex(pathway_order)

    fig, (ax, forest) = plt.subplots(
        1, 2, figsize=(13.8, 6.7),
        gridspec_kw={'width_ratios': [3.2, 1.25], 'wspace': 0.35},
    )
    image = ax.imshow(matrix, cmap='RdBu_r', vmin=-0.8, vmax=0.8, aspect='auto')
    ax.set_xticks(range(len(cohort_order)))
    ax.set_xticklabels(cohort_order, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(pathway_order)))
    ax.set_yticklabels(pathway_order, fontsize=10.5)
    for i, pathway in enumerate(pathway_order):
        for j, cohort in enumerate(cohort_order):
            value = matrix.loc[pathway, cohort]
            if pd.isna(value):
                continue
            marker = '•' if pmat.loc[pathway, cohort] < 0.05 else ''
            ax.text(j, i, f'{value:+.2f}{marker}', ha='center', va='center',
                    fontsize=8.6,
                    color='white' if abs(value) > 0.48 else 'black')
    ax.set_title('Cohort-specific effect\n(TROP2-high minus low)',
                 fontsize=13, fontweight='bold', pad=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.027, pad=0.018)
    cbar.set_label('Hedges g', fontsize=11.5)
    cbar.ax.tick_params(labelsize=9.5)

    y = np.arange(len(pathway_order))
    for yy, (pathway, row) in enumerate(summary.iterrows()):
        color = (PAL['tumor'] if row['ci_lo'] > 0 else
                 PAL['low'] if row['ci_hi'] < 0 else '#888888')
        forest.plot([row['ci_lo'], row['ci_hi']], [yy, yy], color=color, lw=2.2)
        forest.scatter(row['hedges_g'], yy, s=64, color=color,
                       edgecolor='black', lw=0.5, zorder=3)
        forest.text(
            1.02, yy,
            f"{row['hedges_g']:+.2f} "
            f"({row['ci_lo']:+.2f}, {row['ci_hi']:+.2f})\n"
            f"{fmt_p(row['fdr']).replace('p = ', 'FDR = ')}; "
            f"I²={row['i2']:.0f}%",
            transform=forest.get_yaxis_transform(), ha='left', va='center',
            fontsize=9.2,
        )
    forest.axvline(0, color='black', ls='--', lw=0.9)
    forest.set_yticks(y)
    forest.set_yticklabels([])
    forest.set_ylim(len(pathway_order) - 0.5, -0.5)
    forest.set_xlim(-0.75, 0.75)
    forest.set_xlabel('Random-effects Hedges g', fontsize=11.5)
    forest.tick_params(axis='x', labelsize=10)
    forest.set_title('Cross-cohort summary', fontsize=13,
                     fontweight='bold', pad=8)
    clean_ax(forest)
    fig.suptitle('Proliferation and RB-pathway activity by TROP2 expression',
                 fontsize=14, fontweight='bold', y=0.99)
    fig.text(
        0.01, 0.01,
        '• nominal within-cohort p<0.05. Positive values indicate higher pathway '
        'activity in TROP2-high tumors. High RBsig indicates functional RB loss.',
        fontsize=9.2, color='#444444',
    )
    fig.subplots_adjust(left=0.21, right=0.81, top=0.88, bottom=0.19)
    savefig(fig, OUT / STEM, pad_inches=0.04)
    print(summary[['hedges_g', 'ci_lo', 'ci_hi', 'welch_p', 'fdr', 'i2']].to_string())


if __name__ == '__main__':
    main()

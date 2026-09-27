"""Cell-level inferred CNV vs TROP2 (Kumar)."""

from pathlib import Path
import re
import sys

import infercnvpy as cnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import PAL, apply_pub_style, clean_ax, fmt_p, savefig, axis_label
from genolib import PROJECT_ROOT

BASE = PROJECT_ROOT
GTF = BASE / 'data' / 'annotations' / 'gencode.v19.annotation.gtf.gz'
H5AD = BASE / 'data/processed/scrna/gastric/kumar_tumor.h5ad'
RESULTS = BASE / 'trop2' / 'results' / 'scrna_cnv_kumar_cell'
PLOTS = BASE / 'trop2' / 'plots' / 'scrna'
RESULTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)
STEM = 'S1B_Kumar_celllevel_CNV_vs_TROP2'

REFERENCE_LINEAGES = ['T/NK', 'B/Plasma', 'Myeloid', 'Endothelial', 'Fibroblast']
MAX_REFERENCE_PER_LINEAGE = 300
MAX_EPITHELIAL = 5000
SEED = 20260730
N_DECILES = 10


def dense_vector(matrix):
    return matrix.toarray().ravel() if sparse.issparse(matrix) else np.asarray(matrix).ravel()


def safe_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))


def balanced_indices(obs, rng):
    epithelial = np.flatnonzero(obs['lineage'].to_numpy() == 'Epithelial')
    if len(epithelial) > MAX_EPITHELIAL:
        epithelial = rng.choice(epithelial, MAX_EPITHELIAL, replace=False)
    reference = []
    for lineage in REFERENCE_LINEAGES:
        indices = np.flatnonzero(obs['lineage'].to_numpy() == lineage)
        if len(indices) > MAX_REFERENCE_PER_LINEAGE:
            indices = rng.choice(indices, MAX_REFERENCE_PER_LINEAGE, replace=False)
        reference.extend(indices.tolist())
    return np.asarray(epithelial), np.asarray(reference)


def process_sample(sample, source, rng):
    cache = RESULTS / f'Kumar_{safe_name(sample)}.npz'
    if cache.exists():
        loaded = np.load(cache, allow_pickle=True)
        return {
            'sample': sample,
            'cnv_score': loaded['cnv_score'],
            'trop2': loaded['trop2'],
        }

    sample_mask = source.obs['sample'].astype(str).to_numpy() == str(sample)
    sample_data = source[sample_mask].copy()
    epi_idx, ref_idx = balanced_indices(sample_data.obs, rng)
    if len(epi_idx) < 50 or len(ref_idx) < 100:
        print(f'  [{sample}] skipped: epi={len(epi_idx)}, ref={len(ref_idx)}')
        return None
    selected = np.concatenate([epi_idx, ref_idx])
    work = sample_data[selected].copy()
    work.obs['cnv_group'] = ['epithelial'] * len(epi_idx) + ['reference'] * len(ref_idx)

    cnv.tl.infercnv(
        work,
        reference_key='cnv_group',
        reference_cat='reference',
        window_size=100,
        step=10,
        dynamic_threshold=1.5,
        exclude_chromosomes=('chrX', 'chrY'),
        chunksize=1000,
        n_jobs=1,
    )
    representation = work.obsm['X_cnv']
    absolute = abs(representation)
    scores = np.asarray(absolute.mean(axis=1)).ravel()
    epi_scores = scores[:len(epi_idx)]
    trop2 = dense_vector(work[:len(epi_idx), 'TACSTD2'].X)

    np.savez_compressed(cache, cnv_score=epi_scores, trop2=trop2)
    print(f'  [{sample}] epi={len(epi_idx)}, ref={len(ref_idx)}')
    return {'sample': sample, 'cnv_score': epi_scores, 'trop2': trop2}


def main():
    apply_pub_style()
    if not GTF.exists():
        raise FileNotFoundError(f'Missing GENCODE gene annotation: {GTF}')
    rng = np.random.default_rng(SEED)
    limit = None
    if '--limit-samples' in sys.argv:
        limit = int(sys.argv[sys.argv.index('--limit-samples') + 1])

    source = sc.read_h5ad(H5AD)
    cnv.io.genomic_position_from_gtf(GTF, source, gtf_gene_id='gene_name', inplace=True)
    samples = source.obs['sample'].astype(str).unique().tolist()
    if limit is not None:
        samples = samples[:limit]
    print(f'[Kumar] {len(samples)} tumors')

    per_cell_rows = []
    for sample in samples:
        result = process_sample(sample, source, rng)
        if result is None:
            continue
        n = len(result['cnv_score'])
        per_cell_rows.append(pd.DataFrame({
            'sample': [result['sample']] * n,
            'cnv_score': result['cnv_score'],
            'trop2': result['trop2'],
        }))
    del source

    cells = pd.concat(per_cell_rows, ignore_index=True)
    cells.to_csv(RESULTS / f'{STEM}_per_cell.csv', index=False)
    n_tumors = cells['sample'].nunique()
    n_cells = len(cells)
    print(f'Kumar epithelial cells: n={n_cells} across {n_tumors} tumors')

    # Raw pooled-cell association is descriptive only because cells are nested
    # within tumors and between-tumor shifts can induce an ecological trend.
    cell_rho, cell_p = spearmanr(cells['cnv_score'], cells['trop2'])

    # 2) Patient-level sensitivity: per-tumor mean epithelial CNV vs mean TROP2.
    tumor_means = cells.groupby('sample').agg(
        cnv_mean=('cnv_score', 'mean'), trop2_mean=('trop2', 'mean'), n=('trop2', 'size'),
    ).reset_index()
    patient_rho, patient_p = spearmanr(tumor_means['cnv_mean'], tumor_means['trop2_mean'])

    # Primary within-tumor analysis: remove each tumor's mean from both axes,
    # then use a tumor-clustered sandwich SE. This isolates the cell-level
    # association from between-patient differences.
    cells['cnv_within'] = cells['cnv_score'] - cells.groupby('sample')['cnv_score'].transform('mean')
    cells['trop2_within'] = cells['trop2'] - cells.groupby('sample')['trop2'].transform('mean')
    within_rho, within_p_naive = spearmanr(cells['cnv_within'], cells['trop2_within'])
    fixed_effect = smf.ols('trop2_within ~ cnv_within', data=cells).fit(
        cov_type='cluster', cov_kwds={'groups': cells['sample']},
    )
    fe_coef = fixed_effect.params['cnv_within']
    fe_p = fixed_effect.pvalues['cnv_within']

    stats = pd.DataFrame([
        {'analysis': 'Cell-level Spearman (all epithelial cells, unclustered)',
         'n': n_cells, 'n_tumors': n_tumors, 'estimate': cell_rho, 'p': cell_p},
        {'analysis': 'Patient-level sensitivity Spearman (per-tumor epithelial means)',
         'n': n_tumors, 'n_tumors': n_tumors, 'estimate': patient_rho, 'p': patient_p},
        {'analysis': 'Within-tumor centered Spearman (descriptive)',
         'n': n_cells, 'n_tumors': n_tumors, 'estimate': within_rho, 'p': within_p_naive},
        {'analysis': 'Within-tumor centered OLS slope, patient-clustered SE',
         'n': n_cells, 'n_tumors': n_tumors, 'estimate': fe_coef, 'p': fe_p},
    ])
    stats.to_csv(PLOTS / f'{STEM}_stats.csv', index=False)
    print(stats.to_string(index=False))

    # Rank CNV within each tumor before forming deciles, then show centered
    # expression. This keeps the second panel on the same within-tumor estimand.
    cells['cnv_percentile_within'] = cells.groupby('sample')['cnv_score'].rank(pct=True, method='average')
    cells['decile'] = np.minimum((cells['cnv_percentile_within'] * N_DECILES).astype(int), N_DECILES - 1)
    decile_groups = [
        cells.loc[cells['decile'] == d, 'trop2_within'].to_numpy()
        for d in sorted(cells['decile'].dropna().unique())
    ]
    decile_means = np.array([np.mean(g) for g in decile_groups])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), gridspec_kw={'width_ratios': [1.25, 1]})

    hb = axes[0].hexbin(
        cells['cnv_within'], cells['trop2_within'], gridsize=42, cmap='viridis',
        mincnt=1, bins='log',
    )
    xx = np.linspace(cells['cnv_within'].min(), cells['cnv_within'].max(), 100)
    axes[0].plot(xx, fixed_effect.params['Intercept'] + fe_coef * xx, color=PAL['tumor'], lw=2)
    cb = fig.colorbar(hb, ax=axes[0])
    cb.set_label(axis_label('Cell density', transform='log10'), fontsize=10)
    axes[0].text(
        0.03, 0.97,
        f'Within-tumor r={within_rho:+.2f}\n'
        f'Clustered slope {fmt_p(fe_p)}\n'
        f'n={n_cells:,} cells; {n_tumors} tumors',
        transform=axes[0].transAxes, ha='left', va='top', fontsize=7.2,
        bbox={'facecolor': 'white', 'edgecolor': '#CCCCCC', 'boxstyle': 'round,pad=0.3', 'alpha': 0.92},
    )
    axes[0].set_xlabel('Inferred-CNV burden\n(centered within tumor)', fontsize=8.5)
    axes[0].set_ylabel('TROP2 expression\n(log-normalized; centered within tumor)', fontsize=8.5)
    axes[0].set_title('Cell-level association', fontsize=9.5, fontweight='bold')
    axes[0].tick_params(labelsize=7)
    clean_ax(axes[0])

    bp = axes[1].boxplot(
        decile_groups, positions=np.arange(1, len(decile_groups) + 1),
        widths=0.6, showfliers=False, patch_artist=True,
    )
    for patch in bp['boxes']:
        patch.set_facecolor(PAL['low'])
        patch.set_alpha(0.55)
    x_pos = np.arange(1, len(decile_groups) + 1)
    mean_line, = axes[1].plot(x_pos, decile_means, color=PAL['tumor'], lw=2, marker='o', ms=4,
                               label='Mean centered TROP2')
    axes[1].set_xlabel('Within-tumor inferred-CNV decile', fontsize=8.5)
    axes[1].set_ylabel('TROP2 expression\n(log-normalized; centered within tumor)', fontsize=8.5)
    axes[1].set_title('Expression across CNV deciles', fontsize=9.5, fontweight='bold')
    axes[1].tick_params(labelsize=7)
    axes[1].legend([mean_line], [mean_line.get_label()],
                   loc='upper left', fontsize=7, frameon=False)
    clean_ax(axes[1])
    fig.suptitle(
        'Inferred CNV and TROP2 in Kumar cohort epithelial cells',
        fontsize=10.5, fontweight='bold', y=0.995,
    )
    fig.text(
        0.5, 0.005,
        f'Patient-level sensitivity: r={patient_rho:+.2f}, {fmt_p(patient_p)} ({n_tumors} tumors). '
        'Primary slope uses within-tumor centering and patient-clustered SEs.',
        ha='center', fontsize=6.6, color='#444444',
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.23, wspace=0.35)
    savefig(fig, PLOTS / STEM, pad_inches=0.05)


if __name__ == '__main__':
    main()

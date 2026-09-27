"""Single-cell inferred CNV vs TROP2."""

from pathlib import Path
import re
import sys

import anndata as ad
import infercnvpy as cnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import PAL, apply_pub_style, clean_ax, fmt_p, savefig
from genolib import PROJECT_ROOT


BASE = PROJECT_ROOT
GTF = BASE / 'data' / 'annotations' / 'gencode.v19.annotation.gtf.gz'
RESULTS = BASE / 'trop2' / 'results' / 'scrna_cnv'
PLOTS = BASE / 'trop2' / 'plots' / 'scrna'
RESULTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)
STEM = 'F2_scRNA_TROP2_inferred_CNV'

DATASETS = {
    'Kumar': BASE / 'data/processed/scrna/gastric/kumar_tumor.h5ad',
    'Sathe': BASE / 'data/processed/scrna/gastric/sathe_tumor.h5ad',
}
REFERENCE_LINEAGES = ['T/NK', 'B/Plasma', 'Myeloid', 'Endothelial', 'Fibroblast']
MAX_REFERENCE_PER_LINEAGE = 300
MAX_EPITHELIAL = 5000
SEED = 20260730


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


def process_sample(dataset, source, sample, rng):
    cache = RESULTS / f'{dataset}_{safe_name(sample)}.npz'
    if cache.exists():
        loaded = np.load(cache, allow_pickle=True)
        return {
            'dataset': dataset,
            'sample': sample,
            'n_epithelial': int(loaded['n_epithelial']),
            'n_reference': int(loaded['n_reference']),
            'trop2_mean': float(loaded['trop2_mean']),
            'trop2_detected_pct': float(loaded['trop2_detected_pct']),
            'cnv_median': float(loaded['cnv_median']),
            'cnv_mean': float(loaded['cnv_mean']),
            'malignant_fraction': float(loaded['malignant_fraction']),
            'cell_rho': float(loaded['cell_rho']),
            'profile': loaded['profile'],
            'chr_pos_keys': loaded['chr_pos_keys'].tolist(),
            'chr_pos_values': loaded['chr_pos_values'].astype(int).tolist(),
        }

    sample_mask = source.obs['sample'].astype(str).to_numpy() == str(sample)
    sample_data = source[sample_mask].copy()
    epi_idx, ref_idx = balanced_indices(sample_data.obs, rng)
    if len(epi_idx) < 50 or len(ref_idx) < 100:
        print(f'  [{dataset}/{sample}] skipped: epi={len(epi_idx)}, ref={len(ref_idx)}')
        return None
    selected = np.concatenate([epi_idx, ref_idx])
    work = sample_data[selected].copy()
    work.obs['cnv_group'] = (
        ['epithelial'] * len(epi_idx) + ['reference'] * len(ref_idx)
    )

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
    ref_scores = scores[len(epi_idx):]
    ref_threshold = np.quantile(ref_scores, 0.95)

    trop2 = dense_vector(work[:len(epi_idx), 'TACSTD2'].X)
    rho = spearmanr(trop2, epi_scores).statistic
    profile = np.asarray(representation[:len(epi_idx)].mean(axis=0)).ravel()
    chr_pos = work.uns['cnv'].get('chr_pos', {})
    result = {
        'dataset': dataset,
        'sample': sample,
        'n_epithelial': len(epi_idx),
        'n_reference': len(ref_idx),
        'trop2_mean': float(np.mean(trop2)),
        'trop2_detected_pct': float(100 * np.mean(trop2 > 0)),
        'cnv_median': float(np.median(epi_scores)),
        'cnv_mean': float(np.mean(epi_scores)),
        'malignant_fraction': float(np.mean(epi_scores > ref_threshold)),
        'cell_rho': float(rho),
        'profile': profile,
        'chr_pos_keys': list(chr_pos.keys()),
        'chr_pos_values': list(chr_pos.values()),
    }
    np.savez_compressed(
        cache,
        **{key: value for key, value in result.items()
           if key not in {'dataset', 'sample'}},
    )
    print(
        f"  [{dataset}/{sample}] epi={len(epi_idx)}, ref={len(ref_idx)}, "
        f"TROP2={result['trop2_mean']:.3f}, CNV={result['cnv_median']:.4f}"
    )
    return result


def main():
    apply_pub_style()
    if not GTF.exists():
        raise FileNotFoundError(f'Missing GENCODE gene annotation: {GTF}')
    rng = np.random.default_rng(SEED)
    limit = None
    if '--limit-samples' in sys.argv:
        limit = int(sys.argv[sys.argv.index('--limit-samples') + 1])

    all_results = []
    for dataset, path in DATASETS.items():
        source = sc.read_h5ad(path)
        # Attach GRCh37 positions once, then every within-sample object inherits them.
        cnv.io.genomic_position_from_gtf(
            GTF, source, gtf_gene_id='gene_name', inplace=True,
        )
        samples = source.obs['sample'].astype(str).unique().tolist()
        if limit is not None:
            samples = samples[:limit]
        print(f'[{dataset}] {len(samples)} tumors')
        for sample in samples:
            result = process_sample(dataset, source, sample, rng)
            if result is not None:
                all_results.append(result)
        del source

    if len(all_results) < 4:
        raise RuntimeError('Too few tumors completed for an aggregate analysis.')
    tumors = pd.DataFrame([
        {key: value for key, value in row.items()
         if key not in {'profile', 'chr_pos_keys', 'chr_pos_values'}}
        for row in all_results
    ])
    tumors.to_csv(RESULTS / f'{STEM}_per_tumor.csv', index=False)

    # Dataset-standardized pooled association prevents dataset offsets driving rho.
    tumors['trop2_z'] = tumors.groupby('dataset')['trop2_mean'].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )
    tumors['cnv_z'] = tumors.groupby('dataset')['cnv_median'].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )
    stat_rows = []
    for dataset, group in [
        *list(tumors.groupby('dataset')),
        ('Pooled (within-dataset z)', tumors),
    ]:
        x = group['trop2_z'] if dataset.startswith('Pooled') else group['trop2_mean']
        y = group['cnv_z'] if dataset.startswith('Pooled') else group['cnv_median']
        rho, p = spearmanr(x, y)
        stat_rows.append({
            'dataset': dataset, 'n_tumors': len(group),
            'spearman_rho': rho, 'p': p,
        })
    for dataset, group in tumors.groupby('dataset'):
        values = group['cell_rho'].dropna()
        p = wilcoxon(values).pvalue if len(values) >= 6 else np.nan
        stat_rows.append({
            'dataset': f'{dataset} within-tumor cell rho',
            'n_tumors': len(values), 'median_rho': values.median(), 'p': p,
        })
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(PLOTS / f'{STEM}_aggregate_stats.csv', index=False)

    # Separate the heatmap block and tumor-level scatter with a generous gutter.
    # The first draft placed a vertical colorbar, TROP2 strip, and scatter y-axis
    # in the same narrow center region.
    fig = plt.figure(figsize=(13.2, 6.75))
    outer = fig.add_gridspec(
        1, 2, width_ratios=[1.72, 1.0], wspace=0.30,
        left=0.065, right=0.975, top=0.84, bottom=0.18,
    )
    left = outer[0, 0].subgridspec(
        3, 2, width_ratios=[24, 1.25], height_ratios=[1, 1, 0.075],
        wspace=0.045, hspace=0.28,
    )
    heat_images = []
    for row_index, dataset in enumerate(('Kumar', 'Sathe')):
        dataset_results = [row for row in all_results if row['dataset'] == dataset]
        profile_length = min(len(row['profile']) for row in dataset_results)
        dataset_results = [
            row for row in dataset_results if len(row['profile']) == profile_length
        ]
        profile_meta = [
            tumors[
                (tumors['dataset'] == row['dataset']) &
                (tumors['sample'] == row['sample'])
            ].iloc[0]
            for row in dataset_results
        ]
        order = np.argsort([record['trop2_z'] for record in profile_meta])
        profile_matrix = np.vstack(
            [row['profile'] for row in dataset_results]
        )[order]
        sorted_meta = pd.DataFrame(profile_meta).iloc[order].reset_index(drop=True)
        first = dataset_results[0]
        chr_pos = dict(zip(first['chr_pos_keys'], first['chr_pos_values']))
        boundaries = sorted(
            [(chrom, pos) for chrom, pos in chr_pos.items() if pos < profile_length],
            key=lambda item: item[1],
        )

        heat = fig.add_subplot(left[row_index, 0])
        image = heat.imshow(
            profile_matrix, aspect='auto', cmap='RdBu_r',
            vmin=-0.18, vmax=0.18, interpolation='nearest',
        )
        heat_images.append((heat, image))
        for _, position in boundaries:
            heat.axvline(position - 0.5, color='white', lw=0.4, alpha=0.75)
        starts = [pos for _, pos in boundaries]
        ends = starts[1:] + [profile_length]
        centers = [(a + b) / 2 for a, b in zip(starts, ends)]
        heat.set_xticks(centers)
        chrom_labels = [chrom.replace('chr', '') for chrom, _ in boundaries]
        heat.set_xticklabels(
            [label if ((int(label) % 2 == 1 and int(label) < 20) or label == '22') else ''
             for label in chrom_labels],
            fontsize=8.5,
        )
        heat.set_yticks([])
        heat.set_ylabel(f'{dataset}\n(n={len(sorted_meta)})', fontsize=11.2)
        if row_index == 0:
            heat.set_title('Epithelial inferred-CNV profiles',
                           fontsize=13.3, fontweight='bold', pad=7)

        bar = fig.add_subplot(left[row_index, 1])
        bar.imshow(sorted_meta[['trop2_z']].to_numpy(), aspect='auto',
                   cmap='viridis', vmin=-2, vmax=2)
        bar.set_xticks([])
        bar.set_yticks([])
        if row_index == 0:
            bar.set_title('TROP2\nz-score', fontsize=9.4,
                          fontweight='bold', pad=6)
        for spine in bar.spines.values():
            spine.set_visible(False)
    cbar_ax = fig.add_subplot(left[2, 0])
    cbar = fig.colorbar(heat_images[0][1], cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Smoothed expression deviation', fontsize=10.6, labelpad=3)
    cbar.ax.tick_params(labelsize=9.2, pad=2)
    blank_ax = fig.add_subplot(left[2, 1])
    blank_ax.axis('off')

    scatter = fig.add_subplot(outer[0, 1])
    colors = {'Kumar': PAL['low'], 'Sathe': PAL['tumor']}
    for dataset, group in tumors.groupby('dataset'):
        scatter.scatter(
            group['trop2_mean'], group['cnv_median'], s=58,
            color=colors[dataset], edgecolor='white', lw=0.5,
            alpha=0.9, label=f'{dataset} (n={len(group)})',
        )
        if len(group) >= 5:
            coef = np.polyfit(group['trop2_mean'], group['cnv_median'], 1)
            xx = np.linspace(group['trop2_mean'].min(), group['trop2_mean'].max(), 100)
            yy = np.polyval(coef, xx)
            scatter.plot(xx, yy, color=colors[dataset], lw=1.7)
            scatter.annotate(
                dataset, xy=(xx[-1], yy[-1]), xytext=(5, 0),
                textcoords='offset points', ha='left', va='center',
                fontsize=10.2, fontweight='bold', color=colors[dataset],
                clip_on=False,
            )
    annotation = []
    for row in stat_rows[:3]:
        annotation.append(
            f"{row['dataset']}: rho={row['spearman_rho']:+.2f}, {fmt_p(row['p'])}"
        )
    scatter.text(
        0.035, 0.965, '\n'.join(annotation), transform=scatter.transAxes,
        ha='left', va='top', fontsize=9.8, linespacing=1.35,
        bbox={'facecolor': 'white', 'edgecolor': '#CCCCCC',
              'boxstyle': 'round,pad=0.35', 'alpha': 0.92},
    )
    scatter.set_xlabel('Epithelial TROP2 expression\n(mean log-normalized counts)',
                       fontsize=11.5)
    scatter.set_ylabel('Median epithelial inferred-CNV burden', fontsize=11.5)
    scatter.tick_params(labelsize=10)
    scatter.set_title('Tumor-level association', fontsize=13,
                      fontweight='bold', pad=8)
    clean_ax(scatter)
    fig.suptitle('Single-cell inferred CNV and TROP2 expression',
                 fontsize=15, fontweight='bold', y=0.965)
    savefig(fig, PLOTS / STEM, pad_inches=0.04)
    print(stats.to_string(index=False))


if __name__ == '__main__':
    main()

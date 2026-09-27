"""Expression of ADC targets across cohorts."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (PAL, clean_ax, savefig, apply_pub_style, axis_label,
                   LABEL_FS, TITLE_FS, TICK_FS, CELL_FS,
                   HEATMAP_YTICK_FS, HEATMAP_XTICK_FS, LEGEND_FS)
from cohort_utils import tumor_ids

from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
OUT.mkdir(parents=True, exist_ok=True)
LARGE_TEXT = '--large-text' in sys.argv
OUTPUT_STEM = ('F1_ADC_target_landscape_large_text'
               if LARGE_TEXT else 'F1_ADC_target_landscape')

# 15-target ADC landscape (original panel). Keys = primary matrix symbols;
# ALIASES tried if primary missing. Display labels are journal-facing.
ADC_TARGETS = {
    'MUC1':    'MUC1',
    'EPCAM':   'EPCAM / TROP1 (TACSTD1)',
    'CLDN18':  'Claudin-18 (CLDN18)',
    'CEACAM5': 'CEA (CEACAM5)',
    'ERBB3':   'HER3 (ERBB3)',
    'MET':     'c-MET (MET)',
    'MSLN':    'Mesothelin (MSLN)',
    'TACSTD2': 'TROP2 (TACSTD2)',
    'ERBB2':   'HER2 (ERBB2)',
    'EGFR':    'EGFR',
    'CD276':   'B7-H3 (CD276)',
    'FOLR1':   'FRα (FOLR1)',
    'FGFR2':   'FGFR2',
    'MUC16':   'MUC16',
    'PVRL4':   'Nectin-4 (PVRL4)',
}

# Alternate symbols present on some platforms (Illumina Ref-8 etc.)
ALIASES = {
    'EPCAM': ['EPCAM', 'TACSTD1'],
    'PVRL4': ['PVRL4', 'NECTIN4'],
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


def gene_median_percentiles(expr_file, targets, cohort):
    """Percentile rank (0-100) of each target's median expression among all genes.

    targets: dict primary_key -> list of candidate symbols to try.
    Adjacent-normal samples are removed before gene medians are calculated.
    Returns ({primary_key: percentile}, tumor_sample_count).
    """
    want = set()
    for cands in targets.values():
        want.update(cands)

    medians = {}
    hit_medians = {}  # symbol -> median
    with open(expr_file) as f:
        samples = f.readline().rstrip('\n').split('\t')[1:]
        keep_ids = tumor_ids(cohort)
        keep = (np.ones(len(samples), dtype=bool) if keep_ids is None else
                np.array([sample in keep_ids for sample in samples], dtype=bool))
        if not keep.any():
            raise ValueError(f'No tumor samples retained for {cohort}')
        for line in f:
            tab = line.index('\t')
            gene = line[:tab]
            vals = line[tab + 1:].rstrip('\n').split('\t')
            arr = np.array([float(v) if v.strip() else np.nan for v in vals],
                           dtype=float)[keep]
            if np.all(np.isnan(arr)):
                continue
            m = np.nanmedian(arr)
            medians[gene] = m
            if gene in want:
                hit_medians[gene] = m

    all_med = np.array(list(medians.values()))
    n = len(all_med)
    out = {}
    for key, cands in targets.items():
        m = None
        for c in cands:
            if c in hit_medians:
                m = hit_medians[c]
                break
        if m is None:
            continue
        out[key] = 100.0 * np.sum(all_med <= m) / n
    return out, int(keep.sum())


def main():
    apply_pub_style()
    # Increase type relative to a fixed canvas. The earlier large-text draft
    # enlarged both canvas and type, leaving their apparent ratio unchanged.
    extra = 0.5 if LARGE_TEXT else 0.0
    fs = {
        'cell': CELL_FS + 2.0 + extra,
        'xtick': HEATMAP_XTICK_FS + 1.8 + extra,
        'ytick': HEATMAP_YTICK_FS + 1.6 + extra,
        'label': LABEL_FS + 1.6 + extra,
        'title': TITLE_FS + 2.0 + extra,
        'tick': TICK_FS + 1.4 + extra,
    }
    # primary -> candidate symbols
    lookup = {k: ALIASES.get(k, [k]) for k in ADC_TARGETS}
    targets = list(ADC_TARGETS.keys())
    mat = pd.DataFrame(index=targets, columns=list(COHORTS.keys()), dtype=float)

    cohort_sizes = {}

    for cohort, rel in COHORTS.items():
        pct, cohort_sizes[cohort] = gene_median_percentiles(
            BASE / rel, lookup, cohort)
        for g in targets:
            mat.loc[g, cohort] = pct.get(g, np.nan)
        present = [g for g in targets if g in pct]
        print(f'[{cohort}] n={cohort_sizes[cohort]}; {len(present)}/{len(targets)} targets found; '
              f'TROP2 percentile={pct.get("TACSTD2", float("nan")):.0f}; '
              f'Nectin-4={pct.get("PVRL4", float("nan")):.0f}')

    weights = np.array([cohort_sizes[c] for c in mat.columns], dtype=float)
    weighted_means = []
    for gene in mat.index:
        vals = mat.loc[gene].values.astype(float)
        mask = ~np.isnan(vals)
        if mask.sum() == 0:
            weighted_means.append(0.0)
        else:
            weighted_means.append(np.average(vals[mask], weights=weights[mask]))
    mat['__weighted_mean'] = weighted_means
    mat = mat.sort_values('__weighted_mean', ascending=False)
    order = mat.index.tolist()
    mean_col = mat.pop('__weighted_mean')

    # Compact heatmap with larger cells and axis text
    n_c, n_t = len(mat.columns), len(order)
    fig, ax = plt.subplots(figsize=(13.8, 8.55))
    data = mat.values.astype(float)
    im = ax.imshow(data, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=100)

    xlabels = [f'{c}\n(n={cohort_sizes[c]})' for c in mat.columns]
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=fs['xtick'])
    ax.set_yticks(range(n_t))
    ax.set_yticklabels([ADC_TARGETS[g] for g in order], fontsize=fs['ytick'])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        fontsize=fs['cell'],
                        color='black' if 25 < v < 80 else 'white')

    trop2_i = order.index('TACSTD2')
    ax.add_patch(plt.Rectangle((-0.5, trop2_i - 0.5), n_c, 1, fill=False,
                               edgecolor=PAL['accent_low'], lw=2.4, clip_on=False))

    ax.set_title('ADC-target expression landscape',
                 fontsize=fs['title'], fontweight='bold', pad=12)
    ax.set_xlabel('Cohort (tumor samples)', fontsize=fs['label'], labelpad=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(axis_label('Expression', unit='transcriptome percentile'),
                   fontsize=fs['label'])
    cbar.ax.tick_params(labelsize=fs['tick'])

    fig.subplots_adjust(left=0.245, right=0.925, top=0.92, bottom=0.20)
    savefig(fig, OUT / OUTPUT_STEM)

    mat['weighted_mean_percentile'] = mean_col
    csv = OUT / f'{OUTPUT_STEM}_percentiles.csv'
    mat.to_csv(csv)
    print(f'Retained tumor samples: {sum(cohort_sizes.values())} across '
          f'{len(cohort_sizes)} cohorts')
    print(f'\nTROP2 weighted mean transcriptome percentile: '
          f'{mean_col.loc["TACSTD2"]:.0f} (rank {trop2_i+1} of {len(targets)})')
    print(f'Nectin-4 weighted mean: {mean_col.loc["PVRL4"]:.0f}')


if __name__ == '__main__':
    main()

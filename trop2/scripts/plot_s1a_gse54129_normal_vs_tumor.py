"""TROP2 tumor vs normal in GSE54129."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (clean_ax, savefig, fmt_p, apply_pub_style, axis_label,
                   LABEL_FS, TITLE_FS, TICK_FS, ANNOT_FS)

from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
OUT.mkdir(parents=True, exist_ok=True)

GENE = 'TACSTD2'
GENE_LABEL = 'TROP2'

CFG = {
    'expr': 'data/processed/gastric/ACRG_GSE54129_for_xcell.txt',
    'clinical': 'data/clinical/gastric/processed/GSE54129_clinical.csv',
    'id_col': 'gsm_id',
    'normal_filter': ('tissue', 'normal'),
    'tumor_pattern': ('tissue', 'tumor'),
}

TUMOR_COLOR = '#C9A227'    # gold
NORMAL_COLOR = '#B0B0B0'   # soft mid-gray


def load_gene_row(expr_file, gene):
    with open(expr_file) as f:
        header = f.readline().rstrip('\n').split('\t')
        samples = header[1:]
        for line in f:
            tab = line.index('\t')
            if line[:tab] == gene:
                vals = line[tab + 1:].rstrip('\n').split('\t')
                row = [float(v) if v.strip() else np.nan for v in vals]
                return pd.Series(row, index=samples)
    raise ValueError(f"{gene} not found in {expr_file}")


def split_tumor_normal(cfg):
    clin = pd.read_csv(BASE / cfg['clinical'])
    clin[cfg['id_col']] = clin[cfg['id_col']].astype(str).str.strip()

    col, val = cfg['normal_filter']
    n_ids = set(clin.loc[clin[col].astype(str).str.strip().str.lower() == val.lower(),
                         cfg['id_col']])

    col, pat = cfg['tumor_pattern']
    t_ids = set(clin.loc[clin[col].astype(str).str.lower().str.contains(pat, na=False),
                         cfg['id_col']])

    return t_ids, n_ids


def draw_panel(ax, normal_vals, tumor_vals, ylabel):
    n_n, n_t = len(normal_vals), len(tumor_vals)
    _, p_val = mannwhitneyu(tumor_vals, normal_vals, alternative='two-sided')

    positions = [0, 1]
    parts = ax.violinplot(
        [normal_vals, tumor_vals], positions=positions,
        showmeans=False, showmedians=False, showextrema=False, widths=0.7,
    )
    for pc, fill in zip(parts['bodies'], [NORMAL_COLOR, TUMOR_COLOR]):
        pc.set_facecolor(fill)
        pc.set_alpha(0.85)
        pc.set_edgecolor('none')
        pc.set_linewidth(0)

    ax.boxplot(
        [normal_vals, tumor_vals], positions=positions, widths=0.16,
        patch_artist=True, showfliers=True, zorder=4,
        medianprops={'color': 'black', 'linewidth': 1.3},
        boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 0.9},
        whiskerprops={'color': 'black', 'linewidth': 0.9},
        capprops={'color': 'black', 'linewidth': 0.9},
        flierprops={
            'marker': 'o', 'markersize': 2.5, 'markerfacecolor': 'black',
            'markeredgecolor': 'none', 'alpha': 0.55, 'linestyle': 'none',
        },
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f'n={n_n}\nnormal', f'n={n_t}\ntumor'],
        fontsize=TICK_FS + 1,
    )
    ax.tick_params(axis='y', labelsize=TICK_FS)

    ymin = min(normal_vals.min(), tumor_vals.min())
    ymax = max(normal_vals.max(), tumor_vals.max())
    span = max(ymax - ymin, 0.5)
    ax.set_ylim(ymin - 0.06 * span, ymax + 0.28 * span)

    ax.yaxis.grid(True, linestyle=':', linewidth=0.7, color='#BBBBBB', zorder=0)
    ax.set_axisbelow(True)

    bracket_y = ymax + 0.08 * span
    tick_h = 0.02 * span
    ax.plot([0, 0, 1, 1],
            [bracket_y, bracket_y + tick_h, bracket_y + tick_h, bracket_y],
            color='black', lw=0.9, clip_on=False)
    ax.text(0.5, bracket_y + 1.5 * tick_h, fmt_p(p_val),
            ha='center', va='bottom', fontsize=ANNOT_FS + 0.5, clip_on=False)

    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    clean_ax(ax)

    delta = float(np.median(tumor_vals) - np.median(normal_vals))
    return p_val, delta


def main():
    apply_pub_style()

    row = load_gene_row(BASE / CFG['expr'], GENE)
    t_ids, n_ids = split_tumor_normal(CFG)
    tumor_vals = row.loc[list(t_ids)].dropna().values.astype(float)
    normal_vals = row.loc[list(n_ids)].dropna().values.astype(float)
    n_t, n_n = len(tumor_vals), len(normal_vals)
    print(f'GSE54129: tumor n={n_t}, normal n={n_n}')

    fig, ax = plt.subplots(figsize=(2.8, 3.4))
    ylab = axis_label(f'{GENE_LABEL} expression', transform='log2')
    p_val, delta = draw_panel(ax, normal_vals, tumor_vals, ylab)

    ax.set_title('GSE54129', fontsize=TITLE_FS - 1,
                 fontweight='bold', pad=6)

    plt.tight_layout()
    savefig(fig, OUT / 'S1A_TROP2_normal_vs_tumor_GSE54129')

    stats = pd.DataFrame([{
        'cohort': 'GSE54129',
        'normal_n': n_n,
        'tumor_n': n_t,
        'median_delta_log2': delta,
        'mw_p': p_val,
    }])
    stats.to_csv(OUT / 'S1A_TROP2_normal_vs_tumor_GSE54129_stats.csv', index=False)
    print(f'delta={delta:+.2f} log2, {fmt_p(p_val)}')
    print('Saved: S1A_TROP2_normal_vs_tumor_GSE54129.{pdf,png}')


if __name__ == '__main__':
    main()

"""Clinicopathological subgroup forest plot."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import PAL, apply_pub_style, clean_ax, savefig
from survival_revision_utils import (
    FULL_SUBGROUP_FACTORS, load_all, subgroup_stats,
)
from plot_f2_clinicopath_subgroups import fmt_p_col


OUT = Path(__file__).resolve().parent.parent / 'plots' / 'survival'
STEM = 'F2_TROP2_subgroup_forest_truncated'

# Factors shown here, in display order, with the shorter Figure-2 labels.
SHOWN = [
    ('Age', 'Age'),
    ('Sex', 'Gender'),
    ('Stage', 'Stage'),
    ('Lauren type', 'Lauren'),
    ('TCGA subtype', 'TCGA subtype'),
    ('EMP subtype', 'Epithelial phenotype'),
]


def main():
    apply_pub_style()
    full = subgroup_stats(load_all(), FULL_SUBGROUP_FACTORS)
    keep = [f for f, _ in SHOWN]
    stats = full[full['factor'].isin(keep)].copy()
    stats['factor'] = stats['factor'].astype(
        __import__('pandas').CategoricalDtype(keep, ordered=True))
    stats = stats.sort_values('factor', kind='stable').reset_index(drop=True)
    stats['display_factor'] = stats['factor'].map(dict(SHOWN))
    stats.to_csv(OUT / f'{STEM}_stats.csv', index=False)

    y_positions, header_y = [], {}
    cursor = 0.0
    for factor, _ in SHOWN:
        header_y[factor] = cursor
        cursor += 1
        for _ in range((stats['factor'] == factor).sum()):
            y_positions.append(cursor)
            cursor += 1
        cursor += 0.35
    stats['y'] = y_positions
    ymax = cursor
    stats['plot_y'] = ymax - stats['y']
    header_y = {k: ymax - v for k, v in header_y.items()}

    fig = plt.figure(figsize=(6.3, 6.15))
    grid = fig.add_gridspec(
        1, 6, width_ratios=[1.75, 0.80, 1.65, 1.32, 0.56, 0.56], wspace=0.04,
        left=0.018, right=0.986, top=0.912, bottom=0.145,
    )
    label_ax = fig.add_subplot(grid[0, 0])
    n_ax = fig.add_subplot(grid[0, 1])
    ax = fig.add_subplot(grid[0, 2])
    est_ax = fig.add_subplot(grid[0, 3])
    p_ax = fig.add_subplot(grid[0, 4])
    fdr_ax = fig.add_subplot(grid[0, 5])
    all_axes = (label_ax, n_ax, ax, est_ax, p_ax, fdr_ax)

    for i, row in enumerate(stats.itertuples()):
        if i % 2:
            for band_ax in all_axes:
                band_ax.axhspan(row.plot_y - 0.5, row.plot_y + 0.5,
                                color='#F0F0F0', zorder=0, lw=0)

    for row in stats.itertuples():
        display_level = '≥65' if row.level == '>=65' else row.level
        label_ax.text(0.06, row.plot_y, display_level,
                      ha='left', va='center', fontsize=7.5)
        n_ax.text(0.02, row.plot_y, f'{row.n_high} / {row.n_low}',
                  ha='left', va='center', fontsize=7.3)
        if np.isnan(row.HR):
            est_ax.text(0.02, row.plot_y, 'n/e', ha='left', va='center',
                        fontsize=7.3, color='#777777', style='italic')
        else:
            color = (PAL['accent_high'] if row.lo > 1 else
                     PAL['accent_low'] if row.hi < 1 else '#888888')
            ax.plot([max(row.lo, 0.21), min(row.hi, 4.4)],
                    [row.plot_y, row.plot_y], color=color, lw=1.3, zorder=2)
            ax.scatter(row.HR, row.plot_y, s=20, color=color,
                       edgecolor='black', lw=0.35, zorder=3)
            est_ax.text(0.02, row.plot_y,
                        f'{row.HR:.2f} ({row.lo:.2f}-{row.hi:.2f})',
                        ha='left', va='center', fontsize=7.3)
        p_ax.text(0.5, row.plot_y, fmt_p_col(row.p),
                  ha='center', va='center', fontsize=7.3)
        fdr_ax.text(0.5, row.plot_y, fmt_p_col(row.fdr),
                    ha='center', va='center', fontsize=7.3)

    for factor, display in SHOWN:
        block = stats[stats['factor'] == factor]
        y = header_y[factor]
        label_ax.text(0.01, y, display, ha='left', va='center',
                      fontsize=7.8, fontweight='bold')
        p_ax.text(0.5, y, fmt_p_col(block['interaction_p'].iloc[0]),
                  ha='center', va='center', fontsize=7.3, fontweight='bold')
        fdr_ax.text(0.5, y, fmt_p_col(block['interaction_fdr'].iloc[0]),
                    ha='center', va='center', fontsize=7.3, fontweight='bold')

    ax.axvline(1, color='black', ls='--', lw=0.7, zorder=1)
    ax.set_xscale('log')
    ax.set_xlim(0.2, 4.5)
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(['0.25', '0.5', '1', '2', '4'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([])
    ax.set_ylim(-0.7, ymax + 0.9)
    ax.set_xlabel('Hazard ratio for OS\n(TROP2-high vs low)', fontsize=7.8)
    ax.tick_params(axis='x', labelsize=7.2)
    clean_ax(ax, keep_left=False)

    for side_ax in (label_ax, n_ax, est_ax, p_ax, fdr_ax):
        side_ax.set_xlim(0, 1)
        side_ax.set_ylim(ax.get_ylim())
        side_ax.set_xticks([])
        side_ax.set_yticks([])
        for spine in side_ax.spines.values():
            spine.set_visible(False)

    for head_ax, text, loc in (
            (label_ax, 'Subgroup', 'left'),
            (n_ax, 'N (hi/lo)', 'left'),
            (est_ax, 'HR (95% CI)', 'left'),
            (p_ax, 'p', 'center'),
            (fdr_ax, 'FDR', 'center')):
        head_ax.set_title(text, fontsize=7.8, fontweight='bold', loc=loc, pad=4)

    fig.suptitle('TROP2 OS by subgroup', fontsize=10.2, fontweight='bold',
                 y=0.985)
    fig.text(
        0.018, 0.012,
        'N = TROP2-high / TROP2-low. Bold rows are factor-level interaction tests. '
        'Benjamini-Hochberg FDR,\nadjusted across the full nine-factor subgroup '
        'family (Fig. S2a), not across the six factors shown here.',
        fontsize=6.0, linespacing=1.35, color='#666666', ha='left',
    )
    savefig(fig, OUT / STEM, pad_inches=0.03)
    print(stats[['display_factor', 'level', 'n_high', 'n_low', 'HR', 'lo', 'hi',
                 'p', 'fdr', 'interaction_p',
                 'interaction_fdr']].to_string(index=False))


if __name__ == '__main__':
    main()

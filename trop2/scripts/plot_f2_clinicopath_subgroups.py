"""Subgroup forest plot for TROP2 and OS."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import PAL, apply_pub_style, clean_ax, savefig
from survival_revision_utils import (
    FULL_SUBGROUP_FACTORS, interaction_p, load_all, subgroup_stats,
)


OUT = Path(__file__).resolve().parent.parent / 'plots' / 'survival'
STEM = 'F2_TROP2_clinicopathological_subgroup_forest'

# Kept importable: the truncated panel historically pulled interaction_p from
# here before it moved into survival_revision_utils.
__all__ = ['interaction_p', 'main', 'fmt_p_col']


def fmt_p_col(value):
    """Compact p/FDR column text (blank when the test was not estimable)."""
    if value is None or not np.isfinite(value):
        return ' - '
    if value < 0.001:
        return f'{value:.1e}'
    return f'{value:.3f}'


def main():
    apply_pub_style()
    stats = subgroup_stats(load_all(), FULL_SUBGROUP_FACTORS)
    stats.to_csv(OUT / f'{STEM}_stats.csv', index=False)

    # One row per stratum plus a header row per factor, so the header text has
    # its own line and never collides with a stratum label.
    factors = [f[0] for f in FULL_SUBGROUP_FACTORS]
    y_positions, header_y = [], {}
    cursor = 0.0
    for factor in factors:
        header_y[factor] = cursor
        cursor += 1
        for _ in range((stats['factor'] == factor).sum()):
            y_positions.append(cursor)
            cursor += 1
        cursor += 0.45
    stats['y'] = y_positions
    ymax = cursor
    stats['plot_y'] = ymax - stats['y']
    header_y = {k: ymax - v for k, v in header_y.items()}

    fig = plt.figure(figsize=(7.6, 9.4))
    grid = fig.add_gridspec(
        1, 6, width_ratios=[2.30, 0.85, 2.20, 1.42, 0.60, 0.60], wspace=0.04,
        left=0.015, right=0.988, top=0.925, bottom=0.085,
    )
    label_ax = fig.add_subplot(grid[0, 0])
    n_ax = fig.add_subplot(grid[0, 1])
    ax = fig.add_subplot(grid[0, 2])
    est_ax = fig.add_subplot(grid[0, 3])
    p_ax = fig.add_subplot(grid[0, 4])
    fdr_ax = fig.add_subplot(grid[0, 5])

    # Zebra banding across the whole row, as in the reference figure.
    for i, row in enumerate(stats.itertuples()):
        if i % 2 == 0:
            continue
        for band_ax in (label_ax, n_ax, ax, est_ax, p_ax, fdr_ax):
            band_ax.axhspan(row.plot_y - 0.5, row.plot_y + 0.5,
                            color='#F0F0F0', zorder=0, lw=0)

    for row in stats.itertuples():
        display_level = '≥65' if row.level == '>=65' else row.level
        label_ax.text(0.045, row.plot_y, display_level,
                      ha='left', va='center', fontsize=8.4)
        n_ax.text(0.02, row.plot_y, f'{row.n_high} / {row.n_low}',
                  ha='left', va='center', fontsize=8.2)
        if np.isnan(row.HR):
            est_ax.text(0.02, row.plot_y, 'Not estimable',
                        ha='left', va='center', fontsize=8.2,
                        color='#777777', style='italic')
        else:
            color = (PAL['accent_high'] if row.lo > 1 else
                     PAL['accent_low'] if row.hi < 1 else '#888888')
            ax.plot([max(row.lo, 0.21), min(row.hi, 4.4)],
                    [row.plot_y, row.plot_y], color=color, lw=1.8, zorder=2)
            ax.scatter(row.HR, row.plot_y, s=42, color=color,
                       edgecolor='black', lw=0.45, zorder=3)
            est_ax.text(
                0.02, row.plot_y,
                f'{row.HR:.2f} ({row.lo:.2f}-{row.hi:.2f})',
                ha='left', va='center', fontsize=8.2,
            )
        p_ax.text(0.5, row.plot_y, fmt_p_col(row.p),
                  ha='center', va='center', fontsize=8.2)
        fdr_ax.text(0.5, row.plot_y, fmt_p_col(row.fdr),
                    ha='center', va='center', fontsize=8.2)

    # Factor header rows: name at the left, interaction p / FDR in the p and
    # FDR columns (the reference figure's group-level p).
    for factor in factors:
        block = stats[stats['factor'] == factor]
        y = header_y[factor]
        label_ax.text(0.01, y, factor, ha='left', va='center',
                      fontsize=8.6, fontweight='bold')
        p_ax.text(0.5, y, fmt_p_col(block['interaction_p'].iloc[0]),
                  ha='center', va='center', fontsize=8.2, fontweight='bold')
        fdr_ax.text(0.5, y, fmt_p_col(block['interaction_fdr'].iloc[0]),
                    ha='center', va='center', fontsize=8.2, fontweight='bold')

    ax.axvline(1, color='black', ls='--', lw=0.9, zorder=1)
    ax.set_xscale('log')
    ax.set_xlim(0.2, 4.5)
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(['0.25', '0.5', '1', '2', '4'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([])
    ax.set_ylim(-0.75, ymax + 0.9)
    ax.set_xlabel('Hazard ratio for OS (TROP2-high vs low)', fontsize=9.0)
    ax.tick_params(axis='x', labelsize=8.2)
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
        head_ax.set_title(text, fontsize=8.8, fontweight='bold', loc=loc, pad=7)

    fig.suptitle('TROP2 overall survival across clinicopathological subgroups',
                 fontsize=11.5, fontweight='bold', y=0.981)
    fig.text(
        0.015, 0.014,
        'N = TROP2-high / TROP2-low. Bold rows are factor-level TROP2 x subgroup interaction tests; '
        'stratum rows are within-subgroup\ncohort-stratified Cox models. All adjustment is '
        'Benjamini-Hochberg: stratum rows across the estimable stratum tests, '
        'factor rows across the nine interaction tests.',
        fontsize=7.2, linespacing=1.35, color='#666666', ha='left',
    )
    savefig(fig, OUT / STEM, pad_inches=0.04)
    print(stats[['factor', 'level', 'n_high', 'n_low', 'events', 'HR', 'lo',
                 'hi', 'p', 'fdr', 'interaction_p',
                 'interaction_fdr']].to_string(index=False))


if __name__ == '__main__':
    main()

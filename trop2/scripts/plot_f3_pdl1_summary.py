"""TROP2 vs PD-L1 summary."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from style import PAL, clean_ax, savefig

OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
STATS = OUT / 'F3_PDL1_immune_adjusted_stats.csv'

# Presentation version: PD-L1 + B7-H4 only. PDCD1 was dropped, a single
# transcript is too noisy to serve as a clean T-cell-content control (it
# attenuates in TCGA, -0.22 -> -0.14, but is inconsistent in small cohorts).
# It remains in the stats CSV for the record. VTCN1 is the decisive control:
# it proves the adjustment itself manufactures no positive signal.
MARKERS = [
    ('CD274', 'PD-L1 (CD274)\nboth compartments'),
    ('VTCN1', 'B7-H4 (VTCN1)\ntumor-intrinsic control'),
]


def main():
    stats = pd.read_csv(STATS).dropna(subset=['raw_r'])
    stats = stats[~stats['platform_flag']]  # canonical: exclude GSE84437

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rng = np.random.default_rng(3)

    for yi, (sym, disp) in enumerate(MARKERS):
        y = len(MARKERS) - 1 - yi  # CD274 on top
        s = stats[stats['symbol'] == sym]
        raw, adj = s['raw_r'].values, s['adj_immune_r'].values

        for vals, col, filled, lab, dy in [
                (raw, PAL['low'], False, 'raw (whole-tumor bulk)', -0.24),
                (adj, PAL['tumor'], True, 'adjusted for immune content', 0.24)]:
            med, lo, hi = np.median(vals), np.percentile(vals, 25), np.percentile(vals, 75)
            jit = rng.uniform(-0.07, 0.07, len(vals))
            ax.scatter(vals, y + jit + (0.06 if col == PAL['tumor'] else -0.06),
                       s=26, facecolor=col if filled else 'none', edgecolor=col,
                       lw=1.1, alpha=0.55, zorder=2)
            ax.plot([lo, hi], [y, y], color=col, lw=5, alpha=0.35, zorder=3,
                    solid_capstyle='round')
            ax.scatter(med, y, s=150, facecolor=col if filled else 'white',
                       edgecolor=col, lw=2, zorder=4, label=lab if yi == 0 else None)
            ax.text(med, y + dy, f'{med:+.2f}', ha='center', fontsize=9,
                    fontweight='bold', color=col)

        n_pos = int((adj > 0).sum())
        ax.text(0.565, y, f'{n_pos}/{len(s)} cohorts positive', va='center',
                fontsize=8.5, color='#444444', style='italic')

    ax.axvline(0, color='black', lw=0.9, zorder=1)
    ax.set_yticks(range(len(MARKERS)))
    ax.set_yticklabels([d for _, d in MARKERS][::-1], fontsize=10)
    ax.set_xlabel('Spearman ρ with TROP2 (median ± IQR across cohorts)', fontsize=10)
    ax.set_xlim(-0.28, 0.58)
    ax.set_title('Removing immune content reveals a uniformly positive,\n'
                 'weak tumor-intrinsic PD-L1~TROP2 association\n'
                 '(B7-H4 control: the adjustment itself creates no signal)',
                 fontsize=10.5, pad=8)
    ax.set_ylim(-0.55, len(MARKERS) - 0.15)
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.0, 1.0),
              frameon=False)
    clean_ax(ax)

    plt.tight_layout()
    savefig(fig, OUT / 'F3_PDL1_immune_adjusted_summary')
    s = stats[stats['symbol'] == 'CD274']
    print(f"CD274: median raw {s['raw_r'].median():+.3f} -> adj "
          f"{s['adj_immune_r'].median():+.3f}; positive {(s['adj_immune_r'] > 0).sum()}"
          f"/{len(s)}; sig {(s['adj_immune_p'] < 0.05).sum()} (excl. platform-flagged)")


if __name__ == '__main__':
    main()

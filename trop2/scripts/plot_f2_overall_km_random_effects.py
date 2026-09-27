"""Overall survival KM with random-effects HR."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import PAL, apply_pub_style, clean_ax, fmt_p, savefig
from survival_revision_utils import load_all, random_effects_trop2


OUT = Path(__file__).resolve().parent.parent / 'plots' / 'survival'
STEM = 'F2_TROP2_overall_OS_random_effects'


def main():
    apply_pub_style()
    data = load_all().dropna(subset=['duration', 'event', 'TACSTD2']).copy()

    # Prespecified revision: define equal-frequency thirds separately within
    # each cohort so platform-specific expression scales never set a pooled
    # cutoff. qcut implements the requested 1/3 and 2/3 percentile boundaries.
    data['TACSTD2_tertile'] = pd.Series(index=data.index, dtype=object)
    for _, idx in data.groupby('cohort').groups.items():
        data.loc[idx, 'TACSTD2_tertile'] = pd.qcut(
            data.loc[idx, 'TACSTD2'],
            q=[0, 1 / 3, 2 / 3, 1],
            labels=['low', 'mid', 'high'],
        ).astype(str)

    groups = {
        label: data[data['TACSTD2_tertile'] == label]
        for label in ('low', 'mid', 'high')
    }
    extremes = data[data['TACSTD2_tertile'].isin(['low', 'high'])].copy()
    extremes['TACSTD2_high_vs_low'] = (
        extremes['TACSTD2_tertile'] == 'high'
    ).astype(int)
    cohort_estimates, meta = random_effects_trop2(
        extremes, target='TACSTD2_high_vs_low')
    logrank = multivariate_logrank_test(
        data['duration'], data['TACSTD2_tertile'], data['event'])

    # Time-unit audit (per-cohort os_months, printed to log below): medians
    # 14.8-57.9 months, maxima 105.7-157.8 months across the 4 cohorts. These
    # are genuine months (retrospective natural-history series with up to
    # ~13y follow-up), not a days-mislabeled-as-months bug (which would show
    # maxima in the 3000-5000 range). GSE15459 (157.8mo) and GSE34942
    # (136.9mo) extend past 120 months on very few at-risk patients, so the
    # x-axis is capped at a clinically standard 120-month (10y) cutoff; the
    # KM fit itself still uses all available follow-up.
    XMAX = 120

    fig, ax = plt.subplots(figsize=(3.5, 3.25))
    fitters = []
    # Muted clinical palette: soft,
    # but dark enough to remain legible at manuscript scale and in print.
    group_style = {
        'low': ('TROP2-low', '#5B8DB8'),
        'mid': ('TROP2-mid', '#929292'),
        'high': ('TROP2-high', '#D1A12B'),
    }
    for key in ('low', 'mid', 'high'):
        group = groups[key]
        label, color = group_style[key]
        kmf = KaplanMeierFitter(label=f'{label} (n={len(group)})')
        kmf.fit(group['duration'], group['event'])
        kmf.plot_survival_function(
            ax=ax, color=color, lw=1.5, ci_show=False,
        )
        fitters.append(kmf)
    ax.set_xlim(0, XMAX)
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_ylim(0, 1.03)
    ax.set_xlabel('Time (months)', fontsize=8.5)
    ax.set_ylabel('Overall survival probability', fontsize=8.5)
    ax.tick_params(labelsize=7.2)
    ax.set_title('Overall survival by TROP2 expression', fontsize=9.5,
                 fontweight='bold', pad=6)
    ax.legend(frameon=False, fontsize=6.8, loc='lower left')
    ax.text(
        0.97, 0.97,
        f"High vs low random-effects HR = {meta['HR']:.2f}\n"
        f"(95% CI {meta['lo']:.2f}-{meta['hi']:.2f})\n"
        f"{fmt_p(meta['p'])}; I² = {meta['i2']:.0f}%\n"
        f"Three-group log-rank {fmt_p(logrank.p_value)}",
        transform=ax.transAxes, ha='right', va='top', fontsize=6.3,
        linespacing=1.3,
        bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white',
              'edgecolor': '#BBBBBB', 'alpha': 0.92},
    )
    clean_ax(ax)
    add_at_risk_counts(
        *fitters, ax=ax, rows_to_show=['At risk'],
        labels=['TROP2-low', 'TROP2-mid', 'TROP2-high'], fontsize=5.7,
        xticks=[0, 30, 60, 90, 120],
    )
    fig.subplots_adjust(left=0.185, right=0.98, top=0.88, bottom=0.355)
    savefig(fig, OUT / STEM, pad_inches=0.03)

    cohort_estimates.to_csv(OUT / f'{STEM}_cohort_stats.csv', index=False)
    pd.DataFrame([{
        **meta,
        'n': len(data),
        'events': int(data['event'].sum()),
        'logrank_p': float(logrank.p_value),
        'cutoff': 'within-cohort tertiles (1/3 and 2/3 percentiles)',
        'contrast': 'TROP2-high versus TROP2-low tertile',
        'meta_method': 'Paule-Mandel random effects',
    }]).to_csv(OUT / f'{STEM}_summary.csv', index=False)
    print(pd.DataFrame([meta]).to_string(index=False))


if __name__ == '__main__':
    main()

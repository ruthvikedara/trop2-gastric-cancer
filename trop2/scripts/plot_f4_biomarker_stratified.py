"""TROP2 and OS within PD-L1, Claudin-18 and HER2 strata."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import chi2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (
    PAL, clean_ax, savefig, fmt_p, apply_pub_style,
    TICK_FS, LABEL_FS, TITLE_FS, LEGEND_FS, ANNOT_FS,
)
from genolib import PROJECT_ROOT

BASE = PROJECT_ROOT
CLIN = BASE / 'data' / 'clinical' / 'gastric' / 'processed'
PROC = BASE / 'data' / 'processed' / 'gastric'
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'survival'
OUT.mkdir(parents=True, exist_ok=True)

GENE = 'TACSTD2'
STRATIFIERS = {'CD274': 'PD-L1 (CD274)', 'CLDN18': 'Claudin-18', 'ERBB2': 'HER2 (ERBB2)'}
MAIN_STRATIFIERS = {'CD274': STRATIFIERS['CD274'], 'CLDN18': STRATIFIERS['CLDN18']}
SUPPLEMENT_STRATIFIERS = {'ERBB2': STRATIFIERS['ERBB2']}

DATASETS = {
    'TCGA-STAD': dict(clinical='TCGA_clinical.csv', expr='TCGA_STAD_for_xcell.txt',
                      id_col='patient_id', filt=None),
    'GSE66229':  dict(clinical='GSE66229_clinical.csv', expr='ACRG_GSE66229_for_xcell.txt',
                      id_col='gsm_id', filt=('tissue_type', 'Tumor')),
    'GSE15459':  dict(clinical='GSE15459_clinical.csv', expr='ACRG_GSE15459_for_xcell.txt',
                      id_col='gsm_id', filt=None),
    'GSE34942':  dict(clinical='GSE34942_clinical.csv', expr='ACRG_GSE34942_for_xcell.txt',
                      id_col='gsm_id', filt=None),
}


def load_gene_rows(f, genes):
    found = {}
    with open(f) as fh:
        samples = fh.readline().rstrip('\n').split('\t')[1:]
        for line in fh:
            tab = line.index('\t')
            g = line[:tab]
            if g in genes:
                vals = line[tab + 1:].rstrip('\n').split('\t')
                found[g] = pd.Series([float(v) if v.strip() else np.nan for v in vals],
                                     index=samples)
    return found


def load_cohort(name, cfg):
    clin = pd.read_csv(CLIN / cfg['clinical'])
    if cfg['filt']:
        clin = clin[clin[cfg['filt'][0]] == cfg['filt'][1]]
    clin = clin.set_index(cfg['id_col'])
    expr = load_gene_rows(PROC / cfg['expr'], {GENE} | set(STRATIFIERS))
    if name.startswith('TCGA'):
        clin.index = clin.index.astype(str).str[:12]
    for g in expr:
        if name.startswith('TCGA'):
            expr[g].index = expr[g].index.astype(str).str[:12]
    t = expr[GENE].dropna()
    common = clin.index.intersection(t.index)
    clin = clin.loc[common]
    df = pd.DataFrame({
        'duration': pd.to_numeric(clin['os_months'], errors='coerce'),
        'event': pd.to_numeric(clin['os_status'], errors='coerce'),
    })
    df['trop2'] = t.loc[common]
    df['trop2_high'] = (df['trop2'] > df['trop2'].median()).astype(int)
    for s in STRATIFIERS:
        if s in expr:
            v = expr[s].reindex(common)
            df[s + '_high'] = (v > v.median()).astype(float).where(v.notna())
        else:
            df[s + '_high'] = np.nan
    df['cohort'] = name
    # Match the positive-duration eligibility rule used by the overall KM and
    # subgroup forests. Zero-month records otherwise make legend n disagree
    # with the at-risk count at time zero.
    df = df.dropna(subset=['duration', 'event'])
    return df[(df['duration'] > 0) & df['event'].isin([0, 1])]


def cox_hr(sdf):
    sdf = sdf.dropna(subset=['trop2_high', 'duration', 'event'])
    if sdf['trop2_high'].nunique() < 2 or len(sdf) < 30:
        return np.nan, np.nan, np.nan, np.nan
    cph = CoxPHFitter()
    try:
        cph.fit(sdf[['duration', 'event', 'trop2_high', 'cohort']],
                duration_col='duration', event_col='event', strata=['cohort'])
        s = cph.summary.loc['trop2_high']
        return (float(np.exp(s['coef'])), float(np.exp(s['coef lower 95%'])),
                float(np.exp(s['coef upper 95%'])), float(s['p']))
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def interaction_p(sdf, gene):
    sdf = sdf.dropna(subset=['trop2_high', gene + '_high', 'duration', 'event']).copy()
    if sdf[gene + '_high'].nunique() < 2:
        return np.nan
    sdf['bio'] = sdf[gene + '_high'].astype(int)
    sdf['ixn'] = sdf['trop2_high'] * sdf['bio']
    base = ['duration', 'event', 'trop2_high', 'bio', 'cohort']
    c0 = CoxPHFitter(); c0.fit(sdf[base], duration_col='duration',
                               event_col='event', strata=['cohort'])
    c1 = CoxPHFitter(); c1.fit(sdf[base + ['ixn']], duration_col='duration',
                               event_col='event', strata=['cohort'])
    return float(chi2.sf(2 * (c1.log_likelihood_ - c0.log_likelihood_), 1))


# Time-unit audit (matches plot_f2_overall_km_random_effects.py): per-cohort
# os_months medians ~15-58, maxima ~106-158 across the 4 cohorts -- genuine
# months (natural-history series with up to ~13y follow-up), not a
# days-mislabeled-as-months bug. Display x-axis capped at a clinically
# standard 120-month (10y) cutoff; KM fits themselves use full follow-up.
XMAX = 120


RISK_TIMES = [0, 30, 60, 90, 120]


def draw_risk_table(ax, risk_counts):
    """Draw a compact, fixed-layout risk table without adding overlapping axes."""
    order = [(1, 1), (1, 0), (0, 1), (0, 0)]
    labels = {(1, 1): 'B+/T+', (1, 0): 'B+/T-',
              (0, 1): 'B-/T+', (0, 0): 'B-/T-'}
    ax.set_xlim(0, XMAX)
    ax.set_ylim(-0.55, 4.25)
    ax.axis('off')
    ax.text(-0.12, 4.0, 'At risk', transform=ax.get_yaxis_transform(),
            ha='right', va='center', fontsize=6.4, fontweight='bold')
    for y, key in zip((3.05, 2.05, 1.05, 0.05), order):
        ax.text(-0.12, y, labels[key], transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=6.3)
        for x, value in zip(RISK_TIMES, risk_counts.get(key, ['-'] * len(RISK_TIMES))):
            ax.text(x, y, str(value), ha='center', va='center', fontsize=6.3)


def km_stratum_panel(ax, risk_ax, sdf, gene, title, high_color, low_color):
    # Keys are (biomarker_high, trop2_high). Color distinguishes TROP2 with
    # the lab gold/blue palette; line style distinguishes biomarker strata.
    styles = {
        (1, 1): (high_color, '-'),
        (1, 0): (low_color, '-'),
        (0, 1): (high_color, '--'),
        (0, 0): (low_color, '--'),
    }
    atrisk_labels = {(1, 1): 'hi/hi', (1, 0): 'hi/lo', (0, 1): 'lo/hi', (0, 0): 'lo/lo'}
    fitters, logrank_p, risk_counts = {}, {}, {}
    for (b, t), (col, ls) in styles.items():
        g = sdf[(sdf['trop2_high'] == t) & (sdf[gene + '_high'] == b)]
        if len(g) < 10:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(g['duration'], g['event'], label=atrisk_labels[(b, t)])
        kmf.plot_survival_function(
            ax=ax, color=col, linestyle=ls, ci_show=False, lw=1.6,
            legend=False,
        )
        fitters[(b, t)] = kmf
        risk_counts[(b, t)] = [int((g['duration'] >= tm).sum()) for tm in RISK_TIMES]

    # log-rank test for TROP2-high vs low, computed separately WITHIN each
    # biomarker stratum.
    for b, tag in ((1, 'hi'), (0, 'lo')):
        hi = sdf[(sdf['trop2_high'] == 1) & (sdf[gene + '_high'] == b)]
        lo = sdf[(sdf['trop2_high'] == 0) & (sdf[gene + '_high'] == b)]
        if len(hi) >= 10 and len(lo) >= 10:
            lr = logrank_test(hi['duration'], lo['duration'],
                               event_observed_A=hi['event'], event_observed_B=lo['event'])
            logrank_p[b] = float(lr.p_value)
        else:
            logrank_p[b] = np.nan

    ax.set_title(title, fontsize=8.2, fontweight='bold', pad=4)
    ax.set_xlabel('Time (months)', fontsize=7.4, labelpad=3)
    ax.set_ylabel('OS probability', fontsize=7.4)
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, 1.03)
    ax.tick_params(labelsize=6.6)
    ax.text(0.98, 0.98,
            f"Biomarker-high: {fmt_p(logrank_p[1])}\n"
            f"Biomarker-low:  {fmt_p(logrank_p[0])}",
            transform=ax.transAxes, ha='right', va='top', fontsize=6.1,
            linespacing=1.25,
            bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white',
                  'edgecolor': '#BBBBBB', 'alpha': 0.9})
    clean_ax(ax)
    draw_risk_table(risk_ax, risk_counts)
    return logrank_p


def render_figure(pooled, stratifiers, stem, title, figsize,
                  high_color=None, low_color=None):
    stats, fig_rows = [], []
    high_color = high_color or PAL['accent_high']
    low_color = low_color or PAL['low']
    # Three dedicated layout bands prevent the KM panels, risk tables, and
    # forest labels from competing for the same space at manuscript width.
    n_cols = len(stratifiers)
    fig = plt.figure(figsize=figsize)
    single_panel = n_cols == 1
    height_ratios = [3.55, 1.20, 2.10] if single_panel else [2.9, 1.15, 2.35]
    top = 0.79 if single_panel else 0.82
    gs = fig.add_gridspec(
        3, n_cols, height_ratios=height_ratios,
        wspace=0.42, hspace=0.48,
        left=0.12 if single_panel else 0.085,
        right=0.975 if single_panel else 0.985,
        bottom=0.075, top=top,
    )

    for pi, (gene, disp) in enumerate(stratifiers.items()):
        sdf = pooled.dropna(subset=[gene + '_high'])
        hi = sdf[sdf[gene + '_high'] == 1]
        lo = sdf[sdf[gene + '_high'] == 0]
        hr_hi = cox_hr(hi)
        hr_lo = cox_hr(lo)
        ip = interaction_p(sdf, gene)
        ax = fig.add_subplot(gs[0, pi])
        risk_ax = fig.add_subplot(gs[1, pi])
        logrank_p = km_stratum_panel(
            ax, risk_ax, sdf, gene, disp, high_color, low_color)
        for strat, sdd, hr, b in ((f'{gene}-high', hi, hr_hi, 1), (f'{gene}-low', lo, hr_lo, 0)):
            stats.append({'stratifier': gene, 'stratum': strat, 'n': len(sdd),
                          'events': int(sdd['event'].sum()), 'HR': hr[0], 'HR_lo': hr[1],
                          'HR_hi': hr[2], 'cox_p': hr[3], 'logrank_p': logrank_p[b]})
        stats.append({'stratifier': gene, 'stratum': 'interaction', 'cox_p': ip})
        fig_rows.append((gene, disp, hr_hi, hr_lo, ip, len(hi), len(lo)))
        print(f'  {disp:>16}: hi-stratum HR {hr_hi[0]:.2f} ({hr_hi[1]:.2f}-{hr_hi[2]:.2f}) '
              f'cox p={hr_hi[3]:.3f} logrank {fmt_p(logrank_p[1])} | '
              f'lo-stratum HR {hr_lo[0]:.2f} ({hr_lo[1]:.2f}-{hr_lo[2]:.2f}) '
              f'cox p={hr_lo[3]:.3f} logrank {fmt_p(logrank_p[0])} | interaction {fmt_p(ip)}')

    # Full-width forest with dedicated label, estimate, and numeric columns.
    # Keeping labels out of the plotting axes eliminates spillover into KMs.
    forest_gs = gs[2, :].subgridspec(
        1, 3, width_ratios=[1.42, 1.15, 1.55], wspace=0.04,
    )
    label_ax = fig.add_subplot(forest_gs[0, 0])
    ax3 = fig.add_subplot(forest_gs[0, 1])
    estimate_ax = fig.add_subplot(forest_gs[0, 2], sharey=ax3)
    ax3.axvline(1.0, color='black', lw=0.8, ls='--', zorder=1)
    y_top = 1.05 + 2.45 * len(fig_rows)
    for i, (gene, disp, hr_hi, hr_lo, ip, n_hi, n_lo) in enumerate(fig_rows):
        high_y = y_top - 0.90 - 2.45 * i
        low_y = high_y - 0.68
        interaction_y = high_y - 1.30
        short = disp.split(' (')[0]
        for y, tag, hr, n in ((high_y, 'high', hr_hi, n_hi),
                              (low_y, 'low', hr_lo, n_lo)):
            label_ax.text(0.98, y, f'{short} {tag} (n={n})',
                          va='center', ha='right', fontsize=6.6)
            if np.isnan(hr[0]):
                continue
            col = high_color if tag == 'high' else low_color
            ax3.plot([hr[1], hr[2]], [y, y], color=col, lw=1.8, zorder=2)
            ax3.scatter([hr[0]], [y], s=42, color=col,
                        edgecolor='black', lw=0.5, zorder=3)
            estimate_ax.text(
                0.02, y,
                f'{hr[0]:.2f} [{hr[1]:.2f}-{hr[2]:.2f}], {fmt_p(hr[3])}',
                va='center', ha='left', fontsize=6.4, color=col,
            )
        label_ax.text(0.98, interaction_y, 'Interaction', va='center',
                      ha='right', fontsize=6.2, style='italic', color='#555555')
        estimate_ax.text(0.02, interaction_y, fmt_p(ip), va='center',
                         ha='left', fontsize=6.2, style='italic', color='#555555')
        if i < len(fig_rows) - 1:
            sep_y = interaction_y - 0.34
            for sep_ax in (label_ax, ax3, estimate_ax):
                sep_ax.axhline(sep_y, color='#DDDDDD', lw=0.6, zorder=0)

    label_ax.text(0.98, y_top, 'Biomarker stratum', va='center', ha='right',
                  fontsize=6.8, fontweight='bold')
    estimate_ax.text(0.02, y_top, 'HR [95% CI], p', va='center', ha='left',
                     fontsize=6.8, fontweight='bold')
    for table_ax in (label_ax, ax3, estimate_ax):
        table_ax.set_ylim(0.45, y_top + 0.30)

    ax3.set_xscale('log')
    ax3.set_xlim(0.5, 2.0)
    ax3.set_xticks([0.5, 0.75, 1, 1.5, 2])
    ax3.set_xticklabels(['0.5', '0.75', '1', '1.5', '2'])
    ax3.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax3.set_yticks([])
    ax3.set_xlabel('HR (TROP2-high vs low)', fontsize=6.8, labelpad=3)
    ax3.tick_params(axis='x', labelsize=6.2)
    clean_ax(ax3)

    label_ax.set_xlim(0, 1)
    label_ax.set_xticks([])
    label_ax.set_yticks([])
    label_ax.set_title('TROP2 effect by\nbiomarker stratum', fontsize=8.0,
                       fontweight='bold', loc='left', pad=5)
    for spine in label_ax.spines.values():
        spine.set_visible(False)

    estimate_ax.set_xlim(0, 1)
    estimate_ax.set_xticks([])
    estimate_ax.set_yticks([])
    estimate_ax.tick_params(left=False, labelleft=False)
    for spine in estimate_ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(title,
                 fontsize=10.5, fontweight='bold', y=0.975)
    semantic_handles = [
        Line2D([0], [0], color=high_color, lw=1.7, label='TROP2-high'),
        Line2D([0], [0], color=low_color, lw=1.7, label='TROP2-low'),
        Line2D([0], [0], color='#666666', lw=1.5, ls='-', label='Biomarker-high'),
        Line2D([0], [0], color='#666666', lw=1.5, ls='--', label='Biomarker-low'),
    ]
    fig.legend(
        semantic_handles, [h.get_label() for h in semantic_handles],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.905 if single_panel else 0.915),
        ncol=2 if single_panel else 4,
        frameon=False, fontsize=6.8, handlelength=1.8, columnspacing=1.1,
    )
    savefig(fig, OUT / stem)
    return stats


def main():
    apply_pub_style()
    all_dfs = []
    for name, cfg in DATASETS.items():
        try:
            all_dfs.append(load_cohort(name, cfg))
            print(f'  [{name}] loaded')
        except Exception as e:
            print(f'  [{name}] ERROR: {e}')
    pooled = pd.concat(all_dfs)

    main_stats = render_figure(
        pooled,
        MAIN_STRATIFIERS,
        'F4_TROP2_biomarker_stratified',
        'TROP2 OS within PD-L1 / CLDN18 strata',
        (6.0, 5.2),
    )
    her2_stats = render_figure(
        pooled,
        SUPPLEMENT_STRATIFIERS,
        'S2B_TROP2_HER2_stratified',
        'TROP2 OS within HER2 strata',
        (3.75, 5.2),
        high_color='#D1A12B',
        low_color='#5B8DB8',
    )
    pd.DataFrame(main_stats + her2_stats).to_csv(
        OUT / 'F4_TROP2_biomarker_stratified_stats.csv', index=False)
    print(f'  saved combined statistics to {OUT}/F4_TROP2_biomarker_stratified_stats.csv')


if __name__ == '__main__':
    main()

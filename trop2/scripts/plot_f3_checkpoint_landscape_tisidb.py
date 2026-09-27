"""TROP2 vs TISIDB immunomodulators."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from scipy.stats import spearmanr, mannwhitneyu

from cohort_utils import restrict_to_tumor
from style import PAL, clean_ax, savefig
from genolib import PROJECT_ROOT

BASE = PROJECT_ROOT
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
OUT.mkdir(parents=True, exist_ok=True)

TROP2 = 'TACSTD2'
HIGHLIGHT = 'VTCN1'  # B7-H4
MIN_COHORTS = 5      # gene must be testable in >= this many cohorts to enter the ranking

# TISIDB immunoinhibitors (24) and immunostimulators (45). symbol -> display.
# aliases handled in ALIAS below. (TISIDB classes CD276/B7-H3 and VSIR/VISTA as
# stimulators, we keep TISIDB's labels; B7-H4 is in the inhibitor list.)
INHIBITORS = {
    'ADORA2A': 'A2AR', 'BTLA': 'BTLA', 'CD160': 'CD160', 'CD244': 'CD244',
    'CD274': 'PD-L1', 'CD96': 'CD96', 'CSF1R': 'CSF1R', 'CTLA4': 'CTLA-4',
    'HAVCR2': 'TIM-3', 'IDO1': 'IDO1', 'IL10': 'IL10', 'IL10RB': 'IL10RB',
    'KDR': 'VEGFR2', 'KIR2DL1': 'KIR2DL1', 'KIR2DL3': 'KIR2DL3', 'LAG3': 'LAG-3',
    'LGALS9': 'Gal-9', 'PDCD1': 'PD-1', 'PDCD1LG2': 'PD-L2', 'PVRL2': 'CD112',
    'TGFB1': 'TGFB1', 'TGFBR1': 'TGFBR1', 'TIGIT': 'TIGIT', 'VTCN1': 'B7-H4',
}
STIMULATORS = {
    'C10orf54': 'VISTA', 'CD27': 'CD27', 'CD276': 'B7-H3', 'CD28': 'CD28',
    'CD40': 'CD40', 'CD40LG': 'CD40L', 'CD48': 'CD48', 'CD70': 'CD70',
    'CD80': 'CD80', 'CD86': 'CD86', 'CXCL12': 'CXCL12', 'CXCR4': 'CXCR4',
    'ENTPD1': 'CD39', 'HHLA2': 'HHLA2', 'ICOS': 'ICOS', 'ICOSLG': 'ICOSL',
    'IL2RA': 'CD25', 'IL6': 'IL6', 'IL6R': 'IL6R', 'KLRC1': 'NKG2A',
    'KLRK1': 'NKG2D', 'LTA': 'LTA', 'MICB': 'MICB', 'NT5E': 'CD73',
    'PVR': 'CD155', 'RAET1E': 'RAET1E', 'TMEM173': 'STING', 'TMIGD2': 'TMIGD2',
    'TNFRSF13B': 'TACI', 'TNFRSF13C': 'BAFFR', 'TNFRSF14': 'HVEM',
    'TNFRSF17': 'BCMA', 'TNFRSF18': 'GITR', 'TNFRSF25': 'DR3', 'TNFRSF4': 'OX40',
    'TNFRSF8': 'CD30', 'TNFRSF9': '4-1BB', 'TNFSF13': 'APRIL', 'TNFSF13B': 'BAFF',
    'TNFSF14': 'LIGHT', 'TNFSF15': 'TL1A', 'TNFSF18': 'GITRL', 'TNFSF4': 'OX40L',
    'TNFSF9': '4-1BBL', 'ULBP1': 'ULBP1',
}
CLASS = {**{g: 'inhibitor' for g in INHIBITORS}, **{g: 'stimulator' for g in STIMULATORS}}
DISPLAY = {**INHIBITORS, **STIMULATORS}

# legacy / alternate symbols that may appear in our matrices
ALIAS = {
    'C10orf54': ['VSIR'], 'PVRL2': ['NECTIN2'], 'TMEM173': ['STING1'],
    'PVR': ['NECTIN5', 'CD155'], 'TNFRSF9': ['CD137'], 'ENTPD1': ['CD39'],
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
}

CLASS_COLOR = {'inhibitor': PAL['tumor'], 'stimulator': PAL['low']}


def load_gene_rows(expr_file, wanted):
    found = {}
    with open(expr_file) as f:
        samples = f.readline().rstrip('\n').split('\t')[1:]
        flat = set(wanted)
        for line in f:
            tab = line.index('\t')
            gene = line[:tab]
            if gene in flat:
                vals = line[tab + 1:].rstrip('\n').split('\t')
                found[gene] = [float(v) if v.strip() else np.nan for v in vals]
    return pd.DataFrame(found, index=samples) if found else None


def resolve(df, symbol):
    if df is None:
        return None
    if symbol in df.columns:
        return df[symbol]
    for a in ALIAS.get(symbol, []):
        if a in df.columns:
            return df[a]
    return None


def main():
    request = {TROP2}
    for g in DISPLAY:
        request.add(g)
        request.update(ALIAS.get(g, []))

    rows = []
    for cohort, rel in COHORTS.items():
        df = load_gene_rows(BASE / rel, request)
        if df is None or TROP2 not in df.columns:
            print(f'[{cohort}] TACSTD2 missing - skipped')
            continue
        df = restrict_to_tumor(df, cohort)
        trop2 = df[TROP2]
        for g in DISPLAY:
            mvals = resolve(df, g)
            if mvals is None:
                rows.append({'cohort': cohort, 'symbol': g, 'rho': np.nan,
                             'rho_p': np.nan, 'delta': np.nan, 'mw_p': np.nan, 'n': 0})
                continue
            pair = pd.concat([trop2, mvals], axis=1).dropna()
            pair.columns = ['t', 'm']
            if len(pair) < 20 or pair['m'].nunique() < 3:
                rows.append({'cohort': cohort, 'symbol': g, 'rho': np.nan,
                             'rho_p': np.nan, 'delta': np.nan, 'mw_p': np.nan, 'n': len(pair)})
                continue
            rho, rp = spearmanr(pair['t'], pair['m'])
            med = pair['t'].median()
            hi, lo = pair.loc[pair['t'] >= med, 'm'], pair.loc[pair['t'] < med, 'm']
            if len(hi) >= 5 and len(lo) >= 5:
                _, mwp = mannwhitneyu(hi, lo, alternative='two-sided')
                delta = float(hi.median() - lo.median())
            else:
                mwp, delta = np.nan, np.nan
            rows.append({'cohort': cohort, 'symbol': g, 'rho': rho, 'rho_p': rp,
                         'delta': delta, 'mw_p': mwp, 'n': len(pair)})

    stats = pd.DataFrame(rows)
    stats['display'] = stats['symbol'].map(DISPLAY)
    stats['class'] = stats['symbol'].map(CLASS)
    stats.to_csv(OUT / 'F3_checkpoint_landscape_TISIDB_stats.csv', index=False)

    cohort_order = [c for c in COHORTS if c in stats['cohort'].unique()]

    # per-gene aggregate; drop genes testable in < MIN_COHORTS
    agg = []
    for g in DISPLAY:
        s = stats[stats['symbol'] == g]
        tested = int(s['rho'].notna().sum())
        if tested < MIN_COHORTS:
            continue
        rhos = s['rho'].dropna()
        sig = s.dropna(subset=['mw_p'])
        up = int(((sig['mw_p'] < 0.05) & (sig['delta'] > 0)).sum())
        down = int(((sig['mw_p'] < 0.05) & (sig['delta'] < 0)).sum())
        agg.append({'symbol': g, 'display': DISPLAY[g], 'class': CLASS[g],
                    'mean_rho': rhos.mean(), 'tested': tested, 'up': up, 'down': down})
    agg = pd.DataFrame(agg).sort_values('mean_rho', ascending=False).reset_index(drop=True)

    dropped = [DISPLAY[g] for g in DISPLAY
               if g not in set(agg['symbol']) ]
    print(f'\n{len(agg)} genes ranked; {len(dropped)} dropped (<{MIN_COHORTS} cohorts testable): '
          f'{", ".join(dropped)}')

    # build ρ matrix (genes x cohorts), gene order = agg
    rho_mat = stats.pivot(index='symbol', columns='cohort', values='rho')\
                   .reindex(index=agg['symbol']).reindex(columns=cohort_order)
    p_mat = stats.pivot(index='symbol', columns='cohort', values='rho_p')\
                 .reindex(index=agg['symbol']).reindex(columns=cohort_order)

    n_g, n_c = rho_mat.shape
    fig = plt.figure(figsize=(n_c * 0.55 + 8.0, n_g * 0.26 + 2.2),
                     constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[n_c, 4.2, 0.18], wspace=0.03)
    ax_hm = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1], sharey=ax_hm)
    ax_cb = fig.add_subplot(gs[2])

    # HONEST color scale: fix to the full Spearman range [-1, 1] rather than
    # auto-scaling to the data max. Auto-scaling makes a modest ρ≈0.3 saturate to
    # strong red and "look" strong (a scale that stops short of 1 inflates weak
    # signals). Magnitude is read precisely off the mean-ρ
    # bar (real numeric axis) and significance off the * cells; color is a guide.
    vmax = 1.0

    # heatmap
    im = ax_hm.imshow(rho_mat.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    for i in range(n_g):
        for j in range(n_c):
            v = rho_mat.values[i, j]
            if np.isnan(v):
                ax_hm.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, color='#eeeeee', zorder=2))
            else:
                p = p_mat.values[i, j]
                if not np.isnan(p) and p < 0.05:
                    ax_hm.text(j, i, '*', ha='center', va='center', fontsize=8.5,
                               color='white' if abs(v) > 0.55 * vmax else 'black', zorder=3)
    ax_hm.set_xticks(range(n_c))
    ax_hm.set_xticklabels(cohort_order, rotation=45, ha='right', fontsize=8.5)
    # gene labels directly on the heatmap, colored by TISIDB class
    ax_hm.set_yticks(range(n_g))
    ax_hm.set_yticklabels(agg['display'], fontsize=8)
    for tick, (_, r) in zip(ax_hm.get_yticklabels(), agg.iterrows()):
        tick.set_color(CLASS_COLOR[r['class']])
        if r['symbol'] == HIGHLIGHT:
            tick.set_fontweight('bold'); tick.set_color('#b8860b'); tick.set_fontsize(10)
    ax_hm.set_ylim(n_g - 0.5, -0.5)
    ax_hm.set_title('Spearman ρ (checkpoint vs TROP2 / TACSTD2)  ·  * = p<0.05',
                    fontsize=10.5, pad=6)
    ax_hm.tick_params(length=0)

    # mean-ρ bar
    y = np.arange(n_g)
    bar_colors = [CLASS_COLOR[c] for c in agg['class']]
    ax_bar.barh(y, agg['mean_rho'], color=bar_colors, alpha=0.85, zorder=3)
    ax_bar.axvline(0, color='black', lw=0.8)
    for yi, (_, r) in enumerate(agg.iterrows()):
        tag = f"{r['mean_rho']:+.2f}  ↑{r['up']} ↓{r['down']}/{r['tested']}"
        xoff = 0.006 if r['mean_rho'] >= 0 else -0.006
        ax_bar.text(r['mean_rho'] + xoff, yi, tag, va='center',
                    ha='left' if r['mean_rho'] >= 0 else 'right', fontsize=6.6)
    ax_bar.set_ylim(n_g - 0.5, -0.5)
    xm = float(np.nanmax(np.abs(agg['mean_rho']))) * 1.1
    ax_bar.set_xlim(-xm, xm * 2.1)  # room for text
    ax_bar.set_title('mean ρ  ·  ↑sig-up ↓sig-down / n', fontsize=9, pad=6)
    ax_bar.spines[['top', 'right', 'left']].set_visible(False)
    ax_bar.tick_params(labelleft=False, length=0)
    ax_bar.set_xlabel('mean Spearman ρ', fontsize=8.5)

    # box B7-H4 across heatmap + bar
    hl_i = int(agg.index[agg['symbol'] == HIGHLIGHT][0])
    ax_hm.add_patch(Rectangle((-0.5, hl_i - 0.5), n_c, 1, fill=False,
                              edgecolor='#b8860b', lw=2.2, zorder=5))

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('Spearman ρ  (scale fixed to full range −1…+1)', fontsize=8.5)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1]); cb.ax.tick_params(labelsize=7)

    legend_handles = [Patch(facecolor=CLASS_COLOR['inhibitor'], label='TISIDB immunoinhibitor'),
                      Patch(facecolor=CLASS_COLOR['stimulator'], label='TISIDB immunostimulator')]
    ax_hm.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.0, -0.025),
                 ncol=2, fontsize=8.5, frameon=False)

    top = agg.iloc[0]
    fig.suptitle('TROP2 × TISIDB immune-checkpoint panel',
                 fontsize=12, fontweight='bold')
    savefig(fig, OUT / 'F3_checkpoint_landscape_TISIDB')

    print('\nTop of ranking (mean ρ vs TROP2):')
    for _, r in agg.head(8).iterrows():
        flag = '  <-- B7-H4' if r['symbol'] == HIGHLIGHT else ''
        print(f"  {r['display']:>8} ({r['class'][:5]}): ρ={r['mean_rho']:+.3f} "
              f"↑{r['up']} ↓{r['down']}/{r['tested']}{flag}")


if __name__ == '__main__':
    main()

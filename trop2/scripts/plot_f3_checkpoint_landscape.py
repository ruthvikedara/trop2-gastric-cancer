"""TROP2 vs immune checkpoint genes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr, mannwhitneyu

from cohort_utils import restrict_to_tumor
from style import (PAL, clean_ax, savefig, apply_pub_style, axis_label,
                   LABEL_FS, TITLE_FS, TICK_FS, LEGEND_FS, ANNOT_FS)
from genolib import PROJECT_ROOT

BASE = PROJECT_ROOT
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
OUT.mkdir(parents=True, exist_ok=True)

TROP2 = 'TACSTD2'
HIGHLIGHT = 'VTCN1'  # B7-H4

MARKERS = {
    'VTCN1':    ('B7-H4',  'ligand',   []),
    'CD274':    ('PD-L1',  'ligand',   []),
    'PDCD1LG2': ('PD-L2',  'ligand',   []),
    'CD276':    ('B7-H3',  'ligand',   []),
    'VSIR':     ('VISTA',  'ligand',   ['C10orf54']),
    'LGALS9':   ('Gal-9',  'ligand',   []),
    'IDO1':     ('IDO1',   'enzyme',   []),
    'PDCD1':    ('PD-1',   'receptor', []),
    'CTLA4':    ('CTLA-4', 'receptor', []),
    'LAG3':     ('LAG-3',  'receptor', []),
    'HAVCR2':   ('TIM-3',  'receptor', []),
    'TIGIT':    ('TIGIT',  'receptor', []),
    'BTLA':     ('BTLA',   'receptor', []),
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

# Class colours: gold ligands, charcoal receptors, muted red enzyme
SIDE_COLOR = {
    'ligand':   PAL['accent_high'],
    'receptor': '#4A90C8',
    'enzyme':   '#B85C5C',
}
SIDE_LABEL = {
    'ligand': 'tumor-side ligand',
    'receptor': 'T-cell-side receptor',
    'enzyme': 'suppressive enzyme',
}
HL_BAND = '#F5E6A8'


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
    for alias in MARKERS[symbol][2]:
        if alias in df.columns:
            return df[alias]
    return None


def main():
    apply_pub_style()
    request = set([TROP2])
    for sym, (_, _, aliases) in MARKERS.items():
        request.add(sym)
        request.update(aliases)

    rows = []
    for cohort, rel in COHORTS.items():
        df = load_gene_rows(BASE / rel, request)
        if df is None or TROP2 not in df.columns:
            print(f'[{cohort}] TACSTD2 missing - skipped')
            continue
        df = restrict_to_tumor(df, cohort)
        trop2 = df[TROP2]
        for sym, (disp, side, _) in MARKERS.items():
            mvals = resolve(df, sym)
            if mvals is None:
                rows.append({'cohort': cohort, 'symbol': sym, 'display': disp, 'side': side,
                             'n': 0, 'rho': np.nan, 'rho_p': np.nan, 'delta': np.nan, 'mw_p': np.nan})
                continue
            pair = pd.concat([trop2, mvals], axis=1).dropna()
            pair.columns = ['t', 'm']
            if len(pair) < 20:
                rows.append({'cohort': cohort, 'symbol': sym, 'display': disp, 'side': side,
                             'n': len(pair), 'rho': np.nan, 'rho_p': np.nan,
                             'delta': np.nan, 'mw_p': np.nan})
                continue
            rho, rp = spearmanr(pair['t'], pair['m'])
            med = pair['t'].median()
            hi = pair.loc[pair['t'] >= med, 'm']
            lo = pair.loc[pair['t'] < med, 'm']
            if len(hi) >= 5 and len(lo) >= 5:
                _, mwp = mannwhitneyu(hi, lo, alternative='two-sided')
                delta = float(hi.median() - lo.median())
            else:
                mwp, delta = np.nan, np.nan
            rows.append({'cohort': cohort, 'symbol': sym, 'display': disp, 'side': side,
                         'n': len(pair), 'rho': rho, 'rho_p': rp, 'delta': delta, 'mw_p': mwp})

    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / 'F3_checkpoint_landscape_stats.csv', index=False)

    agg = []
    for sym, (disp, side, _) in MARKERS.items():
        s = stats[stats['symbol'] == sym]
        rhos = s['rho'].dropna()
        n_tested = int((s['mw_p'].notna()).sum())
        sig = s.dropna(subset=['mw_p'])
        up = int(((sig['mw_p'] < 0.05) & (sig['delta'] > 0)).sum())
        down = int(((sig['mw_p'] < 0.05) & (sig['delta'] < 0)).sum())
        ns = n_tested - up - down
        agg.append({'symbol': sym, 'display': disp, 'side': side,
                    'mean_rho': rhos.mean() if len(rhos) else np.nan,
                    'n_tested': n_tested, 'up': up, 'down': down, 'ns': ns})
    agg = pd.DataFrame(agg).sort_values('mean_rho', ascending=True).reset_index(drop=True)

    y = np.arange(len(agg))
    rng = np.random.default_rng(0)
    n_m = len(agg)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.8, 0.48 * n_m + 2.0),
        gridspec_kw={'width_ratios': [1.55, 1.0], 'wspace': 0.08},
    )
    fig.suptitle(
        'TROP2 vs standard immune-checkpoint panel - B7-H4 is the standout',
        fontsize=TITLE_FS + 1, fontweight='bold',
    )

    hl_y = int(agg.index[agg['symbol'] == HIGHLIGHT][0])
    for ax in (axL, axR):
        ax.axhspan(hl_y - 0.48, hl_y + 0.48, color=HL_BAND, zorder=0, lw=0)

    # panel a: Spearman strip
    for yi, sym in enumerate(agg['symbol']):
        s = stats[(stats['symbol'] == sym)].dropna(subset=['rho'])
        side = MARKERS[sym][1]
        base_color = SIDE_COLOR[side]
        for _, r in s.iterrows():
            filled = (not np.isnan(r['rho_p'])) and r['rho_p'] < 0.05
            jit = rng.uniform(-0.14, 0.14)
            axL.scatter(
                r['rho'], yi + jit, s=32,
                facecolor=base_color if filled else 'none',
                edgecolor=base_color, linewidth=1.1,
                alpha=0.90 if filled else 0.55, zorder=3,
            )
        mr = agg.loc[agg['symbol'] == sym, 'mean_rho'].values[0]
        axL.scatter(
            mr, yi, marker='D', s=78, color=base_color,
            edgecolor=PAL['accent_low'], linewidth=0.9, zorder=4,
        )

    axL.axvline(0, color=PAL['accent_low'], lw=0.95, zorder=1)
    axL.set_yticks(y)
    axL.set_yticklabels([f'{d}' for d in agg['display']], fontsize=TICK_FS + 1.5)
    for tick, sym in zip(axL.get_yticklabels(), agg['symbol']):
        if sym == HIGHLIGHT:
            tick.set_fontweight('bold')
            tick.set_color(PAL['accent_high'])
            tick.set_fontsize(TICK_FS + 2.5)
    axL.set_ylim(-0.55, n_m - 0.45)
    axL.set_xlabel(
        axis_label('Spearman ρ', unit='checkpoint vs TROP2'),
        fontsize=LABEL_FS,
    )
    axL.set_title('(a) Per-cohort ρ  (◆ mean; filled p < 0.05)',
                  fontsize=TITLE_FS - 1, pad=5)
    clean_ax(axL)
    xm = max(0.12, float(np.nanmax(np.abs(stats['rho'].values))) * 1.08)
    axL.set_xlim(-xm, xm)
    axL.tick_params(axis='x', labelsize=TICK_FS)
    axL.xaxis.grid(True, linestyle=':', linewidth=0.6, color='#CCCCCC', zorder=0)
    axL.set_axisbelow(True)

    # panel b: consistency stacked bar
    up_c, ns_c, dn_c = PAL['tumor'], '#C8C8C8', PAL['low']
    for yi, sym in enumerate(agg['symbol']):
        r = agg[agg['symbol'] == sym].iloc[0]
        axR.barh(yi, r['up'], color=up_c, height=0.72, zorder=3, edgecolor='none')
        axR.barh(yi, r['ns'], left=r['up'], color=ns_c, height=0.72, zorder=3, edgecolor='none')
        axR.barh(yi, r['down'], left=r['up'] + r['ns'], color=dn_c, height=0.72,
                 zorder=3, edgecolor='none')
        if r['up']:
            axR.text(r['up'] / 2, yi, str(int(r['up'])), ha='center', va='center',
                     fontsize=ANNOT_FS, color='white', fontweight='bold', zorder=4)
        if r['down']:
            axR.text(r['up'] + r['ns'] + r['down'] / 2, yi, str(int(r['down'])),
                     ha='center', va='center', fontsize=ANNOT_FS, color='white',
                     fontweight='bold', zorder=4)

    axR.set_yticks(y)
    axR.set_yticklabels([])
    axR.set_ylim(-0.55, n_m - 0.45)
    axR.set_xlabel(
        axis_label('# cohorts', unit='median-split MW p < 0.05'),
        fontsize=LABEL_FS,
    )
    axR.set_title('(b) Consistency in TROP2-high vs low',
                  fontsize=TITLE_FS - 1, pad=5)
    clean_ax(axR)
    axR.tick_params(axis='x', labelsize=TICK_FS)
    xmax = int(agg[['up', 'ns', 'down']].sum(axis=1).max())
    axR.set_xlim(0, xmax + 0.4)

    side_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SIDE_COLOR[s],
               markeredgecolor=SIDE_COLOR[s], markersize=9, label=SIDE_LABEL[s])
        for s in ['ligand', 'receptor', 'enzyme']
    ]
    axL.legend(
        handles=side_handles, loc='lower left', fontsize=LEGEND_FS,
        frameon=False, title='checkpoint class', title_fontsize=LEGEND_FS,
        handletextpad=0.35, borderaxespad=0.2,
    )
    bar_handles = [
        Patch(facecolor=up_c, label='UP in TROP2-high'),
        Patch(facecolor=ns_c, label='n.s.'),
        Patch(facecolor=dn_c, label='DOWN in TROP2-high'),
    ]
    legR = axR.legend(
        handles=bar_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0),
        fontsize=LEGEND_FS, frameon=False, handletextpad=0.35,
    )

    fig.tight_layout(rect=(0.0, 0.0, 0.90, 0.93))
    # keep suptitle in the tight bbox
    savefig(fig, OUT / 'F3_checkpoint_landscape',
            extra_artists=[legR, fig._suptitle], pad_inches=0.08)

    for _, r in agg.sort_values('mean_rho', ascending=False).iterrows():
        flag = '  <-- B7-H4' if r['symbol'] == HIGHLIGHT else ''
        print(f"  {r['display']:>7} ({r['side']:>8}): mean ρ={r['mean_rho']:+.3f} | "
              f"up {r['up']}/{r['n_tested']}, down {r['down']}{flag}")


if __name__ == '__main__':
    main()

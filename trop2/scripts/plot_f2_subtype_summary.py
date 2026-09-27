"""TROP2 by molecular and Lauren subtype."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal

from cohort_utils import restrict_to_tumor
from style import (PAL, clean_ax, savefig, fmt_p, apply_pub_style,
                   TICK_FS, LABEL_FS, TITLE_FS, ANNOT_FS)
from genolib import PROJECT_ROOT

BASE = PROJECT_ROOT
PROC = BASE / 'data/clinical/gastric/processed'
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
OUT.mkdir(parents=True, exist_ok=True)
LARGE_TEXT = '--large-text' in sys.argv
OUTPUT_STEM = ('F2_subtype_summary_bubble_large_text'
               if LARGE_TEXT else 'F2_subtype_summary_bubble')
FS_BUMP = 2.2 if LARGE_TEXT else 1.7

TROP2 = 'TACSTD2'
TCGA_ORDER = ['EBV', 'MSI', 'GS', 'CIN']
EMP_ORDER = ['EP', 'MP']
LAUREN_ORDER = ['Intestinal', 'Diffuse', 'Mixed']
LAUREN_LABEL = {'Intestinal': 'Int', 'Diffuse': 'Diff', 'Mixed': 'Mix'}
FLAG = {'GSE84437': ' *'}  # degenerate TCGA-mode (425 CIN / 2 GS)

# Lauren histology is recorded only in these four cohorts' curated clinical
# tables; the remaining GEO series carry no Lauren annotation, so their Lauren
# cells stay empty.
LAUREN_CLIN = {
    'TCGA-STAD': ('TCGA_clinical.csv', 'patient_id'),
    'GSE66229':  ('GSE66229_clinical.csv', 'gsm_id'),
    'GSE15459':  ('GSE15459_clinical.csv', 'gsm_id'),
    'GSE34942':  ('GSE34942_clinical.csv', 'gsm_id'),
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


def trop2_series(expr_file):
    with open(expr_file) as f:
        samples = f.readline().rstrip('\n').split('\t')[1:]
        for line in f:
            tab = line.index('\t')
            if line[:tab] == TROP2:
                vals = line[tab + 1:].rstrip('\n').split('\t')
                return pd.Series([float(v) if v.strip() else np.nan for v in vals],
                                 index=samples)


def norm_idx(idx):
    return idx.map(lambda x: '-'.join(str(x).split('-')[:3])
                   if str(x).startswith('TCGA') else str(x))


def labels_for(cohort, system):
    """Series sample_id -> subtype for system in {'TCGA','EMP','LAUREN'}."""
    if system == 'LAUREN':
        cfg = LAUREN_CLIN.get(cohort)
        if cfg is None:
            return None
        fname, id_col = cfg
        d = pd.read_csv(PROC / fname, low_memory=False)
        if 'lauren' not in d.columns:
            return None
        lab = d.set_index(id_col)['lauren'].map(
            lambda v: {'intestinal': 'Intestinal', 'diffuse': 'Diffuse',
                       'mixed': 'Mixed'}.get(str(v).strip().lower())
            if pd.notna(v) else None)
        return lab.dropna()
    if system == 'TCGA' and cohort == 'TCGA-STAD':
        t = pd.read_csv(PROC / 'TCGA_clinical_with_subtypes.csv')
        return t.set_index('patient_id')['molecular_subtype']
    f = PROC / f'{cohort}_subtypes_predicted.csv'
    if not f.exists():
        return None
    d = pd.read_csv(f)
    col = f'{system}_subtype'
    if col not in d.columns:
        return None
    return d.set_index('sample_id')[col]


def cohort_cells(expr, labs, order):
    """Per-subtype stats vs cohort median. Returns dict subtype -> (n, delta, mw_p)."""
    e = expr.copy(); e.index = norm_idx(e.index)
    l = labs.copy(); l.index = norm_idx(l.index)
    df = pd.DataFrame({'trop2': e}).join(pd.DataFrame({'sub': l}), how='inner').dropna()
    if len(df) < 10:
        return {}
    med_all = df['trop2'].median()
    out = {}
    for g in order:
        v = df.loc[df['sub'] == g, 'trop2']
        rest = df.loc[df['sub'] != g, 'trop2']
        if len(v) < 3:
            continue
        p = mannwhitneyu(v, rest).pvalue if len(rest) >= 3 else np.nan
        out[g] = (len(v), float(v.median() - med_all), p)
    return out


def main():
    rows_tcga, rows_emp, rows_lau = {}, {}, {}
    tcga_omnibus_ps, emp_omnibus_ps, lau_omnibus_ps = {}, {}, {}
    pooled_z = []
    stats_rows = []
    for cohort, rel in COHORTS.items():
        expr = trop2_series(BASE / rel)
        if expr is None:
            continue
        expr = restrict_to_tumor(expr, cohort).dropna()
        z = (expr - expr.mean()) / expr.std()
        z.index = norm_idx(z.index)
        pooled_z.append(z.rename(cohort))

        lt, le = labels_for(cohort, 'TCGA'), labels_for(cohort, 'EMP')
        ll = labels_for(cohort, 'LAUREN')
        ct = cohort_cells(expr, lt, TCGA_ORDER) if lt is not None else {}
        ce = cohort_cells(expr, le, EMP_ORDER) if le is not None else {}
        cl = cohort_cells(expr, ll, LAUREN_ORDER) if ll is not None else {}
        rows_tcga[cohort], rows_emp[cohort], rows_lau[cohort] = ct, ce, cl

        e = expr.copy(); e.index = norm_idx(e.index)
        for labs, order, store in (
                (lt, TCGA_ORDER, tcga_omnibus_ps),
                (le, EMP_ORDER, emp_omnibus_ps),
                (ll, LAUREN_ORDER, lau_omnibus_ps)):
            if labs is None:
                store[cohort] = np.nan; continue
            l = labs.copy(); l.index = norm_idx(l.index)
            df = pd.DataFrame({'trop2': e}).join(pd.DataFrame({'sub': l}), how='inner').dropna()
            gr = [df.loc[df['sub'] == g, 'trop2'].values for g in order
                  if (df['sub'] == g).sum() >= 3]
            store[cohort] = kruskal(*gr).pvalue if len(gr) >= 2 else np.nan
        for g, (n, d, p) in {**{('TCGA', k): v for k, v in ct.items()},
                             **{('EMP', k): v for k, v in ce.items()},
                             **{('Lauren', k): v for k, v in cl.items()}}.items():
            stats_rows.append({'cohort': cohort, 'system': g[0], 'subtype': g[1],
                               'n': n, 'delta_vs_cohort_median': d, 'mw_p': p})
        print(f'[{cohort}] TCGA cells {list(ct)} | EMP cells {list(ce)} '
              f'| Lauren cells {list(cl)}')

    # pooled row (within-cohort z-scores)
    zdf = pd.concat(pooled_z)
    pooled_t, pooled_e, pooled_l = {}, {}, {}
    for system, order, store in (('TCGA', TCGA_ORDER, pooled_t),
                                 ('EMP', EMP_ORDER, pooled_e),
                                 ('LAUREN', LAUREN_ORDER, pooled_l)):
        labs = []
        for cohort in COHORTS:
            l = labels_for(cohort, system)
            if l is not None:
                l = l.copy(); l.index = norm_idx(l.index)
                labs.append(l)
        labs = pd.concat(labs)
        df = pd.DataFrame({'z': zdf}).join(pd.DataFrame({'sub': labs}),
                                           how='inner').dropna()
        for g in order:
            v = df.loc[df['sub'] == g, 'z']
            rest = df.loc[df['sub'] != g, 'z']
            if len(v) >= 10:
                store[g] = (len(v), float(v.median() - df['z'].median()),
                            mannwhitneyu(v, rest).pvalue)
    return rows_tcga, rows_emp, rows_lau, \
        tcga_omnibus_ps, emp_omnibus_ps, lau_omnibus_ps, \
        cohorts_order(rows_tcga), \
        pooled_t, pooled_e, pooled_l, stats_rows


def cohorts_order(rows_tcga):
    return list(rows_tcga.keys())



def make_figure(rows_tcga, rows_emp, rows_lau, tcga_omnibus_ps, emp_omnibus_ps,
                lau_omnibus_ps, cohorts, pooled_t, pooled_e, pooled_l,
                stats_rows):
    # x layout: TCGA 0-3 | omnibus 3.95 || EMP 5,6 | omnibus 7.05 ||
    # Lauren 8.0-9.8 | omnibus 10.85
    xs = {g: i for i, g in enumerate(TCGA_ORDER)}
    xs.update({g: len(TCGA_ORDER) + 1 + i for i, g in enumerate(EMP_ORDER)})
    xs.update({g: 8.0 + 0.9 * i for i, g in enumerate(LAUREN_ORDER)})
    vmax = 1.4
    row_labels = ['POOLED'] + cohorts
    all_rows = [('POOLED', pooled_t, pooled_e, pooled_l)] + \
               [(c, rows_tcga[c], rows_emp[c], rows_lau[c]) for c in cohorts]

    fig, ax = plt.subplots(figsize=(11.0, 7.85))
    cmap = plt.get_cmap('RdBu_r')
    for yi, (name, ct, ce, cl) in enumerate(all_rows):
        for g, x in xs.items():
            if g in TCGA_ORDER:
                cell = ct.get(g)
            elif g in EMP_ORDER:
                cell = ce.get(g)
            else:
                cell = cl.get(g)
            if not cell:
                continue
            n, delta, p = cell
            color = cmap(0.5 + 0.5 * np.clip(delta / vmax, -1, 1))
            sig = (not np.isnan(p)) and p < 0.05
            ax.scatter(x, yi, s=30 + 1.2 * n, color=color,
                       edgecolor='black' if sig else '#999999',
                       lw=1.7 if sig else 0.4, zorder=3)
        if name != 'POOLED':
            tcga_p = tcga_omnibus_ps.get(name, np.nan)
            emp_p = emp_omnibus_ps.get(name, np.nan)
            ax.text(3.95, yi, fmt_p(tcga_p).replace('p = ', ''),
                    va='center', ha='center',
                    fontsize=ANNOT_FS - 0.3 + FS_BUMP)
            ax.text(7.05, yi, fmt_p(emp_p).replace('p = ', ''),
                    va='center', ha='center',
                    fontsize=ANNOT_FS - 0.3 + FS_BUMP)
            lau_p = lau_omnibus_ps.get(name, np.nan)
            ax.text(10.85, yi, fmt_p(lau_p).replace('p = ', ''),
                    va='center', ha='center',
                    fontsize=ANNOT_FS - 0.3 + FS_BUMP)

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([r + FLAG.get(r, '') for r in row_labels],
                       fontsize=LABEL_FS - 0.5 + FS_BUMP)
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels([LAUREN_LABEL.get(g, g) for g in xs],
                       fontsize=LABEL_FS + 0.5 + FS_BUMP)
    ax.axvline(4.48, color='#BBBBBB', lw=1)
    ax.axvline(7.62, color='#BBBBBB', lw=1)
    ax.text(1.5, -0.9, 'TCGA subtypes', ha='center',
            fontsize=LABEL_FS + 0.5 + FS_BUMP,
            fontweight='bold')
    ax.text(5.5, -0.9, 'EMP', ha='center',
            fontsize=LABEL_FS + 0.5 + FS_BUMP,
            fontweight='bold')
    ax.text(8.9, -0.9, 'Lauren', ha='center',
            fontsize=LABEL_FS + 0.5 + FS_BUMP,
            fontweight='bold')
    ax.text(3.95, -0.58, 'TCGA\nomnibus p',
            fontsize=ANNOT_FS - 0.5 + FS_BUMP,
            ha='center', va='bottom', fontweight='bold', linespacing=0.9)
    ax.text(7.05, -0.58, 'EMP\nomnibus p',
            fontsize=ANNOT_FS - 0.5 + FS_BUMP,
            ha='center', va='bottom', fontweight='bold', linespacing=0.9)
    ax.text(10.85, -0.58, 'Lauren\nomnibus p',
            fontsize=ANNOT_FS - 0.5 + FS_BUMP,
            ha='center', va='bottom', fontweight='bold', linespacing=0.9)
    ax.set_xlim(-0.55, 11.45)
    ax.set_ylim(len(row_labels) - 0.45, -1.3)
    ax.set_title(
        'color = median TROP2 vs cohort median (log$_2$)  ·  '
        'size = subtype n  ·  bold outline = subtype-vs-rest Mann-Whitney p<0.05\n'
        'row-level omnibus p = Kruskal-Wallis across displayed subtypes',
        fontsize=ANNOT_FS + 0.2 + FS_BUMP,
        pad=28, linespacing=1.20)
    fig.suptitle('TROP2 expression by molecular subtype across cohorts',
                 fontsize=TITLE_FS + 1.5 + FS_BUMP, fontweight='bold')
    clean_ax(ax, keep_left=True)

    import matplotlib.colors as mcolors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(-vmax, vmax))
    # Dedicated lower strip: color scale on the left, sample-size key on the
    # right. The prior ax-anchored legend overlapped the colorbar label and
    # expanded the tight bounding box into a large blank lower margin.
    cax = fig.add_axes([0.13, 0.105, 0.37, 0.025])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('TROP2 vs cohort median (log$_2$)',
                 fontsize=ANNOT_FS + 0.5 + FS_BUMP, labelpad=2)
    cb.ax.tick_params(labelsize=TICK_FS + FS_BUMP)
    size_handles = []
    for n_ex in (30, 150, 400):
        size_handles.append(
            plt.Line2D(
                [0], [0], marker='o', linestyle='none',
                markersize=np.sqrt(30 + 1.2 * n_ex),
                markerfacecolor='#BBBBBB', markeredgecolor='black',
                markeredgewidth=0.6, label=f'n={n_ex}',
            )
        )
    fig.legend(
        handles=size_handles, title='Subtype n',
        fontsize=ANNOT_FS + FS_BUMP, title_fontsize=ANNOT_FS + FS_BUMP,
        loc='center', bbox_to_anchor=(0.68, 0.117),
        frameon=False, ncol=3, handletextpad=0.35, columnspacing=0.75,
    )

    fig.text(0.13, 0.018,
             '* TCGA-mode subtype calls degenerate in GSE84437 (425 CIN / 2 GS). '
             'Lauren histology is annotated only in '
             'TCGA-STAD, GSE66229, GSE15459 and GSE34942.',
             fontsize=8 + FS_BUMP, color='#666666', ha='left')
    fig.subplots_adjust(left=0.13, right=0.985, top=0.82, bottom=0.20)
    savefig(fig, OUT / OUTPUT_STEM)
    pd.DataFrame(stats_rows).to_csv(OUT / f'{OUTPUT_STEM}_stats.csv',
                                    index=False)
    print('\nPOOLED TCGA: ' + ', '.join(
        f'{g} {pooled_t.get(g, (0, 0.0, np.nan))[1]:+.2f}' for g in TCGA_ORDER))
    print('POOLED EMP:  ' + ', '.join(
        f'{g} {pooled_e.get(g, (0, 0.0, np.nan))[1]:+.2f}' for g in EMP_ORDER))
    print('POOLED Lauren: ' + ', '.join(
        f'{g} {pooled_l.get(g, (0, 0.0, np.nan))[1]:+.2f}' for g in LAUREN_ORDER))


if __name__ == '__main__':
    apply_pub_style()
    r = main()
    make_figure(*r)

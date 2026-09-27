"""TROP2 vs PD-L1, adjusted for immune content."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata, t as tdist

from cohort_utils import restrict_to_tumor
from style import PAL, clean_ax, savefig
from genolib import PROJECT_ROOT

BASE = PROJECT_ROOT
PROC = BASE / 'data/clinical/gastric/processed'
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'phase1'
OUT.mkdir(parents=True, exist_ok=True)

MARKERS = {'CD274': 'PD-L1 (CD274)', 'VTCN1': 'B7-H4 (VTCN1)', 'PDCD1': 'PD-1 (PDCD1)'}

COHORTS = {
    'TCGA-STAD': 'TCGA_STAD_for_xcell.txt',
    'GSE66229':  'ACRG_GSE66229_for_xcell.txt',
    'GSE15459':  'ACRG_GSE15459_for_xcell.txt',
    'GSE34942':  'ACRG_GSE34942_for_xcell.txt',
    'GSE35809':  'ACRG_GSE35809_for_xcell.txt',
    'GSE51105':  'ACRG_GSE51105_for_xcell.txt',
    'GSE54129':  'ACRG_GSE54129_for_xcell.txt',
    'GSE57303':  'ACRG_GSE57303_for_xcell.txt',
    'GSE84437':  'ACRG_GSE84437_for_xcell.txt',
}
PLATFORM_FLAG = {'GSE84437'}  # uniformly-positive checkpoint column


def gene_series(expr_file, wanted):
    """One pass over the gene x sample matrix -> {gene: Series} for genes in `wanted`."""
    out = {}
    with open(expr_file) as f:
        samples = f.readline().rstrip('\n').split('\t')[1:]
        for line in f:
            tab = line.index('\t')
            gene = line[:tab]
            if gene in wanted:
                vals = line[tab + 1:].rstrip('\n').split('\t')
                out[gene] = pd.Series([float(v) if v.strip() else np.nan for v in vals],
                                      index=samples)
    return out


def partial_spearman(x, y, Z):
    """Partial Spearman r(x, y | Z): rank both, residualize on ranked Z (OLS), Pearson."""
    rx, ry = rankdata(x), rankdata(y)
    A = np.column_stack([np.ones(len(rx))] + [rankdata(z) for z in Z])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]
    n, k = len(rx), len(Z)
    df = n - 2 - k
    tt = r * np.sqrt(df / max(1e-12, 1 - r ** 2))
    p = 2 * tdist.sf(abs(tt), df)
    return r, p



def main():
    rows = []
    confound = []  # (cohort, rho TACSTD2~immune_score) -- the dilution engine
    for cohort, fname in COHORTS.items():
        expr = gene_series(BASE / 'data/processed/gastric' / fname,
                           set(MARKERS) | {'TACSTD2'})
        est_f = PROC / f'{cohort}_estimate.csv'
        if 'TACSTD2' not in expr or not est_f.exists():
            print(f'[{cohort}] missing expression or ESTIMATE - skipped')
            continue
        norm = lambda s: str(s).replace('.', '-')
        est = pd.read_csv(est_f).set_index('sample_id')
        est.index = est.index.map(norm)

        trop2 = expr['TACSTD2']
        trop2.index = trop2.index.map(norm)
        trop2 = restrict_to_tumor(trop2, cohort).dropna()
        common = trop2.index.intersection(est.index)
        if len(common) < 40:
            print(f'[{cohort}] n<40 common tumor samples - skipped')
            continue
        t = trop2.loc[common].astype(float)
        sv = est.loc[common, 'stromal_score'].astype(float)
        iv = est.loc[common, 'immune_score'].astype(float)
        cr, cp = spearmanr(t, iv)
        confound.append((cohort, cr, cp))

        for sym, disp in MARKERS.items():
            base = {'cohort': cohort, 'symbol': sym, 'display': disp,
                    'platform_flag': cohort in PLATFORM_FLAG}
            if sym not in expr:
                rows.append({**base, 'n': 0})
                continue
            g = expr[sym]
            g.index = g.index.map(norm)
            g = restrict_to_tumor(g, cohort).reindex(common).astype(float)
            ok = g.notna() & t.notna()
            if ok.sum() < 40:
                rows.append({**base, 'n': int(ok.sum())})
                continue
            gv, tv = g[ok].values, t[ok].values
            r0, p0 = spearmanr(tv, gv)
            rs, ps = partial_spearman(tv, gv, [sv[ok].values])
            ri, pi = partial_spearman(tv, gv, [iv[ok].values])
            rb, pb = partial_spearman(tv, gv, [sv[ok].values, iv[ok].values])
            rows.append({**base, 'n': int(ok.sum()),
                         'raw_r': r0, 'raw_p': p0,
                         'adj_stroma_r': rs, 'adj_stroma_p': ps,
                         'adj_immune_r': ri, 'adj_immune_p': pi,
                         'adj_both_r': rb, 'adj_both_p': pb})
        print(f'[{cohort}] n={len(common)}  rho(TROP2, immune)={cr:+.2f} (p={cp:.1e})')

    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / 'F3_PDL1_immune_adjusted_stats.csv', index=False)
    ok = stats.dropna(subset=['raw_r'])

    # --- console summary ---
    print('\n=== The confound: rho(TACSTD2, ESTIMATE immune_score) ===')
    for c, r, p in confound:
        print(f'  {c:>10}: {r:+.2f} (p={p:.1e})')
    print('\n=== Median across cohorts (all / excl. GSE84437) ===')
    for sym, disp in MARKERS.items():
        s = ok[ok['symbol'] == sym]
        sf = s[~s['platform_flag']]
        for tag, d in [('all', s), ('excl GSE84437', sf)]:
            if len(d) == 0:
                continue
            n_pos = (d['adj_immune_r'] > 0).sum()
            print(f"  {disp:>16} [{tag:>13}]: raw {d['raw_r'].median():+.3f} -> "
                  f"|stroma {d['adj_stroma_r'].median():+.3f} -> "
                  f"|immune {d['adj_immune_r'].median():+.3f} "
                  f"(positive in {n_pos}/{len(d)}) -> "
                  f"|both {d['adj_both_r'].median():+.3f}")

    # --- figure: 1x2 per-cohort dumbbells, raw -> immune-adjusted ---
    # PDCD1 computed (kept in CSV) but not plotted: too noisy as a single-
    # transcript control; VTCN1 is the decisive one.
    PLOT = ['CD274', 'VTCN1']
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    for ax, sym in zip(axes, PLOT):
        disp = MARKERS[sym]
        s = ok[ok['symbol'] == sym].copy()
        s['cohort_lab'] = s['cohort'] + s['platform_flag'].map({True: ' *', False: ''})
        s = s.sort_values('adj_immune_r')
        y = np.arange(len(s))
        for yi, (_, r) in zip(y, s.iterrows()):
            ax.plot([r['raw_r'], r['adj_immune_r']], [yi, yi],
                    color=PAL['gray'], lw=1.6, zorder=1)
        ax.scatter(s['raw_r'], y, s=52, facecolor='none', edgecolor=PAL['low'],
                   lw=1.4, label='raw', zorder=2)
        sig = s['adj_immune_p'] < 0.05
        ax.scatter(s.loc[sig, 'adj_immune_r'], y[sig.values], s=56, color=PAL['tumor'],
                   label='adj. for immune content', zorder=3)
        ax.scatter(s.loc[~sig, 'adj_immune_r'], y[(~sig).values], s=56, facecolor='none',
                   edgecolor=PAL['tumor'], lw=1.4, zorder=3)
        ax.axvline(0, color='black', lw=0.9, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(s['cohort_lab'], fontsize=8.5)
        med = f"median {s['raw_r'].median():+.2f} → {s['adj_immune_r'].median():+.2f}"
        ax.set_title(f'{disp}\n{med}', fontsize=10, pad=6)
        ax.set_xlabel('Spearman ρ with TROP2', fontsize=9.5)
        clean_ax(ax)
    axes[1].legend(fontsize=8.5, loc='lower right', frameon=False)
    fig.suptitle('PD-L1~TROP2 after removing immune content (filled = adj. p<0.05;  '
                 '* = platform-flagged)', fontsize=11, y=1.02)
    plt.tight_layout()
    savefig(fig, OUT / 'F3_PDL1_immune_adjusted')


if __name__ == '__main__':
    main()

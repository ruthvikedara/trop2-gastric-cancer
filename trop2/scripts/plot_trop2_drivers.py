"""Genomic and epigenomic correlates of TROP2 expression."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, mannwhitneyu, fisher_exact, rankdata, t as tdist

from cohort_utils import restrict_to_tumor
from style import PAL, clean_ax, savefig, fmt_p, apply_pub_style
from genolib import PROJECT_ROOT

BASE = PROJECT_ROOT
CBIOP = BASE / 'data/external/cbioportal/stad_tcga_pan_can_atlas_2018'
OUT = Path(__file__).resolve().parent.parent / 'plots' / 'drivers'
OUT.mkdir(parents=True, exist_ok=True)

TROP2 = 'TACSTD2'
ADC_COMPARATORS = ['ERBB2', 'CLDN18']          # the rest of the ADC trio (F2)
DRIVERS = ['TP53', 'ARID1A', 'PIK3CA', 'RHOA', 'CDH1']
DRIVER_TAG = {'TP53': 'CIN', 'ARID1A': 'MSI', 'PIK3CA': '-',
              'RHOA': 'GS', 'CDH1': 'GS'}
EPITHELIAL_TFS = ['ELF3', 'GRHL2', 'KLF5', 'TP63']   # literature TACSTD2 links
EMT_TFS = ['ZEB2', 'SNAI2']                          # negative controls
# (ZEB2 not ZEB1: ZEB1 is absent from the Affymetrix ACRG platforms' gene lists)

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

# mc3 nonsynonymous classes (whitelist)
NONSYN_KEEP = {'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
               'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Splice_Site',
               'Nonstop_Mutation', 'Translation_Start_Site'}

GISTIC_GROUP = {-2: 'Loss', -1: 'Loss', 0: 'Neutral', 1: 'Gain/Amp', 2: 'Gain/Amp'}
GROUP_ORDER = ['Loss', 'Neutral', 'Gain/Amp']
GROUP_COLOR = {'Loss': PAL['low'], 'Neutral': PAL['gray'], 'Gain/Amp': PAL['high']}



# loaders
def load_gene_rows(path, genes):
    """Line-scan a gene x sample matrix, pulling only the requested gene rows.
    Returns DataFrame (samples x genes), sample IDs truncated to 12 chars."""
    want = set(genes)
    found = {}
    with open(path) as f:
        samples = f.readline().rstrip('\n').split('\t')[1:]
        for line in f:
            tab = line.index('\t')
            g = line[:tab]
            if g in want:
                found[g] = [float(v) if v.strip() not in ('', 'NA') else np.nan
                            for v in line[tab + 1:].rstrip('\n').split('\t')]
                if len(found) == len(want):
                    break
    missing = want - set(found)
    if missing:
        print(f'  [warn] {Path(path).name}: genes not found: {sorted(missing)}')
    df = pd.DataFrame(found, index=[s[:12] for s in samples])
    return df[~df.index.duplicated(keep='first')]


def load_cna_genes(genes):
    """Thresholded GISTIC2 calls (-2..2) for the requested genes.
    Tumor samples only (sample code -01); IDs truncated to 12 chars."""
    keep_cols, sample_ids = [], []
    with open(CBIOP / 'data_cna.txt') as f:
        header = f.readline().rstrip('\n').split('\t')
        for i, s in enumerate(header):
            if i >= 3 and s[13:15] == '01':      # primary tumor aliquot
                keep_cols.append(i)
                sample_ids.append(s[:12])
        rows = {}
        want = set(genes)
        for line in f:
            tab = line.index('\t')
            g = line[:tab]
            if g in want:
                vals = line.rstrip('\n').split('\t')
                rows[g] = [float(vals[i]) if vals[i].strip() not in ('', 'NA')
                           else np.nan for i in keep_cols]
                if len(rows) == len(want):
                    break
    df = pd.DataFrame(rows, index=sample_ids)
    return df[~df.index.duplicated(keep='first')]


def load_methylation_probes(gene):
    """Probe-level beta values for one gene from the HM27/HM450 merged file.
    Returns DataFrame (tumor samples x probes), columns 'cgXXXX (TSS200)'."""
    rows, row_meta = [], []
    with open(CBIOP / 'data_methylation_hm27_hm450_merged.txt') as f:
        header = f.readline().rstrip('\n').split('\t')
        raw_ids = header[4:]
        is_tumor = [s[13:15] == '01' for s in raw_ids]
        sample_ids = [s[:12] for s in raw_ids]
        for line in f:
            parts = line.split('\t', 4)
            if parts[1] == gene:
                vals = [float(v) if v.strip() not in ('', 'NA') else np.nan
                        for v in parts[4].rstrip('\n').split('\t')]
                rows.append(vals)
                row_meta.append(f'{parts[0]} ({parts[2]})')
    df = pd.DataFrame(np.array(rows).T, columns=row_meta, index=sample_ids)
    df = df.loc[is_tumor]
    return df[~df.index.duplicated(keep='first')]


def load_mutations():
    """mc3 MAF restricted to driver genes + TACSTD2 (chunked read of 265MB).
    Returns (hits DF, sequenced patient set) - sequenced = WES case list, which
    defines the WT denominator."""
    hits = []
    usecols = ['Hugo_Symbol', 'Variant_Classification', 'Tumor_Sample_Barcode']
    for chunk in pd.read_csv(CBIOP / 'data_mutations.txt', sep='\t',
                             usecols=usecols, chunksize=500_000,
                             low_memory=False):
        sub = chunk[chunk['Hugo_Symbol'].isin(DRIVERS + [TROP2])]
        hits.append(sub)
    maf = pd.concat(hits, ignore_index=True)
    maf['patient'] = maf['Tumor_Sample_Barcode'].str[:12]
    sequenced = set()
    with open(CBIOP / 'case_lists' / 'cases_sequenced.txt') as f:
        for line in f:
            if line.startswith('case_list_ids:'):
                sequenced = {s[:12] for s in line.split(':', 1)[1].split('\t')}
    return maf, sequenced


def load_aneuploidy():
    clin = pd.read_csv(CBIOP / 'data_clinical_sample.txt', sep='\t',
                       skiprows=4, usecols=['SAMPLE_ID', 'ANEUPLOIDY_SCORE'])
    clin['patient'] = clin['SAMPLE_ID'].str[:12]
    clin['ANEUPLOIDY_SCORE'] = pd.to_numeric(clin['ANEUPLOIDY_SCORE'],
                                             errors='coerce')
    return (clin.dropna().drop_duplicates('patient')
                .set_index('patient')['ANEUPLOIDY_SCORE'])


def load_estimate_stroma(cohort):
    """ESTIMATE stromal score per sample (run_estimate.R outputs). Purity
    confound control - same engine as F2/F3 adjusted analyses."""
    f = BASE / 'data/clinical/gastric/processed' / f'{cohort}_estimate.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return df.drop_duplicates('sample_id').set_index('sample_id')['stromal_score']


def partial_spearman(x, y, Z):
    """Partial Spearman r(x, y | Z): rank both, residualize on ranked Z (OLS),
    Pearson on residuals. Same convention as plot_f3_pdl1_adjusted.py."""
    rx, ry = rankdata(x), rankdata(y)
    A = np.column_stack([np.ones(len(rx))] + [rankdata(z) for z in Z])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]
    df = len(rx) - 2 - len(Z)
    tt = r * np.sqrt(df / max(1e-12, 1 - r ** 2))
    p = 2 * tdist.sf(abs(tt), df)
    return r, p


# per-layer analyses
def cna_dose(expr, cna):
    """(a) Expression by GISTIC group + Spearman(CN, expression)."""
    df = pd.concat([expr[TROP2].rename('expr'), cna[TROP2].rename('cn')],
                   axis=1).dropna()
    df['grp'] = df['cn'].map(GISTIC_GROUP)
    rho, p = spearmanr(df['cn'], df['expr'])
    groups = {g: df.loc[df['grp'] == g, 'expr'].values for g in GROUP_ORDER}
    _, p_gain = mannwhitneyu(groups['Gain/Amp'], groups['Neutral'],
                             alternative='two-sided')
    _, p_loss = mannwhitneyu(groups['Loss'], groups['Neutral'],
                             alternative='two-sided')
    stats = [{'layer': 'CNA', 'group': g, 'n': len(groups[g]),
              'median_expr': float(np.median(groups[g]))} for g in GROUP_ORDER]
    stats += [{'layer': 'CNA', 'group': 'spearman_cn_expr', 'n': len(df),
               'median_expr': rho, 'p': p},
              {'layer': 'CNA', 'group': 'MW_gainamp_vs_neutral', 'n': len(df),
               'median_expr': np.nan, 'p': p_gain},
              {'layer': 'CNA', 'group': 'MW_loss_vs_neutral', 'n': len(df),
               'median_expr': np.nan, 'p': p_loss}]
    print(f'(a) CNA dose: n={len(df)} | Spearman rho={rho:+.2f} ({fmt_p(p)}) | '
          f'Gain/Amp vs Neutral {fmt_p(p_gain)} | Loss vs Neutral {fmt_p(p_loss)}')
    print('    group n / median expr: ' + '; '.join(
        f"{g}: n={len(groups[g])}, {np.median(groups[g]):.2f}" for g in GROUP_ORDER))
    return df, stats


def high_tail(expr, cna):
    """(b) Fraction of top-quartile expressors carrying GISTIC gain/amp,
    TACSTD2 vs ERBB2 vs CLDN18 (amplicon positive control)."""
    rows = []
    for gene in [TROP2] + ADC_COMPARATORS:
        df = pd.concat([expr[gene].rename('expr'), cna[gene].rename('cn')],
                       axis=1).dropna()
        q_hi, q_lo = df['expr'].quantile(0.75), df['expr'].quantile(0.25)
        top = df[df['expr'] >= q_hi]
        bot = df[df['expr'] <= q_lo]
        ga_top = (top['cn'] >= 1)
        ga_bot = (bot['cn'] >= 1)
        or_, fp = fisher_exact([[ga_top.sum(), (~ga_top).sum()],
                                [ga_bot.sum(), (~ga_bot).sum()]])
        rho, rp = spearmanr(df['cn'], df['expr'])
        rows.append({'gene': gene, 'n': len(df),
                     'frac_gainamp_top_q': ga_top.mean(),
                     'frac_gainamp_bot_q': ga_bot.mean(),
                     'fisher_or': or_, 'fisher_p': fp,
                     'spearman_cn_expr': rho, 'spearman_p': rp})
        print(f'(b) {gene}: gain/amp in top-quartile {ga_top.mean():.0%} vs '
              f'bottom-quartile {ga_bot.mean():.0%} (OR={or_:.2f}, {fmt_p(fp)}); '
              f'CN~expr rho={rho:+.2f}')
    return pd.DataFrame(rows)


def driver_mutations(expr, maf, sequenced):
    """(c) Expression by nonsynonymous driver-mutation status (trans)."""
    nonsyn = maf[maf['Variant_Classification'].isin(NONSYN_KEEP)]
    seq_expr = sorted(set(expr.index) & sequenced)
    e = expr.loc[seq_expr, TROP2]
    rows, wt_vals, mut_vals = [], {}, {}
    for gene in DRIVERS:
        mut = set(nonsyn.loc[nonsyn['Hugo_Symbol'] == gene, 'patient'])
        mut &= set(seq_expr)
        wt = set(seq_expr) - mut
        if len(mut) < 8:
            print(f'(c) {gene}: only {len(mut)} mutated - skipped')
            continue
        d_mut, d_wt = e.loc[sorted(mut)].values, e.loc[sorted(wt)].values
        wt_vals[gene], mut_vals[gene] = d_wt, d_mut
        _, p = mannwhitneyu(d_mut, d_wt, alternative='two-sided')
        rows.append({'gene': gene, 'subtype_marks': DRIVER_TAG[gene],
                     'n_mut': len(mut), 'n_wt': len(wt),
                     'median_mut': float(np.median(d_mut)),
                     'median_wt': float(np.median(d_wt)),
                     'delta_mut_minus_wt': float(np.median(d_mut) - np.median(d_wt)),
                     'mw_p': p})
        print(f'(c) {gene} ({DRIVER_TAG[gene]}): mut n={len(mut)}, '
              f'delta={np.median(d_mut) - np.median(d_wt):+.2f}, {fmt_p(p)}')
    n_tac_mut = nonsyn.loc[nonsyn['Hugo_Symbol'] == TROP2, 'patient'].nunique()
    print(f'(c) TACSTD2 itself: {n_tac_mut} nonsynonymous-mutated of '
          f'{len(seq_expr)} sequenced - the target is essentially never mutated')
    stats = pd.DataFrame(rows)
    stats.attrs['tacstd2_nonsyn_n'] = n_tac_mut
    stats.attrs['n_sequenced_expr'] = len(seq_expr)
    stats.attrs['wt_vals'] = wt_vals
    stats.attrs['mut_vals'] = mut_vals
    return stats


def promoter_methylation(expr, meth):
    """(d) Spearman(promoter beta, expression) per probe, raw and
    ESTIMATE-stroma-adjusted (methylation~expression could otherwise just
    read tumor purity)."""
    stroma = load_estimate_stroma('TCGA-STAD')
    df = pd.concat([expr[TROP2].rename('expr'), meth,
                    stroma.rename('stroma')], axis=1).dropna()
    rows = []
    for probe in meth.columns:
        rho, p = spearmanr(df[probe], df['expr'])
        ra, pa = partial_spearman(df[probe].values, df['expr'].values,
                                  [df['stroma'].values])
        rows.append({'probe': probe, 'n': len(df),
                     'spearman_rho': rho, 'spearman_p': p,
                     'stroma_adj_rho': ra, 'stroma_adj_p': pa,
                     'median_beta': float(df[probe].median())})
        print(f'(d) {probe}: n={len(df)}, rho={rho:+.2f} ({fmt_p(p)}) '
              f'-> stroma-adj {ra:+.2f} ({fmt_p(pa)})')
    return df, pd.DataFrame(rows)


def tf_coexpression():
    """(e) Spearman(TACSTD2, TF) per cohort for epithelial TFs + EMT controls,
    raw and ESTIMATE-stroma-adjusted (epithelial genes co-correlate with tumor
    purity in bulk - the adjusted value is the transcriptional co-regulation
    signal, same engine as F2 epithelial-adjusted)."""
    genes = EPITHELIAL_TFS + EMT_TFS
    rows = []
    for cohort, rel in COHORTS.items():
        df = load_gene_rows(BASE / rel, [TROP2] + genes)
        df = restrict_to_tumor(df, cohort)
        if TROP2 not in df.columns:
            continue
        stroma = load_estimate_stroma(cohort)
        if stroma is not None:
            df = df.join(stroma.rename('stroma'), how='left')
        df = df.dropna(subset=[TROP2])
        if len(df) < 20:
            print(f'(e) [{cohort}] n<20 - skipped')
            continue
        for gene in genes:
            if gene not in df.columns:
                continue
            cols = [TROP2, gene] + (['stroma'] if 'stroma' in df.columns else [])
            sub = df[cols].dropna()
            rho, p = spearmanr(sub[TROP2], sub[gene])
            ra, pa = (partial_spearman(sub[gene].values, sub[TROP2].values,
                                       [sub['stroma'].values])
                      if 'stroma' in sub.columns else (np.nan, np.nan))
            rows.append({'gene': gene, 'cohort': cohort, 'n': len(sub),
                         'spearman_rho': rho, 'spearman_p': p,
                         'stroma_adj_rho': ra, 'stroma_adj_p': pa,
                         'class': 'epithelial TF' if gene in EPITHELIAL_TFS
                                  else 'EMT TF (control)'})
    stats = pd.DataFrame(rows)
    for gene in genes:
        g = stats[stats['gene'] == gene]
        npos = int((g['spearman_rho'] > 0).sum())
        nsig = int(((g['spearman_p'] < 0.05) & (g['spearman_rho'] > 0)).sum())
        print(f"(e) {gene}: mean rho={g['spearman_rho'].mean():+.2f} "
              f"(stroma-adj {g['stroma_adj_rho'].mean():+.2f}), "
              f"positive {npos}/{len(g)} cohorts, sig-positive {nsig}/{len(g)}")
    return stats


def variance_explained(expr, cna, meth, maf, sequenced, aneuploidy):
    """(f) Joint OLS expression ~ CNA + promoter methylation + TP53 status;
    delta-R2 per layer. Also Spearman(expression, aneuploidy score)."""
    nonsyn = maf[maf['Variant_Classification'].isin(NONSYN_KEEP)]
    tp53_mut = set(nonsyn.loc[nonsyn['Hugo_Symbol'] == 'TP53', 'patient'])
    probes = list(meth.columns)
    df = pd.concat([expr[TROP2].rename('expr'), cna[TROP2].rename('cn'), meth],
                   axis=1)
    df = df[df.index.isin(sequenced)]
    df['tp53'] = df.index.isin(tp53_mut).astype(float)
    df = df.dropna()

    def r2(cols):
        A = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
        coef, *_ = np.linalg.lstsq(A, df['expr'].values, rcond=None)
        resid = df['expr'].values - A @ coef
        return 1 - (resid ** 2).sum() / ((df['expr'] - df['expr'].mean()) ** 2).sum()

    full = r2(['cn'] + probes + ['tp53'])
    d_cn = full - r2(probes + ['tp53'])
    d_meth = full - r2(['cn', 'tp53'])
    d_tp53 = full - r2(['cn'] + probes)
    an = aneuploidy.reindex(df.index).dropna()
    rho_an, p_an = spearmanr(df.loc[an.index, 'expr'], an.values)
    stats = pd.DataFrame([
        {'term': 'CNA (cis-genetic)', 'delta_r2': d_cn},
        {'term': 'Promoter methylation (epigenetic)', 'delta_r2': d_meth},
        {'term': 'TP53 mutation (trans-genetic)', 'delta_r2': d_tp53},
        {'term': 'FULL MODEL', 'delta_r2': full},
        {'term': 'aneuploidy_score_vs_expr (Spearman rho, separate)', 'delta_r2': rho_an},
    ])
    print(f'(f) n={len(df)} | full-model R2={full:.3f} | dR2 CNA={d_cn:.3f}, '
          f'methylation={d_meth:.3f}, TP53={d_tp53:.3f} | '
          f'aneuploidy rho={rho_an:+.2f} ({fmt_p(p_an)})')
    stats.attrs['aneuploidy_p'] = p_an
    stats.attrs['n'] = len(df)
    return stats, rho_an, p_an


# figure
def _box(ax, data, pos, color, width=0.55):
    bp = ax.boxplot(data, positions=[pos], widths=width, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', lw=1.4),
                    whiskerprops=dict(color=color), capprops=dict(color=color))
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(color)
    jitter = np.random.default_rng(7).uniform(-0.13, 0.13, size=len(data))
    ax.scatter(np.full(len(data), pos) + jitter, data, s=4, color=color,
               alpha=0.25, linewidths=0, zorder=1)


def panel_cna(ax, cna_df):
    groups = [cna_df.loc[cna_df['grp'] == g, 'expr'].values for g in GROUP_ORDER]
    for i, (g, vals) in enumerate(zip(GROUP_ORDER, groups)):
        _box(ax, vals, i, GROUP_COLOR[g])
        ax.text(i, ax.get_ylim()[0], '', fontsize=7)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([f'{g}\n(n={len(v)})' for g, v in zip(GROUP_ORDER, groups)],
                       fontsize=9)
    rho, p = spearmanr(cna_df['cn'], cna_df['expr'])
    _, p_gain = mannwhitneyu(groups[2], groups[1])
    _, p_loss = mannwhitneyu(groups[0], groups[1])
    ax.set_ylabel('TACSTD2 expression (log2)', fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.set_title('(a) Copy number (GISTIC2, 1p32.1)\n'
                 f'CN~expr ρ={rho:+.2f} (n.s.); gain vs neutral {fmt_p(p_gain)}',
                 fontsize=10, loc='left')
    clean_ax(ax)


def panel_hightail(ax, ht):
    x = np.arange(len(ht))
    w = 0.36
    ax.bar(x - w / 2, ht['frac_gainamp_top_q'], w, color=PAL['high'],
           alpha=0.85, label='top-quartile expressors')
    ax.bar(x + w / 2, ht['frac_gainamp_bot_q'], w, color=PAL['low'],
           alpha=0.85, label='bottom-quartile expressors')
    for xi, r in ht.iterrows():
        ax.text(xi - w / 2, r['frac_gainamp_top_q'] + 0.02,
                f"{r['frac_gainamp_top_q']:.0%}", ha='center', fontsize=9)
        ax.text(xi + w / 2, r['frac_gainamp_bot_q'] + 0.02,
                f"{r['frac_gainamp_bot_q']:.0%}", ha='center', fontsize=9)
        top_y = max(r['frac_gainamp_top_q'], r['frac_gainamp_bot_q']) + 0.085
        ax.text(xi, top_y, fmt_p(r['fisher_p']), ha='center', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['TROP2\n(TACSTD2)', 'HER2\n(ERBB2)', 'CLDN18'],
                       fontsize=9)
    ax.set_ylim(0, max(ht['frac_gainamp_top_q'].max() * 1.4, 0.2))
    ax.set_ylabel('Fraction with CN gain/amp', fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=7.8, frameon=False, loc='upper left',
              bbox_to_anchor=(0.0, 0.99), ncol=2, handlelength=1.4,
              columnspacing=0.9, labelspacing=0.2, borderaxespad=0.0)
    ax.set_title('(b) CN gain/amp in top vs bottom expressors\n'
                 'ERBB2 = amplicon-driven positive control',
                 fontsize=10, loc='left')
    clean_ax(ax)


def panel_mutations(ax, mut_stats):
    for i, r in mut_stats.reset_index().iterrows():
        wt = mut_stats.attrs['wt_vals'][r['gene']]
        mut = mut_stats.attrs['mut_vals'][r['gene']]
        _box(ax, wt, i - 0.19, PAL['gray'], width=0.32)
        _box(ax, mut, i + 0.19, PAL['tumor'], width=0.32)
        ax.text(i, ax.get_ylim()[1] * 0.98, fmt_p(r['mw_p']), ha='center',
                va='top', fontsize=8.5)
    n_seq = mut_stats.attrs['n_sequenced_expr']
    n_tac = mut_stats.attrs['tacstd2_nonsyn_n']
    ax.set_xticks(range(len(mut_stats)))
    ax.set_xticklabels(
        [f"{r['gene']}\n({r['subtype_marks']})\nn={r['n_mut']}"
         for _, r in mut_stats.iterrows()], fontsize=7.7, linespacing=0.95)
    ax.set_ylabel('TACSTD2 expression (log2)', fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.set_title('(c) Driver mutations (trans)\n'
                 f'gray=WT, red=mutated · TACSTD2 itself mutated in '
                 f'{n_tac}/{n_seq} WES cases', fontsize=10, loc='left')
    clean_ax(ax)


def panel_methylation(axes, meth_df, meth_stats):
    for k, (ax, (_, r)) in enumerate(zip(axes, meth_stats.iterrows())):
        probe = r['probe']
        x, y = meth_df[probe], meth_df['expr']
        ax.scatter(x, y, s=5, color=PAL['low'], alpha=0.3, linewidths=0)
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), color=PAL['tumor'], lw=1.2)
        ax.set_xlabel(f'{probe} β', fontsize=9)
        if k == 0:
            ax.set_ylabel('TACSTD2 expr (log2)', fontsize=9.5)
        # Keep statistics out of the data cloud and clear of the y-axis.
        ax.text(0.5, 1.025,
                f'raw ρ={r["spearman_rho"]:+.2f}, '
                f'{fmt_p(r["spearman_p"])}\n'
                f'adjusted ρ={r["stroma_adj_rho"]:+.2f}, '
                f'{fmt_p(r["stroma_adj_p"])}',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=7.2, linespacing=1.1)
        if k == 0:
            ax.text(-0.02, 1.20, '(d) Promoter methylation vs expression',
                    transform=ax.transAxes, fontsize=10, ha='left',
                    va='bottom')
        ax.tick_params(labelsize=8.5)
        clean_ax(ax)


def panel_tf_heatmap(ax, tf_stats, fig):
    genes = EPITHELIAL_TFS + EMT_TFS
    cohorts = [c for c in COHORTS if c in set(tf_stats['cohort'])]
    rho = (tf_stats.pivot(index='gene', columns='cohort',
                          values='spearman_rho')
           .reindex(index=genes, columns=cohorts))
    pval = (tf_stats.pivot(index='gene', columns='cohort',
                           values='spearman_p')
            .reindex(index=genes, columns=cohorts))
    order = rho.mean(axis=1).sort_values(ascending=False).index
    rho, pval = rho.loc[order], pval.loc[order]
    vmax = 0.65
    # aspect='equal' -> square cells: same ρ per inch on both axes, and the
    # panel no longer hogs horizontal real estate
    im = ax.imshow(rho.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='equal')
    ax.set_anchor('W')
    for i, g in enumerate(rho.index):
        for j, c in enumerate(rho.columns):
            r, p = rho.loc[g, c], pval.loc[g, c]
            if np.isnan(r):
                ax.text(j, i, ' - ', ha='center', va='center', fontsize=8,
                        color='#888888')
                continue
            txt = f'{r:.2f}' + ('*' if p < 0.05 else '')
            ax.text(j, i, txt, ha='center', va='center', fontsize=7.2,
                    color='white' if abs(r) > 0.45 else 'black')
    ax.set_xticks(range(len(rho.columns)))
    ax.set_xticklabels(rho.columns, rotation=45, ha='right', fontsize=8.5)
    ax.set_yticks(range(len(rho.index)))
    labels = []
    for g in rho.index:
        m = rho.loc[g].mean()
        labels.append(f'{g} ({m:+.2f})')
    ax.set_yticklabels(labels, fontsize=9)
    for tick, g in zip(ax.get_yticklabels(), rho.index):
        cls = tf_stats.loc[tf_stats['gene'] == g, 'class'].iloc[0]
        tick.set_color(PAL['ep'] if cls == 'epithelial TF' else PAL['mp'])
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Spearman ρ', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title('(e) Lineage TF program - ρ(TACSTD2, TF), 9 cohorts\n'
                 'blue = epithelial TFs · red = EMT controls · * p<0.05 · '
                 'row label = mean ρ', fontsize=10, loc='left')


def panel_variance(ax, var_stats, rho_an, p_an):
    d = var_stats.iloc[:3]
    colors = [PAL['low'], PAL['gold'], PAL['gray']]
    values = (d['delta_r2'] * 100).to_numpy()
    ypos = np.arange(3)
    ax.barh(ypos, values, color=colors, alpha=0.85, height=0.58)
    for y, v in zip(ypos, values):
        ax.text(v + 0.25, y, f'{v:.1f}%', va='center', ha='left',
                fontsize=9, fontweight='bold' if v > 1 else 'normal')
    ax.set_yticks(ypos)
    ax.set_yticklabels(['CN', 'Methylation', 'TP53'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Unique variance explained, ΔR² (%)', fontsize=9.5)
    full = var_stats.loc[var_stats['term'] == 'FULL MODEL', 'delta_r2'].iloc[0]
    ax.text(0.62, 0.06,
            f'Joint model R² = {full:.2f}\n'
            f'Aneuploidy ρ = {rho_an:+.2f}\n{fmt_p(p_an)}',
            transform=ax.transAxes, fontsize=8.3, va='bottom', ha='left',
            linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#F7F7F7',
                      edgecolor='#D0D0D0', linewidth=0.8))
    ax.set_title('(f) Variance explained', fontsize=10, loc='left')
    ax.set_xlim(0, values.max() * 2.25)
    ax.tick_params(labelsize=8.5)
    clean_ax(ax)


def main():
    apply_pub_style()
    print('Loading expression (TCGA) ...')
    expr = load_gene_rows(BASE / 'data/processed/gastric/TCGA_STAD_for_xcell.txt',
                          [TROP2] + ADC_COMPARATORS)
    print(f'  {expr.shape[0]} tumors')
    print('Loading GISTIC2 CNA ...')
    cna = load_cna_genes([TROP2] + ADC_COMPARATORS)
    print('Loading promoter methylation (TACSTD2 probes) ...')
    meth = load_methylation_probes(TROP2)
    print(f'  probes: {list(meth.columns)}; {meth.shape[0]} tumor samples')
    print('Loading mc3 MAF (driver genes) ...')
    maf, sequenced = load_mutations()
    print(f'  {len(sequenced)} WES-sequenced patients; '
          f'{len(maf)} driver-gene variant rows')
    aneuploidy = load_aneuploidy()

    cna_df, cna_stats = cna_dose(expr, cna)
    ht_stats = high_tail(expr, cna)
    mut_stats = driver_mutations(expr, maf, sequenced)
    meth_df, meth_stats = promoter_methylation(expr, meth)
    tf_stats = tf_coexpression()
    var_stats, rho_an, p_an = variance_explained(expr, cna, meth, maf,
                                                 sequenced, aneuploidy)

    pd.DataFrame(cna_stats).to_csv(OUT / 'drivers_cna_stats.csv', index=False)
    ht_stats.to_csv(OUT / 'drivers_hightail_stats.csv', index=False)
    mut_stats.to_csv(OUT / 'drivers_mutation_stats.csv', index=False)
    meth_stats.to_csv(OUT / 'drivers_methylation_stats.csv', index=False)
    tf_stats.to_csv(OUT / 'drivers_tf_coexpr_stats.csv', index=False)
    var_stats.to_csv(OUT / 'drivers_variance_explained.csv', index=False)

    # ---- figure: 2 rows; top = 3 equal panels; bottom weighted so the
    # TF heatmap gets just the width its square cells need ----
    fig = plt.figure(figsize=(14, 7.2))
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.52, wspace=0.60,
                           left=0.055, right=0.975, top=0.865, bottom=0.11)
    top = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[0, :],
        width_ratios=[1.0, 1.0, 1.16], wspace=0.50)
    panel_cna(fig.add_subplot(top[0]), cna_df)
    panel_hightail(fig.add_subplot(top[1]), ht_stats)
    panel_mutations(fig.add_subplot(top[2]), mut_stats)
    bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :],
                                           width_ratios=[2.50, 2.90, 1.45],
                                           wspace=0.48)
    sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=bot[0],
                                           wspace=0.45)
    panel_methylation([fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])],
                      meth_df, meth_stats)
    ax_e = fig.add_subplot(bot[1])
    panel_tf_heatmap(ax_e, tf_stats, fig)
    panel_variance(fig.add_subplot(bot[2]), var_stats, rho_an, p_an)
    fig.suptitle('Genetic and epigenetic drivers of TROP2 (TACSTD2)',
                 fontsize=13, fontweight='bold', y=0.985)
    fig.text(0.5, 0.947, 'TCGA-STAD multi-omics · lineage-TF replication '
             'across 9 cohorts', ha='center', fontsize=9.5, color='#444444')
    savefig(fig, OUT / 'TROP2_genomic_epigenomic_drivers')
    print(f'  outputs in {OUT}')


if __name__ == '__main__':
    main()

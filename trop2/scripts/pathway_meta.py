"""Signature scoring and random-effects meta-analysis."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, t as t_dist

from genolib import PROJECT_ROOT


BASE = PROJECT_ROOT
RESOLVED_JSON = (
    BASE / 'lab' / 'pathway_sets' / 'pathway_sets_resolved_2026-07-31.json'
)
TROP2 = 'TACSTD2'

# The 9 tumor-only gastric cohorts retained for the F2D analysis after the
# revision-wide removal of GSE26253.
COHORTS = {
    'TCGA-STAD': 'data/processed/gastric/TCGA_STAD_for_xcell.txt',
    'GSE66229': 'data/processed/gastric/ACRG_GSE66229_for_xcell.txt',
    'GSE15459': 'data/processed/gastric/ACRG_GSE15459_for_xcell.txt',
    'GSE34942': 'data/processed/gastric/ACRG_GSE34942_for_xcell.txt',
    'GSE35809': 'data/processed/gastric/ACRG_GSE35809_for_xcell.txt',
    'GSE51105': 'data/processed/gastric/ACRG_GSE51105_for_xcell.txt',
    'GSE54129': 'data/processed/gastric/ACRG_GSE54129_for_xcell.txt',
    'GSE57303': 'data/processed/gastric/ACRG_GSE57303_for_xcell.txt',
    'GSE84437': 'data/processed/gastric/ACRG_GSE84437_for_xcell.txt',
}

SECTION_ABBR = {
    'IDO1 upstream/downstream mechanism pathways': 'IDO1',
    'CLDN18 / epithelial-junction mechanism pathways': 'Epithelial junction',
    'Broader oncogenic signaling pathways': 'Oncogenic',
    'Broader immune pathway context': 'Immune',
    'Metabolic pathway context': 'Metabolic',
}

MIN_GENES = 2          # at least two measured genes
MIN_COVERAGE = 0.50    # at least 50% of the signature measured


def load_signatures():
    """Ordered (name -> genes) and (name -> section), TACSTD2 removed.

    Returns ``(order, genes_by, section_of, dropped_trop2)`` where
    ``dropped_trop2`` lists any signature that contained TACSTD2 (circularity
    guard; expected to be empty).
    """
    resolved = json.load(open(RESOLVED_JSON))
    order, genes_by, section_of, dropped = [], {}, {}, []
    for section, entries in resolved.items():
        abbr = SECTION_ABBR[section]
        for entry in entries:
            name = entry['name']
            genes = list(dict.fromkeys(entry['genes']))
            if TROP2 in genes:
                dropped.append(name)
                genes = [g for g in genes if g != TROP2]
            order.append(name)
            genes_by[name] = genes
            section_of[name] = abbr
    return order, genes_by, section_of, dropped


def load_signatures_with_rbsig():
    """Return the curated 32 sets plus the published 87-gene RB-loss set."""
    from plot_f2_proliferation_rb_pathways import RBSIG

    order, genes_by, section_of, dropped = load_signatures()
    name = 'RB loss-of-function (RBsig)'
    genes_by[name] = list(dict.fromkeys(g for g in RBSIG if g != TROP2))
    section_of[name] = 'Oncogenic'
    order.insert(order.index('Cell cycle') + 1, name)
    return order, genes_by, section_of, dropped


def complete_expression_and_sets(expression, order, genes_by):
    """Prepare complete rank-score inputs and enforce signature coverage.

    ssGSEA and GSVA require a complete gene-by-sample matrix. Genes containing
    any missing value within a cohort are removed rather than zero-imputed.
    Coverage is then evaluated against each signature's original gene count.
    """
    complete = expression.loc[:, ~expression.isna().any(axis=0)].copy()
    usable, coverage = {}, []
    for pathway in order:
        genes = genes_by[pathway]
        present = [g for g in genes if g in complete.columns]
        fraction = len(present) / len(genes) if genes else 0.0
        coverage.append({
            'pathway': pathway,
            'genes_total': len(genes),
            'genes_measured': len(present),
            'coverage_pct': 100 * fraction,
        })
        if len(present) >= MIN_GENES and fraction >= MIN_COVERAGE:
            usable[pathway] = present
    return complete, usable, pd.DataFrame(coverage)


def rank_signature_scores(expression, gene_sets, method):
    """Compute ssGSEA or GSVA scores for one cohort with gseapy."""
    import warnings
    import gseapy

    kwargs = dict(
        data=expression.T,
        gene_sets={k: list(v) for k, v in gene_sets.items()},
        outdir=None,
        min_size=1,
        max_size=100000,
        threads=4,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if method == 'ssgsea':
            result = gseapy.ssgsea(
                sample_norm_method='rank', correl_norm_type='rank',
                weight=0.25, ascending=False, **kwargs)
            value_col = 'NES'
        elif method == 'gsva':
            result = gseapy.gsva(kcdf='Gaussian', **kwargs)
            value_col = 'ES'
        else:
            raise ValueError(f'unsupported rank-scoring method: {method}')
    return (result.res2d.pivot(index='Term', columns='Name', values=value_col)
            .astype(float))


def read_expression(path):
    data = pd.read_csv(path, sep='\t', index_col=0)
    data = data.apply(pd.to_numeric, errors='coerce')
    return data.loc[~data.index.duplicated()]


def cohort_matrix(cohort, rel):
    """Tumor-only samples x genes, plus the gene-wise within-cohort z-matrix."""
    from cohort_utils import restrict_to_tumor
    expression = read_expression(BASE / rel).T
    expression = restrict_to_tumor(expression, cohort)
    expression = expression.dropna(axis=1, how='all')
    means = expression.mean(axis=0)
    sds = expression.std(axis=0, ddof=0).replace(0, np.nan)
    return expression, (expression - means) / sds


def signature_score(z, genes):
    """Mean-z score + coverage bookkeeping for one signature in one cohort.

    Returns ``(score_or_None, info)``. ``score`` is None when the signature
    fails the >=2 measured genes / >=50% coverage rule.
    """
    total = len(genes)
    present = [g for g in genes if g in z.columns]
    coverage = len(present) / total if total else 0.0
    info = {'genes_total': total, 'genes_measured': len(present),
            'coverage_pct': 100 * coverage}
    if len(present) < MIN_GENES or coverage < MIN_COVERAGE:
        info['excluded'] = True
        return None, info
    info['excluded'] = False
    return z[present].mean(axis=1), info


# Random-effects meta-analysis

def paule_mandel_tau2(effects, variances, max_tau2=1e4):
    """Paule-Mandel estimate of between-study variance.

    Solves sum_i w_i (y_i - mu_w)^2 = k - 1 with w_i = 1/(v_i + tau2) by
    bracketed root-finding. The generalized-Q statistic is strictly decreasing
    in tau2, so a bisection/Brent solve is unconditionally stable; an earlier
    Newton iteration oscillated and returned tau2 = 0 for two highly
    heterogeneous signatures (T cell receptor signaling, Hypoxia), which is
    impossible when Q >> k-1.
    """
    from scipy.optimize import brentq
    effects = np.asarray(effects, float)
    variances = np.asarray(variances, float)
    k = len(effects)
    if k < 2:
        return 0.0

    def gen_q(tau2):
        w = 1.0 / (variances + tau2)
        mu = np.sum(w * effects) / np.sum(w)
        return np.sum(w * (effects - mu) ** 2) - (k - 1)

    if gen_q(0.0) <= 0:
        return 0.0
    hi = max(np.var(effects, ddof=1), np.mean(variances), 1e-8)
    for _ in range(80):
        if gen_q(hi) < 0:
            break
        hi *= 2
        if hi > max_tau2:
            return float(max_tau2)
    return float(brentq(gen_q, 0.0, hi, xtol=1e-14, rtol=1e-12))


def meta_hksj(effects, variances):
    """Paule-Mandel random effects with Hartung-Knapp-Sidik-Jonkman inference.

    Returns the pooled effect on the input scale together with the HKSJ
    standard error, t-based 95% limits (df = k - 1), the HKSJ p-value, and the
    heterogeneity statistics. The classical normal-approximation SE is also
    returned as ``se_normal`` so the two can be compared directly.
    """
    effects = np.asarray(effects, float)
    variances = np.asarray(variances, float)
    k = len(effects)
    if k < 2:
        return None
    tau2 = paule_mandel_tau2(effects, variances)
    w = 1.0 / (variances + tau2)
    mu = float(np.sum(w * effects) / np.sum(w))
    se_normal = float(np.sqrt(1.0 / np.sum(w)))

    # HKSJ: rescale the variance by the weighted residual mean square.
    #
    # Note a property specific to Paule-Mandel: PM chooses tau^2 so that the
    # generalized Q equals k-1 exactly, hence q_gen == 1 and se_hksj == the
    # normal-approximation SE whenever tau^2 > 0. With PM, HKSJ therefore acts
    # entirely through the t(k-1) critical value rather than through the SE.
    # q_gen < 1 arises only when PM truncates tau^2 at 0, and there plain HKSJ
    # would be NARROWER than the normal interval; the standard ad-hoc
    # safeguard (Knapp-Hartung; Rover et al. 2015) truncates the SE upward so
    # the procedure can never be anti-conservative. `hksj_truncated` flags it.
    q_gen = float(np.sum(w * (effects - mu) ** 2) / (k - 1))
    se_raw = float(np.sqrt(q_gen / np.sum(w)))
    # Relative tolerance: under PM q_gen is 1 up to floating point.
    se_hksj = se_raw if se_raw >= se_normal * (1 - 1e-9) else se_normal
    df = k - 1
    tcrit = float(t_dist.ppf(0.975, df))
    p = float(2 * t_dist.sf(abs(mu / se_hksj), df)) if se_hksj > 0 else np.nan

    # Fixed-effect Q / I^2 for heterogeneity reporting.
    wf = 1.0 / variances
    mu_fixed = float(np.sum(wf * effects) / np.sum(wf))
    q_stat = float(np.sum(wf * (effects - mu_fixed) ** 2))
    i2 = float(max(0.0, (q_stat - df) / q_stat) * 100) if q_stat > 0 else 0.0
    return {
        'k': k,
        'effect': mu,
        'se_hksj': se_hksj,
        'se_hksj_raw': se_raw,
        'hksj_truncated': bool(se_raw < se_normal * (1 - 1e-9)),
        'q_gen': q_gen,
        'se_normal': se_normal,
        'lo': mu - tcrit * se_hksj,
        'hi': mu + tcrit * se_hksj,
        'p': p,
        'p_normal': float(2 * norm.sf(abs(mu / se_normal))) if se_normal > 0 else np.nan,
        'tau2': tau2,
        'I2_pct': i2,
        'Q': q_stat,
        'Q_p': float(chi2.sf(q_stat, df)) if df > 0 else np.nan,
        'df': df,
    }


def meta_spearman(rhos, ns):
    """Fisher-z + PM + HKSJ meta-analysis of Spearman coefficients.

    Fisher-z variance is the standard 1/(n-3), matching the previous F2D
    inference so the only deliberate change is PM/HKSJ replacing DL/normal.
    """
    rhos = np.asarray(rhos, float)
    ns = np.asarray(ns, float)
    z = np.arctanh(np.clip(rhos, -0.999999, 0.999999))
    variances = 1.0 / (ns - 3.0)
    out = meta_hksj(z, variances)
    if out is None:
        return None
    out.update({
        'rho_re': float(np.tanh(out['effect'])),
        'rho_lo': float(np.tanh(out['lo'])),
        'rho_hi': float(np.tanh(out['hi'])),
    })
    return out


def hedges_g(high, low):
    """Hedges' g and its variance for two independent groups."""
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    n1, n2 = len(high), len(low)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    s_pooled = np.sqrt(
        ((n1 - 1) * high.var(ddof=1) + (n2 - 1) * low.var(ddof=1)) / (n1 + n2 - 2)
    )
    if not np.isfinite(s_pooled) or s_pooled == 0:
        return np.nan, np.nan
    d = (high.mean() - low.mean()) / s_pooled
    j = 1 - 3 / (4 * (n1 + n2) - 9)          # small-sample correction
    g = j * d
    var = j ** 2 * ((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2 - 2)))
    return float(g), float(var)


def bh(pvalues):
    from statsmodels.stats.multitest import multipletests
    pvalues = np.asarray(pvalues, float)
    out = np.full(pvalues.shape, np.nan)
    ok = np.isfinite(pvalues)
    if ok.any():
        out[ok] = multipletests(pvalues[ok], method='fdr_bh')[1]
    return out

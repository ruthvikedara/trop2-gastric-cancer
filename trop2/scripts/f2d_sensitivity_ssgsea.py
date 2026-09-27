"""Pathway sensitivity analysis: ssGSEA scoring."""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathway_meta import (
    COHORTS, MIN_COVERAGE, MIN_GENES, bh, cohort_matrix, load_signatures,
    meta_spearman,
)

RESULTS = Path(__file__).resolve().parent.parent / 'results' / 'pathways'
RESULTS.mkdir(parents=True, exist_ok=True)


def run_ssgsea(expression, gene_sets):
    """gseapy ssGSEA on one cohort. expression: samples x genes."""
    import gseapy
    matrix = expression.T          # gseapy wants genes x samples
    matrix = matrix.loc[matrix.notna().any(axis=1)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = gseapy.ssgsea(
            data=matrix,
            gene_sets={k: list(v) for k, v in gene_sets.items()},
            outdir=None,
            sample_norm_method='rank',
            min_size=1,            # keep the 2-gene CYT signature
            max_size=100000,
            no_plot=True,
            threads=4,
        )
    scores = result.res2d.pivot(index='Term', columns='Name', values='NES')
    return scores.astype(float)


def main():
    order, genes_by, section_of, _ = load_signatures()
    rows, coverage_rows = [], []
    for cohort, rel in COHORTS.items():
        expression, _ = cohort_matrix(cohort, rel)
        trop2 = expression['TACSTD2'].astype(float)

        usable = {}
        for pathway in order:
            genes = genes_by[pathway]
            present = [g for g in genes if g in expression.columns]
            coverage = len(present) / len(genes) if genes else 0.0
            coverage_rows.append({
                'cohort': cohort, 'pathway': pathway,
                'genes_total': len(genes), 'genes_measured': len(present),
                'coverage_pct': 100 * coverage,
            })
            if len(present) >= MIN_GENES and coverage >= MIN_COVERAGE:
                usable[pathway] = present
        scores = run_ssgsea(expression, usable)

        for pathway, present in usable.items():
            if pathway not in scores.index:
                continue
            series = scores.loc[pathway]
            paired = pd.concat([series.rename('score'), trop2.rename('trop2')],
                               axis=1).dropna()
            if len(paired) < 10:
                continue
            rho, p = spearmanr(paired['score'], paired['trop2'])
            rows.append({'cohort': cohort, 'pathway': pathway,
                         'section': section_of[pathway],
                         'genes_measured': len(present),
                         'n': len(paired), 'rho': float(rho), 'p': float(p)})
        print(f'[{cohort}] ssGSEA on {len(usable)} signatures, '
              f'{len(expression)} tumors')

    per_cohort = pd.DataFrame(rows)
    per_cohort.to_csv(RESULTS / 'F2D_sensitivity_ssgsea_cohort.csv', index=False)

    meta = []
    for pathway in order:
        block = per_cohort[per_cohort['pathway'] == pathway]
        if len(block) < 2:
            continue
        pooled = meta_spearman(block['rho'], block['n'])
        meta.append({'pathway': pathway, 'section': section_of[pathway],
                     'k': pooled['k'], 'rho_re': pooled['rho_re'],
                     'rho_lo': pooled['rho_lo'], 'rho_hi': pooled['rho_hi'],
                     'tau2': pooled['tau2'], 'I2_pct': pooled['I2_pct'],
                     'Q': pooled['Q'], 'Q_p': pooled['Q_p'],
                     'meta_p': pooled['p']})
    summary = pd.DataFrame(meta)
    summary['fdr'] = bh(summary['meta_p'])
    summary['significant'] = summary['fdr'] < 0.05

    primary = pd.read_csv(
        Path(__file__).resolve().parent.parent / 'plots' / 'pathways' /
        'F2D_TROP2_pathways_meta_summary.csv'
    )[['pathway', 'rho_re', 'fdr', 'significant']].rename(
        columns={'rho_re': 'meanz_rho', 'fdr': 'meanz_fdr',
                 'significant': 'meanz_significant'})
    summary = summary.merge(primary, on='pathway', how='outer')
    summary['same_direction'] = (
        np.sign(summary['rho_re']) == np.sign(summary['meanz_rho']))
    summary['same_significance'] = (
        summary['significant'].fillna(False) ==
        summary['meanz_significant'].fillna(False))
    summary.to_csv(RESULTS / 'F2D_sensitivity_ssgsea.csv', index=False)

    print('\n' + summary[['pathway', 'rho_re', 'fdr', 'significant',
                          'meanz_rho', 'meanz_fdr', 'meanz_significant',
                          'same_direction']].to_string(index=False))
    print(f"\nssGSEA FDR<0.05: {int(summary['significant'].fillna(False).sum())} / 32"
          f" | mean-z: {int(summary['meanz_significant'].fillna(False).sum())} / 32")
    print('direction concordant:', int(summary['same_direction'].sum()), '/ 32')
    print('significance concordant:', int(summary['same_significance'].sum()), '/ 32')
    cyt = summary[summary['pathway'].str.contains('CYT', na=False)]
    print('\n2-gene CYT signature retained:', 'yes' if len(cyt) else 'NO')
    if len(cyt):
        print(cyt[['pathway', 'k', 'rho_re', 'fdr']].to_string(index=False))


if __name__ == '__main__':
    main()

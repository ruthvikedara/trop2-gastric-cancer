"""Pathway sensitivity analysis: GSVA scoring."""

from pathlib import Path
import sys

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathway_meta import (
    COHORTS, bh, cohort_matrix, complete_expression_and_sets,
    load_signatures_with_rbsig, meta_spearman, rank_signature_scores,
)

RESULTS = Path(__file__).resolve().parent.parent / 'results' / 'pathways'
RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    order, genes_by, section_of, _ = load_signatures_with_rbsig()
    rows = []
    for cohort, rel in COHORTS.items():
        expression, _ = cohort_matrix(cohort, rel)
        complete, usable, coverage = complete_expression_and_sets(
            expression, order, genes_by)
        scores = rank_signature_scores(complete, usable, method='gsva')
        trop2 = expression['TACSTD2'].astype(float)
        for pathway in order:
            if pathway not in usable or pathway not in scores.index:
                continue
            paired = pd.concat(
                [scores.loc[pathway].rename('score'), trop2.rename('trop2')],
                axis=1,
            ).dropna()
            rho, p = spearmanr(paired['score'], paired['trop2'])
            info = coverage.loc[coverage['pathway'] == pathway].iloc[0]
            rows.append({
                'cohort': cohort,
                'pathway': pathway,
                'section': section_of[pathway],
                'genes_total': int(info['genes_total']),
                'genes_measured': int(info['genes_measured']),
                'coverage_pct': float(info['coverage_pct']),
                'n': len(paired),
                'rho': float(rho),
                'p': float(p),
            })
        print(f'[{cohort}] GSVA: n={len(expression)}, signatures={len(usable)}')

    cohort_stats = pd.DataFrame(rows)
    cohort_stats.to_csv(RESULTS / 'F2D_sensitivity_gsva_cohort.csv', index=False)

    summaries = []
    for pathway in order:
        block = cohort_stats[cohort_stats['pathway'] == pathway]
        pooled = meta_spearman(block['rho'], block['n'])
        summaries.append({
            'pathway': pathway,
            'section': section_of[pathway],
            'n_cohorts': len(block),
            'rho_re': pooled['rho_re'],
            'rho_lo': pooled['rho_lo'],
            'rho_hi': pooled['rho_hi'],
            'tau2': pooled['tau2'],
            'I2_pct': pooled['I2_pct'],
            'meta_p': pooled['p'],
        })
    summary = pd.DataFrame(summaries)
    summary['fdr'] = bh(summary['meta_p'])
    summary['significant'] = summary['fdr'] < 0.05
    summary.to_csv(RESULTS / 'F2D_sensitivity_gsva.csv', index=False)

    print('\n' + summary.to_string(index=False))
    print(f"\nGSVA FDR<0.05: {int(summary['significant'].sum())} / 33")


if __name__ == '__main__':
    main()

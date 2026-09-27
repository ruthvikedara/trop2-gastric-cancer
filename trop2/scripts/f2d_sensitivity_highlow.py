"""Pathway sensitivity analysis: median TROP2-high vs low."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathway_meta import (
    COHORTS, bh, cohort_matrix, hedges_g, load_signatures, meta_hksj,
    signature_score,
)

RESULTS = Path(__file__).resolve().parent.parent / 'results' / 'pathways'
RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    order, genes_by, section_of, _ = load_signatures()
    rows = []
    for cohort, rel in COHORTS.items():
        expression, z = cohort_matrix(cohort, rel)
        trop2 = expression['TACSTD2'].astype(float)
        high_mask = trop2 >= trop2.median()
        for pathway in order:
            score, info = signature_score(z, genes_by[pathway])
            if score is None:
                continue
            paired = pd.concat(
                [score.rename('score'), high_mask.rename('high')], axis=1
            ).dropna()
            g, var = hedges_g(paired.loc[paired['high'], 'score'].to_numpy(),
                              paired.loc[~paired['high'], 'score'].to_numpy())
            if not np.isfinite(g):
                continue
            rows.append({'cohort': cohort, 'pathway': pathway,
                         'section': section_of[pathway],
                         'n_high': int(paired['high'].sum()),
                         'n_low': int((~paired['high']).sum()),
                         'g': g, 'var': var})
        print(f'[{cohort}] done')

    per_cohort = pd.DataFrame(rows)
    meta = []
    for pathway in order:
        block = per_cohort[per_cohort['pathway'] == pathway]
        pooled = meta_hksj(block['g'].to_numpy(), block['var'].to_numpy())
        meta.append({'pathway': pathway, 'section': section_of[pathway],
                     'k': pooled['k'], 'g_re': pooled['effect'],
                     'g_lo': pooled['lo'], 'g_hi': pooled['hi'],
                     'tau2': pooled['tau2'], 'I2_pct': pooled['I2_pct'],
                     'Q': pooled['Q'], 'Q_p': pooled['Q_p'],
                     'meta_p': pooled['p']})
    summary = pd.DataFrame(meta)
    summary['fdr'] = bh(summary['meta_p'])
    summary['significant'] = summary['fdr'] < 0.05

    # Concordance with the continuous primary analysis.
    primary = pd.read_csv(
        Path(__file__).resolve().parent.parent / 'plots' / 'pathways' /
        'F2D_TROP2_pathways_meta_summary.csv'
    )[['pathway', 'rho_re', 'fdr', 'significant']].rename(
        columns={'rho_re': 'primary_rho', 'fdr': 'primary_fdr',
                 'significant': 'primary_significant'})
    summary = summary.merge(primary, on='pathway', how='left')
    summary['same_direction'] = (
        np.sign(summary['g_re']) == np.sign(summary['primary_rho']))
    summary['same_significance'] = (
        summary['significant'] == summary['primary_significant'])
    summary.to_csv(RESULTS / 'F2D_sensitivity_highlow.csv', index=False)

    print('\n' + summary[['pathway', 'g_re', 'g_lo', 'g_hi', 'meta_p', 'fdr',
                          'significant', 'primary_significant',
                          'same_direction']].to_string(index=False))
    print(f"\nFDR<0.05: {int(summary['significant'].sum())} / 32 "
          f"(continuous primary: {int(summary['primary_significant'].sum())})")
    print('direction concordant:', int(summary['same_direction'].sum()), '/ 32')
    print('significance concordant:', int(summary['same_significance'].sum()), '/ 32')
    disagree = summary[~summary['same_significance']]
    if len(disagree):
        print('\nSignificance differs for:')
        print(disagree[['pathway', 'primary_rho', 'primary_fdr', 'g_re',
                        'fdr']].to_string(index=False))


if __name__ == '__main__':
    main()

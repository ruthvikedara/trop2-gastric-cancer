"""Pathway sensitivity analysis: leave-one-cohort-out."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathway_meta import COHORTS, bh, meta_spearman

PLOTS = Path(__file__).resolve().parent.parent / 'plots' / 'pathways'
RESULTS = Path(__file__).resolve().parent.parent / 'results' / 'pathways'
RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    stats = pd.read_csv(PLOTS / 'F2D_TROP2_pathways_cohort_correlations.csv')
    primary = pd.read_csv(PLOTS / 'F2D_TROP2_pathways_meta_summary.csv')
    order = primary['pathway'].tolist()

    runs = []
    for dropped in COHORTS:
        kept = stats[stats['cohort'] != dropped]
        rows = []
        for pathway in order:
            block = kept[kept['pathway'] == pathway]
            pooled = meta_spearman(block['rho'], block['n'])
            rows.append({'dropped_cohort': dropped, 'pathway': pathway,
                         'k': pooled['k'], 'rho_re': pooled['rho_re'],
                         'rho_lo': pooled['rho_lo'], 'rho_hi': pooled['rho_hi'],
                         'tau2': pooled['tau2'], 'I2_pct': pooled['I2_pct'],
                         'meta_p': pooled['p']})
        run = pd.DataFrame(rows)
        run['fdr'] = bh(run['meta_p'])
        run['significant'] = run['fdr'] < 0.05
        runs.append(run)
    loco = pd.concat(runs, ignore_index=True)
    loco.to_csv(RESULTS / 'F2D_sensitivity_loco.csv', index=False)

    base = primary.set_index('pathway')
    summary = []
    for pathway in order:
        block = loco[loco['pathway'] == pathway]
        base_rho = base.loc[pathway, 'rho_re']
        stable_dir = bool((np.sign(block['rho_re']) == np.sign(base_rho)).all())
        summary.append({
            'pathway': pathway,
            'section': base.loc[pathway, 'section'],
            'primary_rho': base_rho,
            'primary_fdr': base.loc[pathway, 'fdr'],
            'primary_significant': bool(base.loc[pathway, 'significant']),
            'loco_rho_min': block['rho_re'].min(),
            'loco_rho_max': block['rho_re'].max(),
            'loco_fdr_max': block['fdr'].max(),
            'n_runs_significant': int(block['significant'].sum()),
            'significant_in_all_runs': bool(block['significant'].all()),
            'direction_stable': stable_dir,
            'rho_without_GSE84437': float(
                block.loc[block['dropped_cohort'] == 'GSE84437', 'rho_re'].iloc[0]),
            'fdr_without_GSE84437': float(
                block.loc[block['dropped_cohort'] == 'GSE84437', 'fdr'].iloc[0]),
            'significant_without_GSE84437': bool(
                block.loc[block['dropped_cohort'] == 'GSE84437',
                          'significant'].iloc[0]),
        })
    out = pd.DataFrame(summary)
    out.to_csv(RESULTS / 'F2D_sensitivity_loco_summary.csv', index=False)

    sig = out[out['primary_significant']]
    print('Primary FDR-significant signatures - leave-one-cohort-out stability')
    print(sig[['pathway', 'primary_rho', 'primary_fdr', 'loco_rho_min',
               'loco_rho_max', 'loco_fdr_max', 'n_runs_significant',
               'rho_without_GSE84437', 'fdr_without_GSE84437',
               'significant_without_GSE84437']].to_string(index=False))
    print(f'\nDirection stable in all {len(COHORTS)} runs:',
          int(sig['direction_stable'].sum()), '/', len(sig))
    print(f'Significant in all {len(COHORTS)} runs:',
          int(sig['significant_in_all_runs'].sum()), '/', len(sig))
    fragile = sig[~sig['significant_in_all_runs']]
    if len(fragile):
        print('\nLoses significance in at least one run:')
        for row in fragile.itertuples():
            block = loco[(loco['pathway'] == row.pathway) & (~loco['significant'])]
            print(f'  {row.pathway}: when dropping '
                  f'{", ".join(block["dropped_cohort"])} '
                  f'(max FDR {row.loco_fdr_max:.3f})')

    gained = out[(~out['primary_significant']) &
                 (out['n_runs_significant'] > 0)]
    if len(gained):
        print('\nNot significant in the primary analysis but significant in '
              'some leave-one-out run:')
        print(gained[['pathway', 'primary_fdr', 'n_runs_significant',
                      'loco_fdr_max']].to_string(index=False))


if __name__ == '__main__':
    main()

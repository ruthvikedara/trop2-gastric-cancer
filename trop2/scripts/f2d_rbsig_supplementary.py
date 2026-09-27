"""RB loss-of-function signature vs TROP2."""

from pathlib import Path
import sys

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathway_meta import (
    COHORTS, cohort_matrix, meta_spearman, signature_score,
)
from plot_f2_proliferation_rb_pathways import RBSIG

RESULTS = Path(__file__).resolve().parent.parent / 'results' / 'pathways'
RESULTS.mkdir(parents=True, exist_ok=True)

NAME = 'RB loss-of-function (RBsig)'


def main():
    genes = [g for g in dict.fromkeys(RBSIG) if g != 'TACSTD2']
    rows = []
    for cohort, rel in COHORTS.items():
        expression, z = cohort_matrix(cohort, rel)
        trop2 = expression['TACSTD2'].astype(float)
        score, info = signature_score(z, genes)
        if score is None:
            print(f'[{cohort}] excluded by coverage rule: {info}')
            continue
        paired = pd.concat([score.rename('score'), trop2.rename('trop2')],
                           axis=1).dropna()
        rho, p = spearmanr(paired['score'], paired['trop2'])
        rows.append({'cohort': cohort, 'pathway': NAME, **info,
                     'n': len(paired), 'rho': float(rho), 'p': float(p)})

    per_cohort = pd.DataFrame(rows)
    pooled = meta_spearman(per_cohort['rho'], per_cohort['n'])
    per_cohort['pooled_rho'] = pooled['rho_re']
    per_cohort['pooled_lo'] = pooled['rho_lo']
    per_cohort['pooled_hi'] = pooled['rho_hi']
    per_cohort['tau2'] = pooled['tau2']
    per_cohort['I2_pct'] = pooled['I2_pct']
    per_cohort['meta_p_unadjusted'] = pooled['p']
    per_cohort['bh_family'] = 'none (single pre-specified extra test)'
    per_cohort.to_csv(RESULTS / 'F2D_RBsig_outside_family.csv', index=False)

    print(per_cohort[['cohort', 'genes_total', 'genes_measured',
                      'coverage_pct', 'n', 'rho', 'p']].to_string(index=False))
    print(f"\nPooled rho = {pooled['rho_re']:+.3f} "
          f"[{pooled['rho_lo']:+.3f}, {pooled['rho_hi']:+.3f}], "
          f"unadjusted meta p = {pooled['p']:.4g}, "
          f"tau2 = {pooled['tau2']:.4f}, I2 = {pooled['I2_pct']:.1f}%")
    print('Not BH-corrected into the 32-signature family.')


if __name__ == '__main__':
    main()

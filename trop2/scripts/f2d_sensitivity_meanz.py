"""Pathway sensitivity analysis: mean-z scoring."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_pub_style, clean_ax, savefig
from pathway_meta import (
    COHORTS, bh, cohort_matrix, load_signatures, meta_spearman,
    signature_score,
)

OUT = Path(__file__).resolve().parent.parent / 'results' / 'pathways'
OUT.mkdir(parents=True, exist_ok=True)
STEM = 'F2D_sensitivity_meanz'

SHORT_LABELS = {
    'IFN-γ response (6-gene)': 'IFN-γ',
    'JAK-STAT signaling': 'JAK-STAT',
    'Tryptophan / kynurenine metabolism': 'Trp/kyn.',
    'AHR signaling': 'AHR',
    'Hippo-YAP/TEAD signaling': 'Hippo-YAP',
    'Tight junction': 'Tight jxn.',
    'Epithelial-mesenchymal transition (EMT)': 'EMT',
    'Wnt signaling': 'Wnt',
    'NF-κB signaling': 'NF-κB',
    'PI3K-Akt signaling': 'PI3K-AKT',
    'mTOR signaling': 'mTOR',
    'TNF signaling': 'TNF',
    'VEGF signaling': 'VEGF',
    'MAPK signaling': 'MAPK',
    'p53 signaling': 'p53',
    'Cell cycle': 'Cell cycle',
    'Apoptosis': 'Apoptosis',
    'Notch signaling': 'Notch',
    'Tumor Inflammation Signature (TIS, 18-gene)': 'TIS',
    'Cytolytic activity (CYT)': 'CYT',
    'Antigen presentation (MHC-I)': 'MHC-I',
    'NK cell-mediated cytotoxicity': 'NK cytotox.',
    'T cell receptor signaling': 'TCR',
    'TGF-β signaling / immune exclusion': 'TGF-β/excl.',
    'Complement and coagulation cascades': 'Complement',
    'Cytosolic DNA-sensing (cGAS-STING)': 'cGAS-STING',
    'Chemokine signaling': 'Chemokine',
    'Hypoxia': 'Hypoxia',
    'Oxidative phosphorylation (OXPHOS)': 'OXPHOS',
    'Glycolysis / gluconeogenesis': 'Glycolysis',
    'Fatty acid metabolism': 'FA metab.',
    'DNA repair': 'DNA repair',
}


def per_cohort_correlations(order, genes_by, section_of, verbose=True):
    """Within-cohort Spearman rho of each signature score vs continuous TACSTD2."""
    rows, excluded = [], []
    for cohort, rel in COHORTS.items():
        expression, z = cohort_matrix(cohort, rel)
        trop2 = expression['TACSTD2'].astype(float)
        for pathway in order:
            score, info = signature_score(z, genes_by[pathway])
            record = {'cohort': cohort, 'pathway': pathway,
                      'section': section_of[pathway], **info}
            if score is None:
                excluded.append(record)
                continue
            paired = pd.concat([score, trop2], axis=1,
                               keys=['score', 'trop2']).dropna()
            if len(paired) < 10:
                record['excluded'] = True
                record['reason'] = 'n<10 paired samples'
                excluded.append(record)
                continue
            rho, p = spearmanr(paired['score'], paired['trop2'])
            rows.append({**record, 'n': len(paired), 'rho': float(rho),
                         'p': float(p)})
        if verbose:
            print(f'[{cohort}] tumor samples n={len(expression)}')
    return pd.DataFrame(rows), pd.DataFrame(excluded)


def meta_table(stats, order, section_of):
    rows = []
    for pathway in order:
        group = stats[stats['pathway'] == pathway]
        pooled = meta_spearman(group['rho'], group['n'])
        rows.append({
            'pathway': pathway,
            'section': section_of[pathway],
            'n_cohorts': len(group),
            'n_nominal_p_lt_0_05': int((group['p'] < 0.05).sum()),
            'rho_re': pooled['rho_re'],
            'rho_lo': pooled['rho_lo'],
            'rho_hi': pooled['rho_hi'],
            'tau2': pooled['tau2'],
            'I2_pct': pooled['I2_pct'],
            'Q': pooled['Q'],
            'Q_p': pooled['Q_p'],
            'meta_p': pooled['p'],
            'meta_p_normal_approx': pooled['p_normal'],
            'se_hksj': pooled['se_hksj'],
            'se_normal': pooled['se_normal'],
            'hksj_truncated': pooled['hksj_truncated'],
            'df': pooled['df'],
        })
    summary = pd.DataFrame(rows)
    summary['fdr'] = bh(summary['meta_p'])
    summary['significant'] = summary['fdr'] < 0.05
    return summary


def main():
    apply_pub_style()
    order, genes_by, section_of, dropped_trop2 = load_signatures()
    assert len(order) == 32, f'expected 32 signatures, got {len(order)}'
    print(f'signatures: {len(order)} | cohorts: {len(COHORTS)}')
    print('TACSTD2 removed from:', dropped_trop2 or 'none (absent from all sets)')

    stats, excluded = per_cohort_correlations(order, genes_by, section_of)
    if len(excluded):
        print('\nEXCLUDED cohort-signature pairs (coverage rule):')
        print(excluded.to_string(index=False))
    else:
        print('\nNo cohort-signature pair excluded by the >=2 gene / >=50% '
              'coverage rule.')

    summary = meta_table(stats, order, section_of)

    cohort_cols = ['cohort', 'pathway', 'section', 'genes_total',
                   'genes_measured', 'coverage_pct', 'n', 'rho', 'p']
    stats[cohort_cols].to_csv(
        OUT / f'{STEM}_cohort_correlations.csv', index=False)
    summary.to_csv(OUT / f'{STEM}_meta_summary.csv', index=False)
    # Canonical legacy merged file preserved for downstream consumers.
    stats.merge(summary, on=['pathway', 'section'], how='left').to_csv(
        OUT / f'{STEM}_stats.csv', index=False)

    cohort_order = list(COHORTS)
    matrix = (stats.pivot(index='pathway', columns='cohort', values='rho')
              .reindex(index=order, columns=cohort_order))
    summary_i = summary.set_index('pathway').reindex(order)
    n_path, n_coh = len(order), len(cohort_order)

    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[31, 1.0], height_ratios=[1.35, 1.0],
        wspace=0.08, hspace=0.34,
        left=0.085, right=0.97, top=0.84, bottom=0.30,
    )
    ax = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[0, 1])
    meta_ax = fig.add_subplot(gs[1, 0], sharex=ax)

    image = ax.imshow(
        matrix.T.to_numpy(dtype=float), cmap='RdBu_r', vmin=-0.5, vmax=0.5,
        aspect='auto', interpolation='nearest',
    )
    ax.set_xticks(range(n_path))
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.set_yticks(range(n_coh))
    ax.set_yticklabels(cohort_order, fontsize=6.2)
    ax.tick_params(axis='y', length=0)

    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label('Spearman ρ', fontsize=6.8)
    cbar.ax.tick_params(labelsize=5.8)

    x = np.arange(n_path)
    sig = summary_i['fdr'].to_numpy() < 0.05
    directions = summary_i['rho_re'].to_numpy()
    colors = np.where(sig,
                      np.where(directions >= 0, '#CD2626', '#1874CD'),
                      '#999999')
    for i in range(n_path):
        meta_ax.plot([i, i],
                     [summary_i.iloc[i]['rho_lo'], summary_i.iloc[i]['rho_hi']],
                     color=colors[i], lw=1.0, zorder=2)
    meta_ax.scatter(x, summary_i['rho_re'], c=colors, s=12,
                    edgecolor='black', linewidth=0.25, zorder=3)
    meta_ax.axhline(0, color='black', lw=0.7, ls='--')
    meta_ax.set_ylim(-0.55, 0.55)
    meta_ax.set_ylabel('Pooled Spearman ρ\n(95% CI)', fontsize=6.8)
    meta_ax.set_xticks(x)
    meta_ax.set_xticklabels([SHORT_LABELS.get(p, p) for p in order],
                            rotation=90, ha='center', va='top', fontsize=5.4)
    meta_ax.tick_params(axis='y', labelsize=5.8)
    meta_ax.tick_params(axis='x', length=0, pad=2)
    meta_ax.set_title(
        'Cross-cohort random-effects estimates '
        '(color indicates BH FDR < 0.05; gray indicates FDR ≥ 0.05)',
        fontsize=7.2, fontweight='bold', pad=3,
    )
    clean_ax(meta_ax)

    section_bounds, start = [], 0
    prev_section = section_of[order[0]]
    for i, pathway in enumerate(order):
        sec = section_of[pathway]
        if sec != prev_section:
            ax.axvline(i - 0.5, color='white', lw=1.2)
            meta_ax.axvline(i - 0.5, color='#BBBBBB', lw=0.7)
            section_bounds.append((start, i - 1, prev_section))
            start, prev_section = i, sec
    section_bounds.append((start, n_path - 1, prev_section))
    for lo, hi, sec in section_bounds:
        ax.text((lo + hi) / 2, -1.35, sec, ha='center', va='bottom',
                fontsize=5.2, fontweight='bold', clip_on=False)

    fig.suptitle(
        'TROP2 association with curated pathway signature scores '
        'across gastric cancer cohorts',
        fontsize=9.5, fontweight='bold', y=0.985,
    )
    fig.text(
        0.085, 0.012,
        'Signature score = mean within-cohort gene-wise z-score. Per-cohort Spearman ρ; '
        'Fisher-z Paule-Mandel random-effects meta-analysis\nwith Hartung-Knapp inference; '
        'BH correction across 32 signatures.',
        fontsize=5.8, color='#444444', linespacing=1.45,
    )
    savefig(fig, OUT / STEM, pad_inches=0.05)

    show = summary[['section', 'pathway', 'n_cohorts', 'rho_re', 'rho_lo',
                    'rho_hi', 'tau2', 'I2_pct', 'meta_p', 'fdr',
                    'significant']]
    print('\n' + show.to_string(index=False))
    print(f'\nFDR<0.05: {int(summary["significant"].sum())} / 32')
    print('HKSJ SE truncated upward (PM tau2=0):',
          int(summary['hksj_truncated'].sum()))


if __name__ == '__main__':
    main()

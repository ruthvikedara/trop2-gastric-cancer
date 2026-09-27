"""Validate subtype and MSI classifiers against TCGA labels."""

from pathlib import Path

import numpy as np
import pandas as pd

from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT
PROC = BASE / 'data/clinical/gastric/processed'
OUT = BASE / 'trop2/plots/phase1/classifier_validation.md'
OUT.parent.mkdir(parents=True, exist_ok=True)


def norm_barcode(s):
    """TCGA sample id -> 12-char patient barcode (TCGA-XX-XXXX)."""
    return '-'.join(str(s).split('-')[:3])


def confusion(truth, pred, labels):
    m = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for t, p in zip(truth, pred):
        if t in labels and p in labels:
            m.loc[t, p] += 1
    return m


def main():
    lines = ['# Classifier validation vs TCGA ground truth\n']

    truth = pd.read_csv(PROC / 'TCGA_clinical_with_subtypes.csv')
    truth['pid'] = truth['patient_id'].map(norm_barcode)
    truth = truth.set_index('pid')

    subs = pd.read_csv(PROC / 'TCGA-STAD_subtypes_predicted.csv')
    subs['pid'] = subs['sample_id'].map(norm_barcode)
    subs = subs.drop_duplicates('pid').set_index('pid')

    msi = pd.read_csv(PROC / 'TCGA-STAD_msi_predicted.csv')
    msi['pid'] = msi['sample_id'].map(norm_barcode)
    msi = msi.drop_duplicates('pid').set_index('pid')

    # 1. PreMSIm vs truth MSI
    df = truth[['molecular_subtype']].join(msi[['msi_premsim']], how='inner').dropna()
    df['truth_msi'] = (df['molecular_subtype'] == 'MSI')
    df['pred_msi'] = (df['msi_premsim'] == 'MSI-H')
    tp = int(((df.truth_msi) & (df.pred_msi)).sum())
    tn = int(((~df.truth_msi) & (~df.pred_msi)).sum())
    fp = int(((~df.truth_msi) & (df.pred_msi)).sum())
    fn = int(((df.truth_msi) & (~df.pred_msi)).sum())
    n = len(df)
    sens = tp / (tp + fn) if (tp + fn) else float('nan')
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    ppv = tp / (tp + fp) if (tp + fp) else float('nan')
    acc = (tp + tn) / n if n else float('nan')
    lines += [
        '## 1. PreMSIm MSI-H vs truth (molecular_subtype == MSI)\n',
        f'- n = {n}; truth MSI-H = {tp + fn}',
        f'- TP={tp} TN={tn} FP={fp} FN={fn}',
        f'- **Sensitivity={sens:.2%}  Specificity={spec:.2%}  PPV={ppv:.2%}  Accuracy={acc:.2%}**\n',
    ]
    print(f'[MSI] n={n} sens={sens:.2%} spec={spec:.2%} ppv={ppv:.2%} acc={acc:.2%}')

    # 2. GCclassifier TCGA-mode vs truth subtype
    if 'TCGA_subtype' in subs.columns:
        labels = ['EBV', 'MSI', 'GS', 'CIN']
        d2 = truth[['molecular_subtype']].join(subs[['TCGA_subtype']], how='inner').dropna()
        d2 = d2[d2['molecular_subtype'].isin(labels)]
        cm = confusion(d2['molecular_subtype'], d2['TCGA_subtype'], labels)
        correct = int(np.trace(cm.values))
        total = int(cm.values.sum())
        acc2 = correct / total if total else float('nan')
        lines += ['## 2. GCclassifier TCGA-mode vs real subtype\n',
                  f'- n = {total}; overall accuracy = **{acc2:.2%}**\n',
                  'Confusion (rows = truth, cols = predicted):\n',
                  '```', cm.to_string(), '```', '']
        for lab in labels:
            row = cm.loc[lab].sum()
            rec = cm.loc[lab, lab] / row if row else float('nan')
            lines.append(f'- recall[{lab}] = {rec:.2%}  (n={row})')
        lines.append('')
        print(f'[TCGA-subtype] n={total} acc={acc2:.2%}')
        print(cm)
    else:
        lines.append('## 2. SKIPPED - no TCGA_subtype column in predictions\n')

    # 3. Cross-classifier MSI concordance
    parts = []
    if 'TCGA_subtype' in subs.columns:
        parts.append(('GCclassifier-TCGA', subs['TCGA_subtype'] == 'MSI'))
    if 'ACRG_subtype' in subs.columns:
        parts.append(('GCclassifier-ACRG', subs['ACRG_subtype'] == 'MSI'))
    base = msi['msi_premsim'] == 'MSI-H'
    lines.append('## 3. Cross-classifier MSI-call concordance (no ground truth needed)\n')
    for name, flags in parts:
        j = pd.concat([base.rename('premsim'), flags.rename('other')], axis=1).dropna()
        agree = (j['premsim'] == j['other']).mean()
        lines.append(f'- PreMSIm vs {name}: {agree:.2%} agreement (n={len(j)})')
        print(f'[concordance] PreMSIm vs {name}: {agree:.2%}')
    lines.append('')

    OUT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote {OUT}')


if __name__ == '__main__':
    main()

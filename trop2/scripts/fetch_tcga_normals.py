"""Fetch TCGA tumor and normal expression from UCSC Xena."""

import gzip
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xenaPython as xena

from genolib import PROJECT_ROOT

GENE = 'TACSTD2'
RAW = PROJECT_ROOT / 'data' / 'raw' / 'gastric'
OUT = PROJECT_ROOT / 'data' / 'processed' / 'gastric'
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def _download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f'  cached {dest.name} ({dest.stat().st_size/1e6:.1f} MB)')
        return
    print(f'  downloading {url}')
    req = urllib.request.Request(url, headers={'User-Agent': 'curl/8'})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, 'wb') as f:
        f.write(r.read())
    print(f'  saved {dest.name} ({dest.stat().st_size/1e6:.1f} MB)')


# 1. WITHIN-TCGA (HiSeqV2, includes solid-tissue-normal)
def fetch_within_tcga():
    print('[within-TCGA] HiSeqV2')
    url = 'https://tcga.xenahubs.net/download/TCGA.STAD.sampleMap/HiSeqV2.gz'
    dest = RAW / 'TCGA_STAD_HiSeqV2.gz'
    _download(url, dest)

    # Pull just the gene row + header (file is genes x samples, tab-separated).
    with gzip.open(dest, 'rt') as f:
        header = f.readline().rstrip('\n').split('\t')
        samples = header[1:]
        row = None
        for line in f:
            tab = line.index('\t')
            if line[:tab] == GENE:
                row = [float(v) if v.strip() else np.nan
                       for v in line[tab + 1:].rstrip('\n').split('\t')]
                break
    if row is None:
        raise ValueError(f'{GENE} not found in HiSeqV2')

    s = pd.Series(row, index=samples, name=GENE).dropna()
    # TCGA barcode suffix: -01 primary tumor, -11 solid-tissue-normal, -06 metastatic.
    suffix = s.index.str.extract(r'-(\d{2})[A-Z]?$', expand=False)
    group = suffix.map(lambda x: 'Normal' if x == '11' else
                       ('Tumor' if x in ('01', '06') else None))
    df = pd.DataFrame({'sample': s.index, 'group': group.values,
                       'source': 'TCGA (HiSeqV2)', GENE: s.values}).dropna(subset=['group'])
    out = OUT / 'TCGA_STAD_TROP2_normal_vs_tumor_withinTCGA.csv'
    df.to_csv(out, index=False)
    print(f'  tumor n={(df.group=="Tumor").sum()}, normal n={(df.group=="Normal").sum()}')
    print(f'  -> {out.name}\n')


# 2. TCGA + GTEx (Toil recompute), stomach only, single-gene query
def fetch_tcga_gtex():
    print('[TCGA+GTEx] Toil recompute')
    hub = 'https://toil.xenahubs.net'
    ds = 'TcgaTargetGtex_rsem_gene_tpm'

    # Phenotype: identify stomach samples + study + sample_type. Small file.
    ph_url = f'{hub}/download/TcgaTargetGTEX_phenotype.txt.gz'
    ph_dest = RAW / 'TcgaTargetGTEX_phenotype.txt.gz'
    _download(ph_url, ph_dest)
    with gzip.open(ph_dest, 'rt', encoding='latin-1') as f:
        ph = pd.read_csv(f, sep='\t')
    ph.columns = [c.strip() for c in ph.columns]
    samp_col = ph.columns[0]
    site_col = next(c for c in ph.columns if c.lower() == '_primary_site')
    type_col = next(c for c in ph.columns if c.lower() == '_sample_type')
    study_col = next(c for c in ph.columns if c.lower() == '_study')

    stomach = ph[ph[site_col].astype(str).str.strip().str.lower() == 'stomach'].copy()
    print(f'  stomach samples in phenotype: {len(stomach)}')
    print(stomach.groupby([study_col, type_col]).size())

    # Tumor = TCGA Primary Tumor. Normal = GTEx Normal Tissue + TCGA Solid Tissue Normal
    # (GEPIA convention: pool TCGA normal with GTEx normal).
    def classify(r):
        st = str(r[type_col]).strip().lower()
        study = str(r[study_col]).strip().upper()
        if 'primary tumor' in st and study == 'TCGA':
            return 'Tumor'
        if 'normal' in st:
            return 'Normal'
        return None
    stomach['group'] = stomach.apply(classify, axis=1)
    stomach = stomach.dropna(subset=['group'])
    target_samples = stomach[samp_col].astype(str).tolist()

    # Query TACSTD2 for just these samples (avoids the 2.5 GB matrix download).
    all_samples = set(xena.dataset_samples(hub, ds, None))
    target_samples = [s for s in target_samples if s in all_samples]
    res = xena.dataset_gene_probe_avg(hub, ds, target_samples, [GENE])
    scores = res[0]['scores'][0]
    expr = pd.Series(scores, index=target_samples, name=GENE).replace('NaN', np.nan)
    expr = pd.to_numeric(expr, errors='coerce')

    df = stomach.set_index(samp_col).loc[target_samples, [study_col, type_col, 'group']].copy()
    df[GENE] = expr.values
    df = df.dropna(subset=[GENE]).reset_index()
    df = df.rename(columns={samp_col: 'sample', study_col: 'source', type_col: 'sample_type'})
    df = df[['sample', 'group', 'source', 'sample_type', GENE]]
    out = OUT / 'TCGA_STAD_TROP2_normal_vs_tumor_TCGA_GTEx.csv'
    df.to_csv(out, index=False)
    n_t = (df.group == 'Tumor').sum()
    n_n = (df.group == 'Normal').sum()
    n_gtex = ((df.group == 'Normal') & (df.source.str.upper() == 'GTEX')).sum()
    print(f'  tumor n={n_t} (TCGA); normal n={n_n} (GTEx {n_gtex} + TCGA {n_n-n_gtex})')
    print(f'  -> {out.name}\n')


if __name__ == '__main__':
    fetch_within_tcga()
    fetch_tcga_gtex()
    print('Done.')

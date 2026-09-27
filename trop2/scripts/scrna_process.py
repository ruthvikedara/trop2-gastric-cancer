"""Process GSE183904 (Kumar) scRNA-seq."""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT
RAW = BASE / 'data/raw/scrna/gastric'
OUTDIR = BASE / 'data/processed/scrna/gastric'
OUTDIR.mkdir(parents=True, exist_ok=True)

sc.settings.verbosity = 1

# All 26 primary gastric TUMOR samples (GSM, sampleN) from GSE183904.
SAMPLES = [
    ('GSM5573467', 'sample2'),  ('GSM5573468', 'sample3'),  ('GSM5573470', 'sample5'),
    ('GSM5573472', 'sample7'),  ('GSM5573473', 'sample8'),  ('GSM5573475', 'sample10'),
    ('GSM5573477', 'sample12'), ('GSM5573478', 'sample13'), ('GSM5573479', 'sample14'),
    ('GSM5573480', 'sample15'), ('GSM5573481', 'sample16'), ('GSM5573482', 'sample17'),
    ('GSM5573483', 'sample18'), ('GSM5573487', 'sample22'), ('GSM5573489', 'sample24'),
    ('GSM5573491', 'sample26'), ('GSM5573492', 'sample27'), ('GSM5573493', 'sample28'),
    ('GSM5573494', 'sample29'), ('GSM5573495', 'sample30'), ('GSM5573497', 'sample32'),
    ('GSM5573498', 'sample33'), ('GSM5573499', 'sample34'), ('GSM5573501', 'sample36'),
    ('GSM5573504', 'sample39'), ('GSM5573505', 'sample40'),
]

# Canonical lineage marker signatures (broad compartments).
LINEAGE_MARKERS = {
    'Epithelial':  ['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'CDH1', 'KRT7'],
    'T/NK':        ['CD3D', 'CD3E', 'CD2', 'TRAC', 'NKG7', 'GNLY'],
    'B/Plasma':    ['CD79A', 'MS4A1', 'CD79B', 'MZB1', 'IGHG1'],
    'Myeloid':     ['LYZ', 'CD68', 'CD14', 'C1QA', 'AIF1'],
    'Endothelial': ['PECAM1', 'VWF', 'CLDN5', 'CDH5'],
    'Fibroblast':  ['COL1A1', 'DCN', 'COL1A2', 'PDGFRB', 'LUM'],
    'Mast':        ['TPSAB1', 'TPSB2', 'CPA3'],
}


def load():
    adatas = []
    for gsm, sname in SAMPLES:
        f = next(RAW.glob(f'{gsm}_{sname}.csv.gz'))
        df = pd.read_csv(f, index_col=0)          # genes x cells
        df = df[~df.index.duplicated(keep='first')]
        X = csr_matrix(df.values.T.astype(np.float32))   # cells x genes
        a = ad.AnnData(X=X,
                       obs=pd.DataFrame(index=[f'{sname}_{b}' for b in df.columns]),
                       var=pd.DataFrame(index=df.index.astype(str)))
        a.obs['sample'] = sname
        adatas.append(a)
        print(f'  {sname}: {a.n_obs} cells x {a.n_vars} genes')
        del df
    adata = ad.concat(adatas, join='inner', merge='same')
    adata.var_names_make_unique()
    print(f'Combined: {adata.n_obs} cells x {adata.n_vars} genes')
    for g in ['TACSTD2', 'VTCN1', 'EPCAM']:
        print(f'  gene present: {g} = {g in adata.var_names}')
    return adata


def qc(adata):
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True, percent_top=None)
    n0 = adata.n_obs
    adata = adata[(adata.obs.n_genes_by_counts >= 200) &
                  (adata.obs.n_genes_by_counts <= 7000) &
                  (adata.obs.pct_counts_mt < 20)].copy()
    sc.pp.filter_genes(adata, min_cells=3)
    print(f'QC: {n0} -> {adata.n_obs} cells; {adata.n_vars} genes')
    return adata


def normalize(adata):
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    return adata


def embed(adata):
    a = adata.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor='seurat')
    a = a[:, a.var.highly_variable].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=30)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=30)
    sc.tl.umap(a)
    adata.obsm['X_pca'] = a.obsm['X_pca']
    adata.obsm['X_umap'] = a.obsm['X_umap']
    return adata


def annotate(adata):
    """Per-cell lineage = highest-scoring canonical signature."""
    score_cols = []
    for lineage, genes in LINEAGE_MARKERS.items():
        present = [g for g in genes if g in adata.var_names]
        sc.tl.score_genes(adata, present, score_name=f'score_{lineage}')
        score_cols.append(f'score_{lineage}')
    scores = adata.obs[score_cols]
    best = scores.idxmax(axis=1).str.replace('score_', '', regex=False)
    # cells with all-low scores -> Unassigned
    best[scores.max(axis=1) < 0] = 'Unassigned'
    adata.obs['lineage'] = pd.Categorical(best)
    print('\nLineage composition:')
    print(adata.obs['lineage'].value_counts())
    return adata


def main():
    print('=== Load ===')
    adata = load()
    print('\n=== QC ===')
    adata = qc(adata)
    print('\n=== Normalize ===')
    adata = normalize(adata)
    print('\n=== Embed (PCA/UMAP) ===')
    adata = embed(adata)
    print('\n=== Annotate lineages ===')
    adata = annotate(adata)
    out = OUTDIR / 'kumar_tumor.h5ad'
    adata.write(out)
    print(f'\nSaved {out}  ({adata.n_obs} cells)')


if __name__ == '__main__':
    main()

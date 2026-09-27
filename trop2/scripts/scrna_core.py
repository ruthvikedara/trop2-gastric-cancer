"""Shared scRNA-seq QC, normalization and annotation."""

import numpy as np
import pandas as pd
import scanpy as sc

sc.settings.verbosity = 1

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


def qc(adata, min_genes=200, max_genes=7000, max_pct_mt=20):
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True, percent_top=None)
    n0 = adata.n_obs
    keep = (adata.obs.n_genes_by_counts >= min_genes) & (adata.obs.n_genes_by_counts <= max_genes)
    if adata.var['mt'].any():
        keep &= adata.obs.pct_counts_mt < max_pct_mt
    adata = adata[keep].copy()
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
    best[scores.max(axis=1) < 0] = 'Unassigned'
    adata.obs['lineage'] = pd.Categorical(best)
    print('\nLineage composition:')
    print(adata.obs['lineage'].value_counts())
    return adata

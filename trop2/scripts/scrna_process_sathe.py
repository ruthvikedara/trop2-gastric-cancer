"""Process GSE150290 (Sathe) scRNA-seq."""

import sys
import tarfile
import tempfile
import shutil
from pathlib import Path

import scanpy as sc
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrna_core import qc, normalize, embed, annotate

from genolib import PROJECT_ROOT
BASE = PROJECT_ROOT
BIG = BASE / 'data/raw/scrna/gastric/GSE150290_RAW.tar'
OUTDIR = BASE / 'data/processed/scrna/gastric'
OUTDIR.mkdir(parents=True, exist_ok=True)

MIN_GENES_CELLCALL = 200  # empty-droplet cutoff on the raw matrix


def load_sample(big_tar, member, workdir):
    """Extract one tumor's raw 10x matrix, read it, call cells, label sample."""
    sample = member.split('_', 1)[1].split('.')[0]  # e.g. Pat01-B
    inner = Path(workdir) / 'inner'
    inner.mkdir(exist_ok=True)
    big_tar.extract(member, inner)
    inner_tar = inner / member
    exdir = Path(workdir) / 'ex'
    exdir.mkdir(exist_ok=True)
    with tarfile.open(inner_tar, 'r:gz') as it:
        it.extractall(exdir)
    mtx = next(exdir.rglob('matrix.mtx'))
    a = sc.read_10x_mtx(mtx.parent, var_names='gene_symbols', cache=False)
    a.var_names_make_unique()
    sc.pp.filter_cells(a, min_genes=MIN_GENES_CELLCALL)  # drop empty droplets
    a.obs['sample'] = sample
    a.obs_names = [f'{sample}_{bc}' for bc in a.obs_names]
    print(f'  {sample}: {a.n_obs} cells after cell-calling')
    shutil.rmtree(inner); shutil.rmtree(exdir)
    return a


def load():
    with tarfile.open(BIG) as big:
        tumor_members = sorted(n for n in big.getnames()
                               if n.endswith('.raw_gene_bc_matrices.tar.gz') and '-B.' in n)
        print(f'{len(tumor_members)} tumor samples')
        adatas = []
        with tempfile.TemporaryDirectory() as tmp:
            for m in tumor_members:
                adatas.append(load_sample(big, m, tmp))
    adata = ad.concat(adatas, join='inner', merge='same')
    adata.var_names_make_unique()
    print(f'Combined: {adata.n_obs} cells x {adata.n_vars} genes')
    for g in ['TACSTD2', 'VTCN1', 'EPCAM']:
        print(f'  gene present: {g} = {g in adata.var_names}')
    return adata


def main():
    print('=== Load (Sathe raw 10x, cell-call) ===')
    adata = load()
    print('\n=== QC ==='); adata = qc(adata)
    print('\n=== Normalize ==='); adata = normalize(adata)
    print('\n=== Embed ==='); adata = embed(adata)
    print('\n=== Annotate ==='); adata = annotate(adata)
    out = OUTDIR / 'sathe_tumor.h5ad'
    adata.write(out)
    print(f'\nSaved {out}  ({adata.n_obs} cells)')


if __name__ == '__main__':
    main()

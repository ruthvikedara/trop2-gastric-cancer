#!/usr/bin/env python
"""Preprocess expression matrices for xCell."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from genolib import RAW_DIR, PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


#  GEO preprocessing

def preprocess_geo(
    accession: str,
    cancer_type: str,
    *,
    dry_run: bool = False,
) -> Path:
    """Download a GEO dataset, map probes to genes, and export TSV.

    Steps:
      1. Download / cache SOFT file via GEOparse
      2. Extract probe-level expression matrix
      3. Map probes -> gene symbols using platform GPL annotation
      4. Aggregate duplicate genes (mean)
      5. Detect and apply log2 transform if needed
      6. Export to ``data/processed/{cancer_type}/``

    Returns:
        Path to the exported TSV file.
    """
    import GEOparse

    dest = str(RAW_DIR / cancer_type)

    logger.info(f"Fetching {accession} via GEOparse (destdir={dest})")
    gse = GEOparse.get_GEO(accession, destdir=dest, silent=True)

    platform_ids = list(gse.gpls.keys())
    n_samples = len(gse.gsms)
    logger.info(f"  Platform: {platform_ids}, Samples: {n_samples}")

    if dry_run:
        logger.info("[dry-run] Would process %d samples on platform %s",
                     n_samples, platform_ids)
        return Path()

    # --- expression matrix (probes x samples) ---
    expression_df = gse.pivot_samples('VALUE')
    logger.info(f"  Raw matrix: {expression_df.shape}")

    # --- probe -> gene mapping from GPL annotation ---
    gpl = list(gse.gpls.values())[0]
    table = gpl.table

    # Try common column names for gene symbol
    gene_col = None
    for candidate in ('Gene Symbol', 'gene_assignment', 'GENE_SYMBOL', 'Symbol'):
        if candidate in table.columns:
            gene_col = candidate
            break

    if gene_col is None:
        logger.error("Cannot find gene symbol column in GPL table. "
                      f"Available columns: {table.columns.tolist()}")
        raise RuntimeError("Probe-to-gene mapping failed")

    probe_to_gene = table[['ID', gene_col]].copy()
    probe_to_gene.columns = ['probe_id', 'gene_symbol']
    probe_to_gene = probe_to_gene.dropna()
    probe_to_gene = probe_to_gene[
        ~probe_to_gene['gene_symbol'].isin(['', '---'])
    ]
    # Take first symbol when multiple are joined with ' /// '
    probe_to_gene['gene_symbol'] = (
        probe_to_gene['gene_symbol'].str.split(' /// ').str[0]
    )
    logger.info(f"  Probes with gene symbols: {len(probe_to_gene)}")

    # --- merge and aggregate ---
    expression_df = expression_df.reset_index()
    expression_df.columns = ['probe_id'] + list(expression_df.columns[1:])
    merged = expression_df.merge(probe_to_gene, on='probe_id', how='inner')
    merged = merged.drop('probe_id', axis=1).set_index('gene_symbol')
    gene_expr = merged.groupby(merged.index).mean()

    # --- log2 if needed ---
    median_val = gene_expr.median().median()
    if median_val > 20:
        logger.info(f"  Median={median_val:.1f} -> applying log2(x+1)")
        gene_expr = np.log2(gene_expr.clip(lower=0) + 1)
    else:
        logger.info(f"  Median={median_val:.2f} -> data appears already log-scaled")

    # --- export ---
    out_dir = PROCESSED_DIR / cancer_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ACRG_{accession}_for_xcell.txt"
    gene_expr.to_csv(out_path, sep='\t')

    logger.info(f"Exported {gene_expr.shape[0]} genes x {gene_expr.shape[1]} "
                f"samples -> {out_path}")
    return out_path


#  TCGA preprocessing

def preprocess_tcga(
    accession: str,
    cancer_type: str,
    *,
    rsem_path: str | None = None,
    dry_run: bool = False,
) -> Path:
    """Preprocess a TCGA RSEM file for xCell.

    Steps:
      1. Load RSEM expression file
      2. Drop Entrez_Gene_Id column
      3. Log2(x+1) transform if data is in linear scale
      4. Aggregate duplicate genes (mean)
      5. Filter to tumor samples only (barcode -01)
      6. Trim patient IDs to 12 characters
      7. Export to ``data/processed/{cancer_type}/``

    Args:
        accession: TCGA project ID (e.g. ``TCGA-STAD``)
        cancer_type: Cancer type folder name
        rsem_path: Path to RSEM file.  Defaults to
            ``data/raw/{cancer_type}/data_mrna_seq_v2_rsem.txt``
        dry_run: If True, only show what would happen

    Returns:
        Path to the exported TSV file.
    """
    if rsem_path is None:
        rsem_path = RAW_DIR / cancer_type / 'data_mrna_seq_v2_rsem.txt'
    rsem_path = Path(rsem_path)

    if not rsem_path.exists():
        raise FileNotFoundError(f"RSEM file not found: {rsem_path}")

    logger.info(f"Loading TCGA RSEM data from {rsem_path}")
    tcga = pd.read_csv(rsem_path, sep='\t', index_col='Hugo_Symbol')

    if dry_run:
        n_tumor = sum(1 for c in tcga.columns if c.endswith('-01'))
        logger.info(f"[dry-run] {tcga.shape[0]} genes, ~{n_tumor} tumor samples")
        return Path()

    # Drop Entrez_Gene_Id if present
    if 'Entrez_Gene_Id' in tcga.columns:
        tcga = tcga.drop(columns=['Entrez_Gene_Id'])

    # Log2 if needed
    med = tcga.median().median()
    if med > 100:
        logger.info(f"  Median={med:.1f} -> applying log2(x+1)")
        tcga = np.log2(tcga + 1)
    else:
        logger.info(f"  Median={med:.2f} -> data appears already log-scaled")

    # Deduplicate genes
    tcga = tcga.groupby(tcga.index).mean()

    # Keep only tumor samples (-01)
    tumor_cols = [c for c in tcga.columns if c.endswith('-01')]
    logger.info(f"  Keeping {len(tumor_cols)} tumor samples "
                f"(dropped {len(tcga.columns) - len(tumor_cols)} non-tumor)")
    tcga = tcga[tumor_cols]

    # Trim IDs to patient-level (first 12 chars)
    tcga.columns = [c[:12] for c in tcga.columns]

    # Export
    project_short = accession.replace('TCGA-', '')
    out_dir = PROCESSED_DIR / cancer_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"TCGA_{project_short}_for_xcell.txt"
    tcga.to_csv(out_path, sep='\t')

    logger.info(f"Exported {tcga.shape[0]} genes x {tcga.shape[1]} samples "
                f"-> {out_path}")
    return out_path


#  CLI

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess raw expression data for xCell deconvolution',
    )
    parser.add_argument(
        '--accession', '-a', required=True,
        help='Dataset accession (e.g. GSE62254, TCGA-STAD)',
    )
    parser.add_argument(
        '--cancer', '-c', required=True,
        help='Cancer type (e.g. gastric)',
    )
    parser.add_argument(
        '--rsem-path', default=None,
        help='(TCGA only) Path to RSEM file',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be done without writing files',
    )

    args = parser.parse_args()

    if args.accession.startswith('TCGA-'):
        preprocess_tcga(
            args.accession, args.cancer,
            rsem_path=args.rsem_path,
            dry_run=args.dry_run,
        )
    elif args.accession.startswith('GSE'):
        preprocess_geo(
            args.accession, args.cancer,
            dry_run=args.dry_run,
        )
    else:
        logger.error(f"Unknown accession format: {args.accession}")
        sys.exit(1)


if __name__ == '__main__':
    main()

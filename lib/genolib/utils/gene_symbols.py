"""Gene symbol harmonization."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class GeneMapper:
    """Handles gene ID mapping and symbol harmonization."""
    
    def __init__(self):
        self._probe_maps: Dict[str, pd.DataFrame] = {}
        self._alias_map: Optional[Dict[str, str]] = None
    
    def load_probe_mapping(self, platform: str) -> pd.DataFrame:
        """Load probe-to-gene mapping for a platform.
        
        Args:
            platform: Platform identifier (e.g., 'affymetrix_hgu133plus2')
            
        Returns:
            DataFrame with 'probe_id' and 'gene_symbol' columns
        """
        if platform in self._probe_maps:
            return self._probe_maps[platform]
        
        from genolib import CONFIG_DIR
        mapping_dir = CONFIG_DIR / 'platform_mappings'
        mapping_file = mapping_dir / f'{platform}.tsv'
        
        if not mapping_file.exists():
            logger.warning(f"No mapping file for platform {platform}, will need to generate")
            return pd.DataFrame(columns=['probe_id', 'gene_symbol'])
        
        mapping = pd.read_csv(mapping_file, sep='\t')
        self._probe_maps[platform] = mapping
        
        return mapping
    
    def map_probes_to_genes(
        self,
        expression_df: pd.DataFrame,
        platform: str,
        aggregation: str = 'mean'
    ) -> pd.DataFrame:
        """Convert probe-level expression to gene-level.
        
        Args:
            expression_df: Expression matrix with probe IDs as index
            platform: Platform identifier for mapping
            aggregation: How to handle multiple probes per gene ('mean', 'max', 'median')
            
        Returns:
            Expression matrix with gene symbols as index
        """
        mapping = self.load_probe_mapping(platform)
        
        if mapping.empty:
            logger.warning("No probe mapping available, returning original data")
            return expression_df
        
        # Merge expression with mapping
        expr_reset = expression_df.reset_index()
        expr_reset.columns = ['probe_id'] + list(expression_df.columns)
        
        merged = expr_reset.merge(mapping[['probe_id', 'gene_symbol']], on='probe_id', how='left')
        
        # Remove unmapped probes
        unmapped = merged['gene_symbol'].isna().sum()
        if unmapped > 0:
            logger.info(f"Dropping {unmapped} unmapped probes")
        
        merged = merged.dropna(subset=['gene_symbol'])
        
        # Aggregate multiple probes per gene
        sample_cols = [c for c in merged.columns if c not in ['probe_id', 'gene_symbol']]
        
        if aggregation == 'mean':
            gene_expr = merged.groupby('gene_symbol')[sample_cols].mean()
        elif aggregation == 'max':
            gene_expr = merged.groupby('gene_symbol')[sample_cols].max()
        elif aggregation == 'median':
            gene_expr = merged.groupby('gene_symbol')[sample_cols].median()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        logger.info(f"Mapped {len(expression_df)} probes to {len(gene_expr)} genes")
        
        return gene_expr
    
    def resolve_alias(self, gene: str) -> str:
        """Resolve a gene alias to the canonical symbol.
        
        Args:
            gene: Gene symbol or alias
            
        Returns:
            Canonical gene symbol
        """
        # Load gene targets config for aliases
        from genolib.utils.io import load_config
        
        try:
            config = load_config('gene_targets')
        except FileNotFoundError:
            return gene
        
        # Build alias map if not cached
        if self._alias_map is None:
            self._alias_map = {}
            for category in config.get('targets', {}).values():
                for gene_symbol, info in category.items():
                    self._alias_map[gene_symbol.upper()] = gene_symbol
                    for alias in info.get('aliases', []):
                        self._alias_map[alias.upper()] = gene_symbol
        
        return self._alias_map.get(gene.upper(), gene)
    
    def find_gene_in_matrix(
        self,
        expression_df: pd.DataFrame,
        gene: str
    ) -> Optional[str]:
        """Find a gene in an expression matrix, handling aliases.
        
        Args:
            expression_df: Expression matrix with gene symbols as index
            gene: Gene to find (symbol or alias)
            
        Returns:
            Matching gene symbol from the matrix, or None
        """
        genes_upper = {g.upper(): g for g in expression_df.index}
        
        # Try exact match first
        if gene in expression_df.index:
            return gene
        
        if gene.upper() in genes_upper:
            return genes_upper[gene.upper()]
        
        # Try resolving alias
        canonical = self.resolve_alias(gene)
        if canonical.upper() in genes_upper:
            return genes_upper[canonical.upper()]
        
        # Try known aliases from config
        from genolib.utils.io import load_config
        try:
            config = load_config('gene_targets')
            for category in config.get('targets', {}).values():
                for gene_symbol, info in category.items():
                    if gene.upper() == gene_symbol.upper() or gene.upper() in [a.upper() for a in info.get('aliases', [])]:
                        # Found config entry, check all aliases against matrix
                        for alias in [gene_symbol] + info.get('aliases', []):
                            if alias.upper() in genes_upper:
                                return genes_upper[alias.upper()]
        except FileNotFoundError:
            pass
        
        logger.warning(f"Gene {gene} not found in expression matrix")
        return None


def harmonize_gene_symbols(
    expression_df: pd.DataFrame,
    reference_genes: Optional[List[str]] = None
) -> pd.DataFrame:
    """Standardize gene symbols in an expression matrix.
    
    Args:
        expression_df: Expression matrix with gene symbols as index
        reference_genes: Optional list of reference genes to filter to
        
    Returns:
        Expression matrix with harmonized gene symbols
    """
    # Basic cleanup
    df = expression_df.copy()
    
    # Remove any suffix after | or _ (common in some formats)
    df.index = df.index.str.split('|').str[0]
    df.index = df.index.str.split('_').str[0]
    
    # Uppercase for consistency
    df.index = df.index.str.upper()
    
    # Remove duplicates (keep first)
    df = df[~df.index.duplicated(keep='first')]
    
    # Filter to reference genes if provided
    if reference_genes is not None:
        reference_upper = [g.upper() for g in reference_genes]
        common = df.index.intersection(reference_upper)
        logger.info(f"Keeping {len(common)} of {len(reference_upper)} reference genes")
        df = df.loc[common]
    
    return df


def get_common_genes(*expression_dfs: pd.DataFrame) -> List[str]:
    """Find genes common to all expression matrices.
    
    Args:
        *expression_dfs: Variable number of expression DataFrames
        
    Returns:
        List of common gene symbols
    """
    if not expression_dfs:
        return []
    
    common = set(expression_dfs[0].index)
    
    for df in expression_dfs[1:]:
        common = common.intersection(df.index)
    
    logger.info(f"Found {len(common)} genes common to {len(expression_dfs)} datasets")
    
    return sorted(list(common))


# Singleton instance for convenience
gene_mapper = GeneMapper()

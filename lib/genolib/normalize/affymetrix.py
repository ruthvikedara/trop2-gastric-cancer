"""Affymetrix normalization."""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from .base import BaseNormalizer

logger = logging.getLogger(__name__)


class AffymetrixNormalizer(BaseNormalizer):
    """Normalize Affymetrix microarray data."""
    
    SUPPORTED_PLATFORMS = [
        'affymetrix_hgu133plus2',
        'affymetrix_hgu133a',
        'affymetrix_hgu133b',
        'GPL570',  # HG-U133 Plus 2.0
        'GPL96',   # HG-U133A
        'GPL97',   # HG-U133B
    ]
    
    def __init__(
        self,
        cancer_type: str,
        dataset_id: str,
        platform: str = 'affymetrix_hgu133plus2'
    ):
        """Initialize Affymetrix normalizer.
        
        Args:
            cancer_type: Cancer type
            dataset_id: Dataset accession
            platform: Specific Affymetrix platform
        """
        super().__init__(cancer_type, dataset_id)
        self.platform = platform
    
    @property
    def output_suffix(self) -> str:
        return 'log2rma'
    
    def normalize(self, expression_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Affymetrix microarray data.
        
        GEO series matrix files are typically already RMA-normalized and log2-transformed.
        This method verifies the scale and applies additional QC.
        
        Args:
            expression_df: Expression matrix (probes × samples)
            
        Returns:
            Normalized expression matrix (genes × samples)
        """
        df = expression_df.copy()
        
        # Check if data is log-transformed
        if not self._is_log_transformed(df):
            logger.info("Data not log-transformed, applying log2")
            # Handle potential zeros or negative values
            df = df.clip(lower=0.01)
            df = np.log2(df)
        else:
            logger.info("Data appears already log2-transformed")
        
        # Quantile normalization if needed
        if not self._is_quantile_normalized(df):
            logger.info("Applying quantile normalization")
            df = self._quantile_normalize(df)
        
        # Map probes to genes
        df = self._map_probes_to_genes(df)
        
        # Remove duplicates
        df = self.remove_duplicate_genes(df, keep='max')
        
        logger.info(f"Final matrix: {df.shape[0]} genes × {df.shape[1]} samples")
        
        return df
    
    def _is_log_transformed(self, df: pd.DataFrame) -> bool:
        """Check if data appears to be log-transformed.
        
        Log2 transformed microarray data typically has values in range 2-15.
        """
        median_val = df.median().median()
        max_val = df.max().max()
        
        # Log2 data: median around 6-10, max around 15
        # Linear data: median in hundreds/thousands
        return median_val < 20 and max_val < 20
    
    def _is_quantile_normalized(self, df: pd.DataFrame) -> bool:
        """Check if data appears quantile normalized.
        
        After quantile normalization, all samples have similar distributions.
        """
        # Compare distributions across samples
        sample_medians = df.median()
        sample_stds = df.std()
        
        # If medians and stds are very similar across samples, likely normalized
        median_cv = sample_medians.std() / sample_medians.mean()
        std_cv = sample_stds.std() / sample_stds.mean()
        
        return median_cv < 0.05 and std_cv < 0.1
    
    def _quantile_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quantile normalization.
        
        Forces all samples to have the same distribution.
        """
        # Rank the data
        rank_mean = df.stack().groupby(
            df.rank(method='first').stack().astype(int)
        ).mean()
        
        # Apply the average to the ranks
        normalized = df.rank(method='min').stack().astype(int).map(rank_mean).unstack()
        
        return normalized
    
    def _map_probes_to_genes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map probe IDs to gene symbols.
        
        Args:
            df: Expression matrix with probe IDs as index
            
        Returns:
            Expression matrix with gene symbols as index
        """
        from genolib.utils.gene_symbols import gene_mapper
        
        # Try to load platform-specific mapping
        try:
            mapped = gene_mapper.map_probes_to_genes(
                df, 
                platform=self.platform,
                aggregation='mean'
            )
            
            if len(mapped) > 0:
                return mapped
        except Exception as e:
            logger.warning(f"Probe mapping failed: {e}")
        
        # If mapping fails, check if index already looks like gene symbols
        sample_ids = df.index[:10].tolist()
        
        # Probe IDs typically look like: 1007_s_at, 200000_s_at, AFFX-...
        # Gene symbols look like: TP53, BRCA1, EGFR
        if any('_at' in str(idx) or 'AFFX' in str(idx) for idx in sample_ids):
            logger.warning("Data has probe IDs but no mapping available. "
                          "Please add mapping file to config/platform_mappings/")
            return df
        else:
            logger.info("Index appears to be gene symbols already")
            return df


def normalize_affymetrix(
    cancer_type: str,
    dataset_id: str,
    expression_df: pd.DataFrame,
    platform: str = 'affymetrix_hgu133plus2'
) -> pd.DataFrame:
    """Convenience function to normalize Affymetrix data.
    
    Args:
        cancer_type: Cancer type
        dataset_id: Dataset accession
        expression_df: Raw expression matrix
        platform: Affymetrix platform
        
    Returns:
        Normalized expression matrix
    """
    normalizer = AffymetrixNormalizer(cancer_type, dataset_id, platform)
    return normalizer.normalize(expression_df)

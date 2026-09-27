"""RNA-seq normalization (counts to log2 TPM)."""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

from .base import BaseNormalizer

logger = logging.getLogger(__name__)


class RNAseqNormalizer(BaseNormalizer):
    """Normalize RNA-seq count data to log2(TPM+1)."""
    
    def __init__(
        self,
        cancer_type: str,
        dataset_id: str,
        gene_lengths: Optional[pd.Series] = None
    ):
        """Initialize RNA-seq normalizer.
        
        Args:
            cancer_type: Cancer type
            dataset_id: Dataset accession
            gene_lengths: Gene lengths for TPM calculation (gene symbol → length in kb)
        """
        super().__init__(cancer_type, dataset_id)
        self.gene_lengths = gene_lengths
    
    @property
    def output_suffix(self) -> str:
        return 'log2tpm'
    
    def normalize(self, expression_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize RNA-seq data.
        
        If input appears to be counts, converts to TPM then log2.
        If input appears to be TPM already, just log2 transforms.
        
        Args:
            expression_df: Expression matrix (genes × samples)
            
        Returns:
            Log2(TPM+1) normalized matrix
        """
        df = expression_df.copy()
        
        # Determine if data is counts or already normalized
        if self._is_count_data(df):
            logger.info("Detected count data, converting to TPM")
            df = self._counts_to_tpm(df)
        else:
            logger.info("Data appears pre-normalized, checking scale")
            # If values are very large, might be TPM that needs log transform
            # If values are small (0-15 range), might already be log2
            if df.max().max() > 100:
                logger.info("Values suggest linear scale TPM")
            else:
                logger.info("Values suggest already log-transformed, skipping log2")
                return df
        
        # Log2 transform
        logger.info("Applying log2(x+1) transformation")
        df = self.log2_transform(df, pseudocount=1.0)
        
        # Filter and deduplicate
        df = self.filter_low_expression(df, min_samples=3, min_value=0)
        df = self.remove_duplicate_genes(df, keep='max')
        
        logger.info(f"Final matrix: {df.shape[0]} genes × {df.shape[1]} samples")
        
        return df
    
    def _is_count_data(self, df: pd.DataFrame) -> bool:
        """Heuristic to detect if data is raw counts.
        
        Counts are typically:
        - Integers (or close to it)
        - Large values (hundreds to millions)
        - Sum per sample varies widely
        """
        # Check if values are approximately integers
        sample = df.iloc[:1000, :10].values.flatten()
        sample = sample[~np.isnan(sample)]
        
        frac_integer = np.mean(np.abs(sample - np.round(sample)) < 0.01)
        
        # Check magnitude
        max_val = df.max().max()
        
        # Counts: integer-ish, large values
        if frac_integer > 0.9 and max_val > 1000:
            return True
        
        return False
    
    def _counts_to_tpm(self, counts_df: pd.DataFrame) -> pd.DataFrame:
        """Convert raw counts to TPM.
        
        TPM = (reads_i / length_i) / sum(reads_j / length_j) * 1e6
        
        If gene lengths not available, uses a simplified RPM calculation.
        """
        if self.gene_lengths is not None:
            # Full TPM calculation with gene lengths
            common_genes = counts_df.index.intersection(self.gene_lengths.index)
            
            if len(common_genes) < len(counts_df) * 0.5:
                logger.warning("Less than 50% genes have length info, using RPM instead")
                return self._counts_to_rpm(counts_df)
            
            counts = counts_df.loc[common_genes]
            lengths = self.gene_lengths.loc[common_genes]
            
            # Divide by length (in kb)
            rpk = counts.div(lengths, axis=0)
            
            # Scale to TPM
            tpm = rpk.div(rpk.sum(axis=0), axis=1) * 1e6
            
            return tpm
        else:
            logger.warning("No gene lengths provided, using simplified RPM normalization")
            return self._counts_to_rpm(counts_df)
    
    def _counts_to_rpm(self, counts_df: pd.DataFrame) -> pd.DataFrame:
        """Convert counts to RPM (Reads Per Million).
        
        Simplified normalization when gene lengths unavailable.
        """
        rpm = counts_df.div(counts_df.sum(axis=0), axis=1) * 1e6
        return rpm


class TCGANormalizer(RNAseqNormalizer):
    """Specialized normalizer for TCGA data.
    
    TCGA data from GDC often comes with TPM already calculated.
    """
    
    def normalize(self, expression_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize TCGA RNA-seq data.
        
        Args:
            expression_df: Expression matrix (may already be TPM)
            
        Returns:
            Log2(TPM+1) normalized matrix
        """
        df = expression_df.copy()
        
        # TCGA data from GDC typically has TPM already
        # Just need to log2 transform if not already done
        
        max_val = df.max().max()
        
        if max_val > 100:
            # Linear scale, needs log2
            logger.info("TCGA data in linear scale, applying log2(x+1)")
            df = self.log2_transform(df, pseudocount=1.0)
        else:
            logger.info("TCGA data appears already log-transformed")
        
        # Clean up gene names (remove version numbers)
        df.index = df.index.str.split('.').str[0]
        
        # Filter and deduplicate
        df = self.filter_low_expression(df, min_samples=3, min_value=0)
        df = self.remove_duplicate_genes(df, keep='max')
        
        return df

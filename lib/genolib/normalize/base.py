"""Base normalizer."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseNormalizer(ABC):
    """Abstract base class for expression data normalization."""
    
    def __init__(self, cancer_type: str, dataset_id: str):
        """Initialize normalizer.
        
        Args:
            cancer_type: Cancer type
            dataset_id: Dataset accession
        """
        self.cancer_type = cancer_type
        self.dataset_id = dataset_id
    
    @abstractmethod
    def normalize(self, expression_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize expression data.
        
        Args:
            expression_df: Raw expression matrix
            
        Returns:
            Normalized expression matrix (log2 scale)
        """
        pass
    
    def process_and_save(self, expression_df: pd.DataFrame) -> Path:
        """Normalize and save expression data.
        
        Args:
            expression_df: Raw expression matrix
            
        Returns:
            Path to saved file
        """
        normalized = self.normalize(expression_df)
        
        from genolib.utils import save_expression_matrix
        return save_expression_matrix(
            normalized,
            self.cancer_type,
            self.dataset_id,
            suffix=self.output_suffix,
            stage='processed'
        )
    
    @property
    @abstractmethod
    def output_suffix(self) -> str:
        """Suffix for output filename (e.g., 'log2tpm')."""
        pass
    
    @staticmethod
    def log2_transform(df: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
        """Apply log2 transformation with pseudocount.
        
        Args:
            df: Expression matrix
            pseudocount: Value to add before log (default 1.0)
            
        Returns:
            Log2-transformed matrix
        """
        return np.log2(df + pseudocount)
    
    @staticmethod
    def filter_low_expression(
        df: pd.DataFrame,
        min_samples: int = 3,
        min_value: float = 1.0
    ) -> pd.DataFrame:
        """Filter out lowly expressed genes.
        
        Args:
            df: Expression matrix
            min_samples: Minimum samples with expression above threshold
            min_value: Minimum expression value
            
        Returns:
            Filtered expression matrix
        """
        mask = (df >= min_value).sum(axis=1) >= min_samples
        n_removed = (~mask).sum()
        
        if n_removed > 0:
            logger.info(f"Filtered {n_removed} lowly expressed genes")
        
        return df.loc[mask]
    
    @staticmethod
    def remove_duplicate_genes(
        df: pd.DataFrame,
        keep: str = 'max'
    ) -> pd.DataFrame:
        """Remove duplicate gene entries.
        
        Args:
            df: Expression matrix with gene symbols as index
            keep: How to handle duplicates ('max', 'mean', 'first')
            
        Returns:
            Deduplicated expression matrix
        """
        if not df.index.duplicated().any():
            return df
        
        n_dups = df.index.duplicated().sum()
        logger.info(f"Handling {n_dups} duplicate gene entries (method: {keep})")
        
        if keep == 'first':
            return df[~df.index.duplicated(keep='first')]
        elif keep == 'max':
            return df.groupby(df.index).max()
        elif keep == 'mean':
            return df.groupby(df.index).mean()
        else:
            raise ValueError(f"Unknown keep method: {keep}")

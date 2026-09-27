"""GEO downloader."""

import gzip
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
import urllib.request
import tempfile
import shutil

from .base import BaseDownloader

logger = logging.getLogger(__name__)


class GEODownloader(BaseDownloader):
    """Download and parse GEO datasets."""
    
    GEO_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"
    
    def __init__(self, cancer_type: str, dataset_id: str):
        super().__init__(cancer_type, dataset_id)
        
        if not dataset_id.startswith('GSE'):
            raise ValueError(f"GEO dataset ID must start with 'GSE', got: {dataset_id}")
        
        self._expression_matrix: Optional[pd.DataFrame] = None
        self._clinical_data: Optional[pd.DataFrame] = None
    
    @property
    def series_path(self) -> str:
        """Get the FTP path component for this series."""
        # GSE62254 -> GSE62nnn
        series_stub = self.dataset_id[:-3] + 'nnn'
        return f"{series_stub}/{self.dataset_id}"
    
    def _get_series_matrix_url(self) -> str:
        """Get URL for the series matrix file."""
        return f"{self.GEO_FTP_BASE}/{self.series_path}/matrix/{self.dataset_id}_series_matrix.txt.gz"
    
    def _get_soft_url(self) -> str:
        """Get URL for the SOFT format family file."""
        return f"{self.GEO_FTP_BASE}/{self.series_path}/soft/{self.dataset_id}_family.soft.gz"
    
    def download(self, force: bool = False) -> Path:
        """Download GEO dataset.
        
        Downloads the series matrix file (expression + minimal metadata)
        and optionally the full SOFT file for detailed annotations.
        
        Args:
            force: Re-download even if exists
            
        Returns:
            Path to download directory
        """
        if self.is_downloaded() and not force:
            logger.info(f"{self.dataset_id} already downloaded, skipping (use force=True to re-download)")
            return self.output_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download series matrix (main file we need)
        matrix_url = self._get_series_matrix_url()
        matrix_file = self.output_dir / f"{self.dataset_id}_series_matrix.txt.gz"
        
        logger.info(f"Downloading series matrix from {matrix_url}")
        try:
            urllib.request.urlretrieve(matrix_url, matrix_file)
            logger.info(f"Downloaded to {matrix_file}")
        except Exception as e:
            logger.error(f"Failed to download series matrix: {e}")
            raise
        
        # Save metadata
        from genolib.utils import save_metadata
        metadata = {
            'dataset_id': self.dataset_id,
            'source': 'geo',
            'download_url': matrix_url,
            'platform': self.config.get('platform', 'unknown'),
            'cancer_type': self.cancer_type
        }
        save_metadata(metadata, self.cancer_type, self.dataset_id, stage='raw')
        
        return self.output_dir
    
    def _parse_series_matrix(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Parse the series matrix file into expression and clinical data.
        
        Returns:
            Tuple of (expression_matrix, clinical_data)
        """
        matrix_file = self.output_dir / f"{self.dataset_id}_series_matrix.txt.gz"
        
        if not matrix_file.exists():
            raise FileNotFoundError(f"Series matrix not found: {matrix_file}. Run download() first.")
        
        # Read the file and separate metadata from expression data
        metadata_lines = []
        expression_lines = []
        in_expression = False
        
        with gzip.open(matrix_file, 'rt', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith('!series_matrix_table_begin'):
                    in_expression = True
                    continue
                elif line.startswith('!series_matrix_table_end'):
                    in_expression = False
                    continue
                
                if in_expression:
                    expression_lines.append(line)
                elif line.startswith('!Sample_'):
                    metadata_lines.append(line)
        
        # Parse expression matrix
        if expression_lines:
            from io import StringIO
            expr_text = ''.join(expression_lines)
            expression_df = pd.read_csv(StringIO(expr_text), sep='\t', index_col=0)
            expression_df.index.name = 'probe_id'
        else:
            expression_df = pd.DataFrame()
        
        # Parse clinical/sample metadata
        clinical_data = self._parse_sample_metadata(metadata_lines)
        
        return expression_df, clinical_data
    
    def _parse_sample_metadata(self, metadata_lines: list) -> pd.DataFrame:
        """Parse sample metadata from series matrix header.
        
        Args:
            metadata_lines: Lines starting with !Sample_
            
        Returns:
            Clinical data DataFrame (samples × attributes)
        """
        metadata_dict = {}
        
        for line in metadata_lines:
            # Parse lines like: !Sample_geo_accession\t"GSM123"\t"GSM456"
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            
            key = parts[0].replace('!Sample_', '')
            values = [v.strip('"') for v in parts[1:]]
            metadata_dict[key] = values
        
        if not metadata_dict:
            return pd.DataFrame()
        
        # Convert to DataFrame
        clinical_df = pd.DataFrame(metadata_dict)
        
        # Use geo_accession as index if available
        if 'geo_accession' in clinical_df.columns:
            clinical_df.set_index('geo_accession', inplace=True)
        
        return clinical_df
    
    def get_expression_matrix(self) -> pd.DataFrame:
        """Get the expression matrix.
        
        Returns:
            Expression matrix (probes/genes × samples)
        """
        if self._expression_matrix is None:
            self._expression_matrix, self._clinical_data = self._parse_series_matrix()
        
        return self._expression_matrix
    
    def get_clinical_data(self) -> pd.DataFrame:
        """Get clinical/phenotype data.
        
        Returns:
            Clinical data DataFrame
        """
        if self._clinical_data is None:
            self._expression_matrix, self._clinical_data = self._parse_series_matrix()
        
        return self._clinical_data
    
    def get_platform_id(self) -> str:
        """Extract the platform ID (GPL*) from the dataset.
        
        Returns:
            Platform accession (e.g., 'GPL570')
        """
        clinical = self.get_clinical_data()
        
        if 'platform_id' in clinical.columns:
            return clinical['platform_id'].iloc[0]
        
        # Try to get from config
        return self.config.get('platform', 'unknown')


def download_geo_dataset(cancer_type: str, dataset_id: str, force: bool = False) -> Path:
    """Convenience function to download a GEO dataset.
    
    Args:
        cancer_type: Cancer type
        dataset_id: GEO accession (GSE*)
        force: Re-download if exists
        
    Returns:
        Path to downloaded data
    """
    downloader = GEODownloader(cancer_type, dataset_id)
    return downloader.download(force=force)

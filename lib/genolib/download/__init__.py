"""Data download modules for various sources."""

from .base import BaseDownloader
from .geo import GEODownloader, download_geo_dataset
from .tcga import TCGADownloader, download_tcga_dataset


def get_downloader(cancer_type: str, dataset_id: str) -> BaseDownloader:
    """Factory function to get appropriate downloader for a dataset.
    
    Args:
        cancer_type: Cancer type
        dataset_id: Dataset accession
        
    Returns:
        Appropriate downloader instance
    """
    if dataset_id.startswith('GSE'):
        return GEODownloader(cancer_type, dataset_id)
    elif dataset_id.startswith('TCGA-'):
        return TCGADownloader(cancer_type, dataset_id)
    else:
        raise ValueError(f"Unknown dataset type for: {dataset_id}")


__all__ = [
    'BaseDownloader',
    'GEODownloader',
    'TCGADownloader',
    'download_geo_dataset',
    'download_tcga_dataset',
    'get_downloader'
]

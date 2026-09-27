"""Normalization modules for different platforms."""

from .base import BaseNormalizer
from .rnaseq import RNAseqNormalizer, TCGANormalizer
from .affymetrix import AffymetrixNormalizer, normalize_affymetrix


def get_normalizer(cancer_type: str, dataset_id: str) -> BaseNormalizer:
    """Factory function to get appropriate normalizer for a dataset.
    
    Args:
        cancer_type: Cancer type
        dataset_id: Dataset accession
        
    Returns:
        Appropriate normalizer instance
    """
    from genolib.utils import get_dataset_config
    
    config = get_dataset_config(cancer_type, dataset_id)
    platform = config.get('platform', '').lower()
    
    if dataset_id.startswith('TCGA-'):
        return TCGANormalizer(cancer_type, dataset_id)
    elif 'affymetrix' in platform or platform.startswith('gpl'):
        return AffymetrixNormalizer(cancer_type, dataset_id, platform)
    elif 'illumina' in platform and 'humanht' in platform:
        # Illumina BeadArray - similar to Affymetrix for processing
        return AffymetrixNormalizer(cancer_type, dataset_id, platform)
    elif 'rnaseq' in platform or 'rna-seq' in platform:
        return RNAseqNormalizer(cancer_type, dataset_id)
    else:
        # Default to RNA-seq normalizer for unknown platforms
        return RNAseqNormalizer(cancer_type, dataset_id)


__all__ = [
    'BaseNormalizer',
    'RNAseqNormalizer', 
    'TCGANormalizer',
    'AffymetrixNormalizer',
    'normalize_affymetrix',
    'get_normalizer'
]

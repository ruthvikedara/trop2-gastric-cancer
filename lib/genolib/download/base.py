"""Base downloader."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseDownloader(ABC):
    """Abstract base class for data source downloaders."""
    
    def __init__(self, cancer_type: str, dataset_id: str):
        """Initialize downloader.
        
        Args:
            cancer_type: Cancer type (e.g., 'gastric')
            dataset_id: Dataset accession (e.g., 'GSE62254')
        """
        self.cancer_type = cancer_type
        self.dataset_id = dataset_id
        self._config: Optional[Dict[str, Any]] = None
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        if self._config is None:
            from genolib.utils import get_dataset_config
            self._config = get_dataset_config(self.cancer_type, self.dataset_id)
        return self._config
    
    @property
    def output_dir(self) -> Path:
        """Get output directory for raw data."""
        from genolib.utils import get_data_path
        return get_data_path(self.cancer_type, self.dataset_id, stage='raw')
    
    @abstractmethod
    def download(self, force: bool = False) -> Path:
        """Download the dataset.
        
        Args:
            force: If True, re-download even if data exists
            
        Returns:
            Path to downloaded data directory
        """
        pass
    
    @abstractmethod
    def get_expression_matrix(self) -> 'pd.DataFrame':
        """Extract the expression matrix from downloaded data.
        
        Returns:
            Expression matrix (genes × samples)
        """
        pass
    
    @abstractmethod
    def get_clinical_data(self) -> 'pd.DataFrame':
        """Extract clinical/phenotype data.
        
        Returns:
            Clinical data DataFrame
        """
        pass
    
    def is_downloaded(self) -> bool:
        """Check if data has already been downloaded."""
        return self.output_dir.exists() and any(self.output_dir.iterdir())
    
    def validate(self) -> bool:
        """Validate downloaded data integrity.
        
        Returns:
            True if data is valid
        """
        if not self.is_downloaded():
            return False
        
        # Subclasses can add more specific validation
        return True

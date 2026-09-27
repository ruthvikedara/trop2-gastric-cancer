"""TCGA downloader (GDC)."""

import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import logging

from .base import BaseDownloader

logger = logging.getLogger(__name__)


class TCGADownloader(BaseDownloader):
    """Download TCGA data from GDC Portal."""
    
    GDC_API_BASE = "https://api.gdc.cancer.gov"
    GDC_FILES_ENDPOINT = f"{GDC_API_BASE}/files"
    GDC_DATA_ENDPOINT = f"{GDC_API_BASE}/data"
    
    def __init__(self, cancer_type: str, dataset_id: str):
        super().__init__(cancer_type, dataset_id)
        
        if not dataset_id.startswith('TCGA-'):
            raise ValueError(f"TCGA dataset ID must start with 'TCGA-', got: {dataset_id}")
        
        self.project_id = dataset_id
        self._manifest: Optional[List[Dict]] = None
    
    def _query_gdc_files(
        self,
        data_type: str = "Gene Expression Quantification",
        workflow_type: str = "STAR - Counts",
        data_format: str = "TSV",
        size: int = 1000
    ) -> List[Dict]:
        """Query GDC for available files.
        
        Args:
            data_type: Type of data to query
            workflow_type: Analysis workflow
            data_format: File format
            size: Max number of results
            
        Returns:
            List of file metadata dictionaries
        """
        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": [self.project_id]}},
                {"op": "in", "content": {"field": "data_type", "value": [data_type]}},
                {"op": "in", "content": {"field": "analysis.workflow_type", "value": [workflow_type]}},
                {"op": "in", "content": {"field": "data_format", "value": [data_format]}}
            ]
        }
        
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.case_id,cases.submitter_id,cases.samples.sample_type",
            "format": "JSON",
            "size": str(size)
        }
        
        response = requests.get(self.GDC_FILES_ENDPOINT, params=params)
        response.raise_for_status()
        
        data = response.json()
        return data.get('data', {}).get('hits', [])
    
    def _download_file(self, file_id: str, output_path: Path) -> Path:
        """Download a single file from GDC.
        
        Args:
            file_id: GDC file UUID
            output_path: Where to save the file
            
        Returns:
            Path to downloaded file
        """
        url = f"{self.GDC_DATA_ENDPOINT}/{file_id}"
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return output_path
    
    def download(self, force: bool = False) -> Path:
        """Download TCGA dataset from GDC.
        
        This downloads individual sample files. For large datasets,
        consider using the GDC Data Transfer Tool instead.
        
        Args:
            force: Re-download even if exists
            
        Returns:
            Path to download directory
        """
        if self.is_downloaded() and not force:
            logger.info(f"{self.dataset_id} already downloaded, skipping")
            return self.output_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        expr_dir = self.output_dir / 'expression'
        expr_dir.mkdir(exist_ok=True)
        
        # Query available files
        logger.info(f"Querying GDC for {self.project_id} files...")
        files = self._query_gdc_files()
        
        if not files:
            raise ValueError(f"No files found for project {self.project_id}")
        
        logger.info(f"Found {len(files)} files to download")
        
        # Save manifest
        manifest_path = self.output_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(files, f, indent=2)
        
        # Download files (with progress)
        for i, file_info in enumerate(files):
            file_id = file_info['file_id']
            file_name = file_info['file_name']
            output_path = expr_dir / file_name
            
            if output_path.exists() and not force:
                continue
            
            logger.info(f"Downloading {i+1}/{len(files)}: {file_name}")
            
            try:
                self._download_file(file_id, output_path)
            except Exception as e:
                logger.error(f"Failed to download {file_name}: {e}")
                continue
        
        # Save metadata
        from genolib.utils import save_metadata
        metadata = {
            'dataset_id': self.dataset_id,
            'source': 'gdc',
            'project_id': self.project_id,
            'n_files': len(files),
            'cancer_type': self.cancer_type
        }
        save_metadata(metadata, self.cancer_type, self.dataset_id, stage='raw')
        
        return self.output_dir
    
    def get_expression_matrix(self) -> pd.DataFrame:
        """Combine individual sample files into expression matrix.
        
        Returns:
            Expression matrix (genes × samples)
        """
        expr_dir = self.output_dir / 'expression'
        
        if not expr_dir.exists():
            raise FileNotFoundError(f"Expression directory not found: {expr_dir}")
        
        # Find all TSV files
        tsv_files = list(expr_dir.glob('*.tsv'))
        
        if not tsv_files:
            raise FileNotFoundError(f"No TSV files found in {expr_dir}")
        
        logger.info(f"Combining {len(tsv_files)} expression files...")
        
        # Read and combine
        expression_data = {}
        
        for tsv_file in tsv_files:
            # TCGA files have columns: gene_id, gene_name, gene_type, various counts
            # We want TPM (tpm_unstranded column) or raw counts (unstranded)
            try:
                df = pd.read_csv(tsv_file, sep='\t', comment='#')
                
                # Extract sample ID from filename or manifest
                sample_id = tsv_file.stem.split('.')[0]
                
                # Get TPM values (prefer) or counts
                if 'tpm_unstranded' in df.columns:
                    values = df.set_index('gene_name')['tpm_unstranded']
                elif 'unstranded' in df.columns:
                    values = df.set_index('gene_name')['unstranded']
                else:
                    logger.warning(f"Unknown format in {tsv_file}, skipping")
                    continue
                
                expression_data[sample_id] = values
                
            except Exception as e:
                logger.warning(f"Error reading {tsv_file}: {e}")
                continue
        
        # Combine into matrix
        expression_matrix = pd.DataFrame(expression_data)
        expression_matrix.index.name = 'gene_symbol'
        
        logger.info(f"Created matrix: {expression_matrix.shape[0]} genes × {expression_matrix.shape[1]} samples")
        
        return expression_matrix
    
    def get_clinical_data(self) -> pd.DataFrame:
        """Download and return clinical data.
        
        Returns:
            Clinical data DataFrame
        """
        # Query clinical data from GDC
        filters = {
            "op": "in",
            "content": {
                "field": "project.project_id",
                "value": [self.project_id]
            }
        }
        
        fields = [
            "case_id",
            "submitter_id",
            "demographic.gender",
            "demographic.vital_status",
            "demographic.days_to_death",
            "diagnoses.tumor_stage",
            "diagnoses.age_at_diagnosis",
            "diagnoses.primary_diagnosis"
        ]
        
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "format": "JSON",
            "size": "1000"
        }
        
        response = requests.get(f"{self.GDC_API_BASE}/cases", params=params)
        response.raise_for_status()
        
        data = response.json().get('data', {}).get('hits', [])
        
        # Flatten nested structure
        clinical_records = []
        for case in data:
            record = {
                'case_id': case.get('case_id'),
                'submitter_id': case.get('submitter_id')
            }
            
            # Add demographic info
            demo = case.get('demographic', {})
            record['gender'] = demo.get('gender')
            record['vital_status'] = demo.get('vital_status')
            record['days_to_death'] = demo.get('days_to_death')
            
            # Add diagnosis info (take first if multiple)
            diagnoses = case.get('diagnoses', [{}])
            if diagnoses:
                diag = diagnoses[0]
                record['tumor_stage'] = diag.get('tumor_stage')
                record['age_at_diagnosis'] = diag.get('age_at_diagnosis')
                record['primary_diagnosis'] = diag.get('primary_diagnosis')
            
            clinical_records.append(record)
        
        return pd.DataFrame(clinical_records)


def download_tcga_dataset(cancer_type: str, dataset_id: str, force: bool = False) -> Path:
    """Convenience function to download a TCGA dataset.
    
    Args:
        cancer_type: Cancer type
        dataset_id: TCGA project ID (e.g., 'TCGA-STAD')
        force: Re-download if exists
        
    Returns:
        Path to downloaded data
    """
    downloader = TCGADownloader(cancer_type, dataset_id)
    return downloader.download(force=force)

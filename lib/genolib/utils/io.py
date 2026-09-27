"""File I/O helpers."""

import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_name: str) -> Dict[str, Any]:
    """Load a YAML configuration file from the config directory.
    
    Args:
        config_name: Name of config file (with or without .yaml extension)
        
    Returns:
        Dictionary containing configuration
    """
    from genolib import CONFIG_DIR
    
    if not config_name.endswith('.yaml'):
        config_name += '.yaml'
    
    config_path = CONFIG_DIR / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_dataset_config(cancer_type: str, dataset_id: str) -> Dict[str, Any]:
    """Get configuration for a specific dataset.
    
    Args:
        cancer_type: Cancer type (e.g., 'gastric')
        dataset_id: Dataset accession (e.g., 'GSE62254')
        
    Returns:
        Dataset configuration dictionary
    """
    datasets = load_config('datasets')
    
    if cancer_type not in datasets:
        raise ValueError(f"Unknown cancer type: {cancer_type}. Available: {list(datasets.keys())}")
    
    if dataset_id not in datasets[cancer_type]:
        raise ValueError(f"Unknown dataset: {dataset_id}. Available for {cancer_type}: {list(datasets[cancer_type].keys())}")
    
    return datasets[cancer_type][dataset_id]


def get_data_path(cancer_type: str, dataset_id: str, stage: str = 'raw') -> Path:
    """Get the path for a dataset at a given processing stage.
    
    Args:
        cancer_type: Cancer type
        dataset_id: Dataset accession
        stage: Processing stage ('raw', 'processed', 'xcell_output')
        
    Returns:
        Path to the dataset directory or file
    """
    from genolib import RAW_DIR, PROCESSED_DIR, XCELL_DIR
    
    stage_dirs = {
        'raw': RAW_DIR,
        'processed': PROCESSED_DIR,
        'xcell_output': XCELL_DIR
    }
    
    if stage not in stage_dirs:
        raise ValueError(f"Unknown stage: {stage}. Available: {list(stage_dirs.keys())}")
    
    return stage_dirs[stage] / cancer_type / dataset_id


def save_expression_matrix(
    df: pd.DataFrame,
    cancer_type: str,
    dataset_id: str,
    suffix: str = 'log2tpm',
    stage: str = 'processed'
) -> Path:
    """Save an expression matrix to parquet format.
    
    Args:
        df: Expression matrix (genes x samples)
        cancer_type: Cancer type
        dataset_id: Dataset accession
        suffix: Descriptive suffix (e.g., 'log2tpm', 'log2rma')
        stage: Processing stage
        
    Returns:
        Path to saved file
    """
    from genolib import PROCESSED_DIR, XCELL_DIR
    
    if stage == 'processed':
        base_dir = PROCESSED_DIR
    elif stage == 'xcell_output':
        base_dir = XCELL_DIR
    else:
        raise ValueError(f"Cannot save to stage: {stage}")
    
    out_dir = base_dir / cancer_type
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / f"{dataset_id}_{suffix}.parquet"
    df.to_parquet(out_path)
    
    logger.info(f"Saved expression matrix to {out_path}")
    return out_path


def _resolve_dataset_filename(dataset_id: str) -> str:
    """Map a dataset ID to the filename prefix used on disk.

    Examples:
        GSE62254  -> ACRG_GSE62254
        TCGA-STAD -> TCGA_STAD
    """
    if dataset_id.startswith('TCGA-'):
        return dataset_id.replace('-', '_')
    # GEO datasets use ACRG_ prefix
    return f"ACRG_{dataset_id}"


def load_expression_matrix(
    cancer_type: str,
    dataset_id: str,
    stage: str = 'processed'
) -> pd.DataFrame:
    """Load an expression matrix from processed data files.

    Searches for parquet files first, then falls back to TSV (.txt) files.

    Args:
        cancer_type: Cancer type
        dataset_id: Dataset accession
        stage: Processing stage

    Returns:
        Expression matrix as DataFrame (genes x samples)
    """
    from genolib import PROCESSED_DIR, XCELL_DIR

    if stage == 'processed':
        base_dir = PROCESSED_DIR
    elif stage == 'xcell_output':
        base_dir = XCELL_DIR
    else:
        raise ValueError(f"Cannot load from stage: {stage}")

    data_dir = base_dir / cancer_type

    # Build search patterns: try both raw dataset_id and resolved filename prefix
    prefixes = [dataset_id, _resolve_dataset_filename(dataset_id)]

    # Search for parquet first, then txt
    for ext in ('*.parquet', '*.txt'):
        for prefix in prefixes:
            matches = list(data_dir.glob(f"{prefix}_*{ext[1:]}"))
            if matches:
                path = matches[0]
                if len(matches) > 1:
                    logger.warning(f"Multiple files found for {dataset_id}, using: {path.name}")
                if path.suffix == '.parquet':
                    return pd.read_parquet(path)
                else:
                    df = pd.read_csv(path, sep='\t', index_col=0)
                    return df

    raise FileNotFoundError(
        f"No processed data found for {dataset_id} in {data_dir}. "
        f"Searched prefixes: {prefixes}"
    )


def list_available_datasets(cancer_type: str, stage: str = 'processed') -> list:
    """List datasets available at a given processing stage.
    
    Args:
        cancer_type: Cancer type
        stage: Processing stage
        
    Returns:
        List of dataset IDs
    """
    from genolib import PROCESSED_DIR, XCELL_DIR
    
    if stage == 'processed':
        base_dir = PROCESSED_DIR
    elif stage == 'xcell_output':
        base_dir = XCELL_DIR
    else:
        from genolib import RAW_DIR
        base_dir = RAW_DIR
    
    data_dir = base_dir / cancer_type
    
    if not data_dir.exists():
        return []
    
    if stage == 'raw':
        # Raw data is in subdirectories
        return [d.name for d in data_dir.iterdir() if d.is_dir()]
    else:
        # Processed data may be parquet or txt
        files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.txt"))
        # Extract dataset ID: filenames like ACRG_GSE62254_for_xcell.txt -> GSE62254
        #                      or TCGA_STAD_for_xcell.txt -> TCGA-STAD
        dataset_ids = set()
        for f in files:
            name = f.stem
            if name.startswith('ACRG_'):
                # Extract GSE ID after ACRG_ prefix
                parts = name[5:].split('_', 1)
                dataset_ids.add(parts[0])
            elif name.startswith('TCGA_'):
                parts = name.split('_', 2)
                dataset_ids.add(f"{parts[0]}-{parts[1]}")
            else:
                dataset_ids.add(name.split('_')[0])
        return list(dataset_ids)


def filter_normal_samples(
    cancer_type: str,
    dataset_id: str,
    expression_df: pd.DataFrame,
    xcell_df: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, tuple]:
    """Remove normal tissue samples from GEO datasets.

    Uses the dataset config's ``n_normal`` field to decide if filtering is
    needed, then downloads GEO metadata via GEOparse to identify normal
    sample IDs by checking the ``title`` and ``source_name_ch1`` fields.

    Args:
        cancer_type: Cancer type
        dataset_id: Dataset accession
        expression_df: Expression matrix (genes x samples)
        xcell_df: Optional xCell scores DataFrame (cell types x samples)

    Returns:
        Filtered expression_df if xcell_df is None, else
        tuple of (filtered expression_df, filtered xcell_df).
    """
    try:
        config = get_dataset_config(cancer_type, dataset_id)
    except ValueError:
        config = {}

    n_normal = config.get('n_normal', 0) or 0
    if n_normal == 0:
        return (expression_df, xcell_df) if xcell_df is not None else expression_df

    # For TCGA: normals have barcode element [3][:2] in ('10','11','12')
    if dataset_id.startswith('TCGA-'):
        tumor_cols = []
        for col in expression_df.columns:
            parts = col.split('-')
            if len(parts) >= 4:
                sample_type = parts[3][:2]
                if sample_type in ('01', '02', '03', '04', '05', '06', '09'):
                    tumor_cols.append(col)
            else:
                tumor_cols.append(col)

        n_removed = len(expression_df.columns) - len(tumor_cols)
        if n_removed > 0:
            logger.info(f"{dataset_id}: removed {n_removed} normal samples")
        expression_df = expression_df[tumor_cols]
        if xcell_df is not None:
            xcell_cols = [c for c in tumor_cols if c in xcell_df.columns]
            xcell_df = xcell_df[xcell_cols]

        return (expression_df, xcell_df) if xcell_df is not None else expression_df

    # For GEO datasets: use GEOparse to identify normals
    try:
        import GEOparse
    except ImportError:
        logger.warning("GEOparse not installed, cannot filter normal samples")
        return (expression_df, xcell_df) if xcell_df is not None else expression_df

    from genolib import RAW_DIR
    destdir = str(RAW_DIR / cancer_type)

    try:
        gse = GEOparse.get_GEO(dataset_id, destdir=destdir, silent=True)
        meta = gse.phenotype_data
        normal_mask = (
            meta['title'].str.contains('normal', case=False, na=False) |
            meta.get('source_name_ch1', pd.Series(dtype=str)).str.contains(
                'normal', case=False, na=False
            )
        )
        normal_ids = set(meta[normal_mask].index.tolist())
    except Exception as e:
        logger.warning(f"Could not fetch metadata for {dataset_id}: {e}")
        return (expression_df, xcell_df) if xcell_df is not None else expression_df

    if not normal_ids:
        return (expression_df, xcell_df) if xcell_df is not None else expression_df

    before = expression_df.shape[1]
    keep = [c for c in expression_df.columns if c not in normal_ids]
    expression_df = expression_df[keep]
    logger.info(f"{dataset_id}: removed {before - len(keep)} normal samples "
                f"({before} -> {len(keep)})")

    if xcell_df is not None:
        keep_x = [c for c in xcell_df.columns if c not in normal_ids]
        xcell_df = xcell_df[keep_x]
        return expression_df, xcell_df

    return expression_df


def save_metadata(
    metadata: Dict[str, Any],
    cancer_type: str,
    dataset_id: str,
    stage: str = 'raw'
) -> Path:
    """Save dataset metadata as JSON.
    
    Args:
        metadata: Metadata dictionary
        cancer_type: Cancer type
        dataset_id: Dataset accession
        stage: Processing stage
        
    Returns:
        Path to saved file
    """
    data_path = get_data_path(cancer_type, dataset_id, stage)
    data_path.mkdir(parents=True, exist_ok=True)
    
    out_path = data_path / 'metadata.json'
    
    with open(out_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return out_path

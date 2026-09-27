"""xCell deconvolution wrapper."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import logging
import subprocess
import tempfile

logger = logging.getLogger(__name__)


def run_xcell_r(
    expression_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Run xCell using R via rpy2.
    
    Args:
        expression_df: Log2-transformed expression matrix (genes × samples)
        output_path: Optional path to save results
        
    Returns:
        xCell scores DataFrame (cell types × samples)
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        
        pandas2ri.activate()
        
        # Load xCell
        xcell = importr('xCell')
        
        # Convert to R dataframe
        with (ro.default_converter + pandas2ri.converter).context():
            r_expr = ro.conversion.get_conversion().py2rpy(expression_df)
        
        # Run xCell
        logger.info(f"Running xCell on {expression_df.shape[1]} samples...")
        result = xcell.xCellAnalysis(r_expr)
        
        # Convert back to pandas
        with (ro.default_converter + pandas2ri.converter).context():
            xcell_df = ro.conversion.get_conversion().rpy2py(result)
        
        if output_path:
            xcell_df.to_parquet(output_path)
            logger.info(f"Saved xCell results to {output_path}")
        
        return xcell_df
        
    except ImportError:
        logger.error("rpy2 not available. Install with: pip install rpy2")
        logger.info("Falling back to R script method...")
        return run_xcell_script(expression_df, output_path)


def run_xcell_script(
    expression_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Run xCell via R script (fallback when rpy2 not available).
    
    Args:
        expression_df: Expression matrix
        output_path: Where to save results
        
    Returns:
        xCell scores DataFrame
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save expression matrix
        expr_file = tmpdir / 'expression.tsv'
        expression_df.to_csv(expr_file, sep='\t')
        
        # Output file
        out_file = tmpdir / 'xcell_results.tsv'
        
        # R script
        r_script = f'''
library(xCell)

# Read expression data
expr <- read.table("{expr_file}", header=TRUE, row.names=1, sep="\\t", check.names=FALSE)
expr <- as.matrix(expr)

# Run xCell
result <- xCellAnalysis(expr)

# Save results
write.table(result, "{out_file}", sep="\\t", quote=FALSE)
'''
        
        script_file = tmpdir / 'run_xcell.R'
        with open(script_file, 'w') as f:
            f.write(r_script)
        
        # Run R script
        logger.info("Running xCell via R script...")
        try:
            result = subprocess.run(
                ['Rscript', str(script_file)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"R script failed: {result.stderr}")
                raise RuntimeError(f"xCell R script failed: {result.stderr}")
            
            # Read results
            xcell_df = pd.read_csv(out_file, sep='\t', index_col=0)
            
            if output_path:
                xcell_df.to_parquet(output_path)
            
            return xcell_df
            
        except FileNotFoundError:
            logger.error("Rscript not found. Please install R and xCell package.")
            raise


def run_xcell_api(
    expression_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Run xCell via web API (if available).
    
    Note: This is a placeholder. xCell doesn't have a public API,
    but this could be implemented for a local server setup.
    """
    raise NotImplementedError(
        "xCell API not available. Use run_xcell_r() or run_xcell_script() instead."
    )


def run_xcell(
    expression_df: pd.DataFrame,
    cancer_type: str,
    dataset_id: str,
    method: str = 'auto'
) -> pd.DataFrame:
    """Run xCell deconvolution and save results.
    
    This is the main entry point for xCell analysis.
    
    Args:
        expression_df: Normalized expression matrix (genes × samples)
        cancer_type: Cancer type for organizing output
        dataset_id: Dataset ID for filename
        method: Method to use ('auto', 'rpy2', 'script')
        
    Returns:
        xCell scores DataFrame
    """
    from genolib import XCELL_DIR
    
    # Output path
    out_dir = XCELL_DIR / cancer_type
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{dataset_id}_xcell.parquet"
    
    # Check if already exists
    if output_path.exists():
        logger.info(f"Loading existing xCell results from {output_path}")
        return pd.read_parquet(output_path)
    
    # Run xCell
    if method == 'auto':
        try:
            return run_xcell_r(expression_df, output_path)
        except Exception as e:
            logger.warning(f"rpy2 method failed: {e}, trying script method")
            return run_xcell_script(expression_df, output_path)
    elif method == 'rpy2':
        return run_xcell_r(expression_df, output_path)
    elif method == 'script':
        return run_xcell_script(expression_df, output_path)
    else:
        raise ValueError(f"Unknown method: {method}")


def load_xcell_results(cancer_type: str, dataset_id: str) -> pd.DataFrame:
    """Load previously computed xCell results.

    Searches for parquet files first, then TSV (.txt) files whose name
    contains the dataset ID. Handles the TCGA dot-vs-dash column name
    issue (``TCGA.3M.AB46`` -> ``TCGA-3M-AB46``).

    Args:
        cancer_type: Cancer type
        dataset_id: Dataset ID

    Returns:
        xCell scores DataFrame (cell types x samples)
    """
    from genolib import XCELL_DIR
    from genolib.utils.io import _resolve_dataset_filename

    xcell_dir = XCELL_DIR / cancer_type

    # Try parquet first (legacy path)
    parquet_path = xcell_dir / f"{dataset_id}_xcell.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    # Search for TSV files matching the dataset
    prefix = _resolve_dataset_filename(dataset_id)
    matches = list(xcell_dir.glob(f"{prefix}_*"))
    if not matches:
        # Broader fallback: any file containing the dataset ID
        matches = list(xcell_dir.glob(f"*{dataset_id}*"))

    if not matches:
        raise FileNotFoundError(
            f"xCell results not found for {dataset_id} in {xcell_dir}"
        )

    path = matches[0]
    if len(matches) > 1:
        logger.warning(f"Multiple xCell files for {dataset_id}, using: {path.name}")

    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep='\t', index_col=0)

    # Fix TCGA dot-to-dash column names (R's check.names converts - to .)
    if dataset_id.startswith('TCGA'):
        df.columns = df.columns.str.replace('.', '-', regex=False)

    return df


# Cell type categories for interpretation
XCELL_CELL_CATEGORIES = {
    'T_cells': [
        'T cell CD4+', 'T cell CD4+ (non-regulatory)', 'T cell CD4+ central memory',
        'T cell CD4+ effector memory', 'T cell CD4+ memory', 'T cell CD4+ naive',
        'T cell CD4+ Th1', 'T cell CD4+ Th2', 'T cell CD8+', 'T cell CD8+ central memory',
        'T cell CD8+ effector memory', 'T cell CD8+ naive', 'T cell gamma delta',
        'T cell NK', 'T cell regulatory (Tregs)'
    ],
    'B_cells': [
        'B cell', 'B cell memory', 'B cell naive', 'B cell plasma',
        'Class-switched memory B cell'
    ],
    'Myeloid': [
        'Macrophage', 'Macrophage M1', 'Macrophage M2', 'Monocyte',
        'Myeloid dendritic cell', 'Myeloid dendritic cell activated',
        'Plasmacytoid dendritic cell', 'Neutrophil', 'Eosinophil', 'Mast cell'
    ],
    'NK_cells': ['NK cell'],
    'Progenitors': [
        'Common lymphoid progenitor', 'Common myeloid progenitor',
        'Granulocyte-monocyte progenitor', 'Hematopoietic stem cell'
    ],
    'Stromal': [
        'Cancer associated fibroblast', 'Endothelial cell'
    ],
    'Scores': [
        'immune score', 'stroma score', 'microenvironment score'
    ]
}

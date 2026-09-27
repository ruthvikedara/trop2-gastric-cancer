"""Shared download, normalization, deconvolution, analysis and plotting utilities."""

__version__ = "0.1.0"
__author__ = "Ruthvik"

from pathlib import Path

# Repo root: lib/genolib/__init__.py -> parents[2] == repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Central data store (shared across all projects)
DATA_DIR = PROJECT_ROOT / "data"
GEO_DIR = DATA_DIR / "geo"
TCGA_DIR = DATA_DIR / "tcga"
RAW_DIR = DATA_DIR / "raw"               # legacy alias; kept for back-compat
PROCESSED_DIR = DATA_DIR / "processed"
CLINICAL_DIR = DATA_DIR / "clinical"
XCELL_DIR = DATA_DIR / "xcell_output"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
SHARE_DIR = DATA_DIR / "share"

# Shared pipeline config lives with the library (lib/genolib/config)
CONFIG_DIR = Path(__file__).resolve().parent / "config"


# Back-compat: generic pipeline CLIs (run_analysis/run_full_pipeline) write here.
# Target/paper scripts use their own per-project plots/ instead.
RESULTS_DIR = PROJECT_ROOT / "results"


def results_dir(project: str) -> Path:
    """Per-project results directory (results are NOT shared across projects)."""
    d = PROJECT_ROOT / project / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Ensure shared input dirs exist (cheap, idempotent). Results are per-project.
for dir_path in [DATA_DIR, PROCESSED_DIR, XCELL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

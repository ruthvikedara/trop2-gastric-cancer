"""Visualization modules for immune deconvolution results."""

from .dotplot import create_dotplot, create_summary_table
from .heatmap import create_xcell_heatmap, create_combined_heatmap
from .boxplot import create_boxplot_grid

__all__ = [
    'create_dotplot',
    'create_summary_table',
    'create_xcell_heatmap',
    'create_combined_heatmap',
    'create_boxplot_grid',
]


# Set default plotting style
import matplotlib.pyplot as plt

def set_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False
    })

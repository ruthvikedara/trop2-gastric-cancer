"""xCell heatmaps."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


def create_xcell_heatmap(
    xcell_df: pd.DataFrame,
    patient_groups: Dict[str, List[str]],
    gene: str,
    figsize: Tuple[float, float] = (14, 8),
    scale: str = 'row',
    cmap: str = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Create a clustered heatmap of xCell scores with patient annotations.
    
    Args:
        xcell_df: xCell results (cell types × samples)
        patient_groups: Dict from stratify_patients() with 'high' and 'low' keys
        gene: Gene name for annotation
        figsize: Figure size
        scale: Scaling method ('row', 'column', or None)
        cmap: Colormap (default: blue-white-red)
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    # Filter to patients in groups
    all_patients = patient_groups.get('high', []) + patient_groups.get('low', [])
    available_patients = [p for p in all_patients if p in xcell_df.columns]
    
    if len(available_patients) < 5:
        logger.warning(f"Only {len(available_patients)} patients available, insufficient for heatmap")
        return None
    
    # Subset data
    data = xcell_df[available_patients].copy()
    
    # Remove summary scores for cleaner visualization
    exclude_rows = ['immune score', 'stroma score', 'microenvironment score']
    data = data.loc[~data.index.str.lower().isin([e.lower() for e in exclude_rows])]
    
    # Remove rows with all zeros or NaN
    data = data.loc[(data != 0).any(axis=1)]
    data = data.dropna(how='all')
    
    # Scale data
    if scale == 'row':
        data = data.subtract(data.mean(axis=1), axis=0).divide(data.std(axis=1), axis=0)
        data = data.fillna(0)
        # Clip extreme values
        data = data.clip(-3, 3)
    elif scale == 'column':
        data = data.subtract(data.mean(axis=0), axis=1).divide(data.std(axis=0), axis=1)
        data = data.fillna(0)
        data = data.clip(-3, 3)
    
    # Create annotation for columns (patients)
    col_colors = []
    high_set = set(patient_groups.get('high', []))
    low_set = set(patient_groups.get('low', []))
    
    for patient in available_patients:
        if patient in high_set:
            col_colors.append('#B22222')  # firebrick for high
        elif patient in low_set:
            col_colors.append('#1E90FF')  # dodgerblue for low
        else:
            col_colors.append('#808080')  # gray for mid/unknown
    
    # Custom colormap
    if cmap is None:
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['#1E90FF', '#1E90FF', 'white', '#B22222', '#B22222']
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
    
    # Create clustermap
    g = sns.clustermap(
        data,
        figsize=figsize,
        col_colors=col_colors,
        cmap=cmap,
        center=0,
        vmin=-3 if scale else None,
        vmax=3 if scale else None,
        xticklabels=False,
        yticklabels=True,
        dendrogram_ratio=(0.1, 0.15),
        colors_ratio=0.02,
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        method='ward',
        metric='euclidean'
    )
    
    # Adjust y-axis labels
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), 
        fontsize=7,
        rotation=0
    )
    
    # Add legend for patient groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#B22222', label=f'{gene}-high'),
        Patch(facecolor='#1E90FF', label=f'{gene}-low')
    ]
    g.ax_heatmap.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.15),
        fontsize=9,
        frameon=False
    )
    
    # Title
    g.fig.suptitle(f'xCell Immune Deconvolution - {gene}', y=1.02, fontsize=12)
    
    # Save if requested
    if save_path:
        g.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved heatmap to {save_path}")
        
        # Also save as PDF
        if not save_path.endswith('.pdf'):
            pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
            g.savefig(pdf_path, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return g.fig


def create_combined_heatmap(
    xcell_dfs: Dict[str, pd.DataFrame],
    patient_groups_dict: Dict[str, Dict[str, List[str]]],
    gene: str,
    figsize: Tuple[float, float] = (16, 10),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Create a combined heatmap from multiple cohorts.
    
    Args:
        xcell_dfs: Dict of cohort_name → xCell DataFrame
        patient_groups_dict: Dict of cohort_name → patient_groups dict
        gene: Gene name
        figsize: Figure size
        save_path: Path to save
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    # Combine all xCell data
    all_data = []
    all_annotations = []
    
    for cohort_name, xcell_df in xcell_dfs.items():
        if cohort_name not in patient_groups_dict:
            continue
        
        groups = patient_groups_dict[cohort_name]
        all_patients = groups.get('high', []) + groups.get('low', [])
        available = [p for p in all_patients if p in xcell_df.columns]
        
        if not available:
            continue
        
        # Subset and rename columns to include cohort
        subset = xcell_df[available].copy()
        subset.columns = [f"{cohort_name}_{c}" for c in subset.columns]
        
        all_data.append(subset)
        
        # Annotations
        for patient in available:
            all_annotations.append({
                'patient': f"{cohort_name}_{patient}",
                'cohort': cohort_name,
                'group': 'high' if patient in groups.get('high', []) else 'low'
            })
    
    if not all_data:
        logger.warning("No data to combine for heatmap")
        return None
    
    # Concatenate
    combined = pd.concat(all_data, axis=1)
    annotations_df = pd.DataFrame(all_annotations).set_index('patient')
    
    # Remove summary scores
    exclude_rows = ['immune score', 'stroma score', 'microenvironment score']
    combined = combined.loc[~combined.index.str.lower().isin([e.lower() for e in exclude_rows])]
    
    # Row-scale
    combined = combined.subtract(combined.mean(axis=1), axis=0).divide(combined.std(axis=1), axis=0)
    combined = combined.fillna(0).clip(-3, 3)
    
    # Colors for annotation
    col_colors_group = annotations_df['group'].map({
        'high': '#B22222',
        'low': '#1E90FF'
    }).values
    
    # Create unique colors for cohorts
    cohorts = annotations_df['cohort'].unique()
    cohort_palette = sns.color_palette('husl', len(cohorts))
    cohort_color_map = dict(zip(cohorts, cohort_palette))
    col_colors_cohort = annotations_df['cohort'].map(cohort_color_map).values
    
    # Stack colors
    col_colors = pd.DataFrame({
        f'{gene} group': col_colors_group,
        'Cohort': col_colors_cohort
    }, index=combined.columns)
    
    # Create colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#1E90FF', '#1E90FF', 'white', '#B22222', '#B22222']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
    
    # Create clustermap
    g = sns.clustermap(
        combined,
        figsize=figsize,
        col_colors=col_colors,
        cmap=cmap,
        center=0,
        vmin=-3,
        vmax=3,
        xticklabels=False,
        yticklabels=True,
        dendrogram_ratio=(0.08, 0.12),
        colors_ratio=0.03,
        method='ward',
        metric='euclidean'
    )
    
    # Styling
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        fontsize=6,
        rotation=0
    )
    
    g.fig.suptitle(f'Combined xCell Analysis - {gene}', y=1.02, fontsize=12)
    
    # Save
    if save_path:
        g.savefig(save_path, dpi=300, bbox_inches='tight')
        if not save_path.endswith('.pdf'):
            g.savefig(save_path.rsplit('.', 1)[0] + '.pdf', bbox_inches='tight')
    
    if show:
        plt.show()
    
    return g.fig

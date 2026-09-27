"""Cross-cohort dot plots."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def create_dotplot(
    results_df: pd.DataFrame,
    gene: str,
    figsize: Tuple[float, float] = (12, 5),
    cluster_cells: bool = True,
    cluster_cohorts: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Create a dot plot showing cell type changes across cohorts.
    
    This recreates the R visualization showing:
    - X-axis: Cell types (optionally clustered)
    - Y-axis: Cohorts (optionally clustered)
    - Color: Direction (red = increased in high, blue = decreased in high)
    - Size: Absolute t-statistic
    - Opacity: Significance (solid = p<0.05, faint = n.s.)
    
    Args:
        results_df: Results from run_multi_cohort_analysis()
        gene: Gene name for title
        figsize: Figure size
        cluster_cells: Hierarchically cluster cell types
        cluster_cohorts: Hierarchically cluster cohorts
        save_path: Path to save figure (optional)
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    if results_df.empty:
        logger.warning("Empty results DataFrame, cannot create plot")
        return None
    
    # Prepare data
    df = results_df.copy()
    
    # Create significance column for display
    df['sig_label'] = df['significant'].map({True: 'p<0.05', False: 'n.s.'})
    
    # Create direction label
    df['direction_label'] = df.apply(
        lambda r: f"increased in {gene}-high GC" if r['direction'] == 'increased' 
                  else f"decreased in {gene}-high GC",
        axis=1
    )
    
    # Pivot to matrix for clustering
    t_stat_matrix = df.pivot(index='cohort', columns='cell_type', values='t_statistic')
    t_stat_matrix = t_stat_matrix.fillna(0)
    
    # Hierarchical clustering
    if cluster_cells and t_stat_matrix.shape[1] > 2:
        cell_linkage = linkage(pdist(t_stat_matrix.T, metric='euclidean'), method='ward')
        cell_order = leaves_list(cell_linkage)
        cell_types_ordered = t_stat_matrix.columns[cell_order].tolist()
    else:
        cell_types_ordered = t_stat_matrix.columns.tolist()
    
    if cluster_cohorts and t_stat_matrix.shape[0] > 2:
        cohort_linkage = linkage(pdist(t_stat_matrix, metric='euclidean'), method='ward')
        cohort_order = leaves_list(cohort_linkage)
        cohorts_ordered = t_stat_matrix.index[cohort_order].tolist()
    else:
        cohorts_ordered = t_stat_matrix.index.tolist()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color mapping
    colors = {
        f"increased in {gene}-high GC": '#B22222',  # firebrick
        f"decreased in {gene}-high GC": '#1E90FF'   # dodgerblue
    }
    
    # Plot points
    for _, row in df.iterrows():
        x = cell_types_ordered.index(row['cell_type'])
        y = cohorts_ordered.index(row['cohort'])
        
        color = colors.get(row['direction_label'], 'gray')
        alpha = 1.0 if row['significant'] else 0.2
        size = abs(row['t_statistic']) * 50  # Scale for visibility
        
        ax.scatter(x, y, c=color, s=size, alpha=alpha, edgecolors='none')
    
    # Styling
    ax.set_xticks(range(len(cell_types_ordered)))
    ax.set_xticklabels(cell_types_ordered, rotation=90, ha='center', fontsize=8)
    
    ax.set_yticks(range(len(cohorts_ordered)))
    ax.set_yticklabels(cohorts_ordered, fontsize=10)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Grid
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#B22222', label=f'increased in {gene}-high GC'),
        mpatches.Patch(color='#1E90FF', label=f'decreased in {gene}-high GC'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=10, alpha=1.0, label='p<0.05'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=10, alpha=0.2, label='n.s.')
    ]
    
    # Size legend
    for t_val in [1, 3, 5]:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                      markersize=np.sqrt(t_val * 50 / np.pi) * 2, 
                      label=f'|t|={t_val}')
        )
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=8, frameon=False)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved dot plot to {save_path}")
        
        # Also save as PDF
        if not save_path.endswith('.pdf'):
            pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
            fig.savefig(pdf_path, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def create_summary_table(
    results_df: pd.DataFrame,
    gene: str
) -> pd.DataFrame:
    """Create a summary table of significant findings.
    
    Args:
        results_df: Results DataFrame
        gene: Gene analyzed
        
    Returns:
        Summary DataFrame with consistent findings highlighted
    """
    if results_df.empty:
        return pd.DataFrame()
    
    # Count consistent directions across cohorts
    summary = results_df.groupby('cell_type').agg({
        'direction': lambda x: x.mode().iloc[0] if len(x) > 0 else 'mixed',
        'significant': 'sum',
        't_statistic': ['mean', 'std'],
        'cohort': 'count'
    }).reset_index()
    
    summary.columns = ['cell_type', 'predominant_direction', 'n_significant', 
                       't_stat_mean', 't_stat_std', 'n_cohorts']
    
    # Check consistency
    direction_counts = results_df.groupby(['cell_type', 'direction']).size().unstack(fill_value=0)
    
    summary['consistency'] = summary['cell_type'].apply(
        lambda ct: 'consistent' if (
            ct in direction_counts.index and
            direction_counts.loc[ct].max() == direction_counts.loc[ct].sum()
        ) else 'mixed'
    )
    
    # Sort by number of significant cohorts and effect size
    summary = summary.sort_values(
        ['n_significant', 't_stat_mean'], 
        ascending=[False, False]
    )
    
    return summary

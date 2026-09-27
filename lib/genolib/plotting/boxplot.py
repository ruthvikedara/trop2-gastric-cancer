"""Boxplots of cell-type scores by expression group."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def create_boxplot_grid(
    expression_dfs: Dict[str, pd.DataFrame],
    xcell_dfs: Dict[str, pd.DataFrame],
    gene: str,
    cell_types: List[str],
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Create a grid of boxplots comparing high/low groups across cohorts.

    Pools all samples from every cohort and compares gene-high vs gene-low
    enrichment scores for each cell type.

    Args:
        expression_dfs: Dict of cohort_name -> expression DataFrame
        xcell_dfs: Dict of cohort_name -> xCell DataFrame
        gene: Gene symbol used for stratification
        cell_types: Cell types to plot
        save_path: Path to save figure (PDF recommended)
        show: Whether to call plt.show()

    Returns:
        Matplotlib Figure, or None if no data
    """
    from genolib.analysis.compare import stratify_patients

    n_cells = len(cell_types)
    if n_cells == 0:
        return None

    n_cols = 4
    n_rows = int(np.ceil(n_cells / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)
    axes = axes.flatten()

    for idx, cell_type in enumerate(cell_types):
        ax = axes[idx]
        all_high: list = []
        all_low: list = []

        for name, xcell_df in xcell_dfs.items():
            if name not in expression_dfs:
                continue
            if cell_type not in xcell_df.index:
                continue

            try:
                groups = stratify_patients(expression_dfs[name], gene)
            except ValueError:
                continue

            scores = xcell_df.loc[cell_type]
            all_high.extend(
                scores[s] for s in groups['high'] if s in scores.index
            )
            all_low.extend(
                scores[s] for s in groups['low'] if s in scores.index
            )

        if not all_high or not all_low:
            ax.set_visible(False)
            continue

        t_stat, p_val = stats.ttest_ind(all_high, all_low)
        if p_val < 2.2e-16:
            p_text = "p < 2.2e-16"
        elif p_val < 0.001:
            p_text = f"p = {p_val:.2e}"
        else:
            p_text = f"p = {p_val:.3f}"

        box = ax.boxplot(
            [all_low, all_high],
            labels=[f'{gene}-low', f'{gene}-high'],
            patch_artist=True,
            widths=0.6,
        )
        box['boxes'][0].set_facecolor('#BEBEBE')
        box['boxes'][1].set_facecolor('#1E90FF')

        ax.set_ylabel('enrichment score', fontsize=9)
        ax.set_title(f'{cell_type}\nT-test, {p_text}', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=8)

    for idx in range(len(cell_types), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f'{gene} High vs Low: Immune Cell Enrichment',
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved boxplot grid to {save_path}")

    if show:
        plt.show()

    return fig

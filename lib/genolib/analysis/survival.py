"""Kaplan-Meier and Cox survival helpers (lifelines)."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logger.warning("lifelines not installed - survival analysis functions unavailable")


def _check_lifelines():
    if not LIFELINES_AVAILABLE:
        raise ImportError(
            "lifelines is required for survival analysis. "
            "Install with: pip install lifelines"
        )


def run_km_analysis(
    clinical_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    time_col: str = 'os_months',
    event_col: str = 'os_event',
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Kaplan-Meier analysis with log-rank test between groups.

    Args:
        clinical_df: Clinical data indexed by patient/sample ID.
            Must contain time_col and event_col columns.
        groups: Dictionary of group_name -> list of sample IDs
            (e.g. {'high': [...], 'low': [...]}).
        time_col: Column name for time-to-event (months, days, etc.).
        event_col: Column name for event indicator (1=event, 0=censored).
        alpha: Confidence interval alpha level.

    Returns:
        Dictionary with:
            - 'kmf': Dict of group_name -> fitted KaplanMeierFitter
            - 'log_rank_p': p-value from log-rank test
            - 'log_rank_stat': test statistic
            - 'median_survival': Dict of group_name -> median survival time
            - 'n_per_group': Dict of group_name -> sample count

    Example::

        groups = stratify_patients(expression_df, 'CDH6')
        results = run_km_analysis(clinical_df, groups, 'os_months', 'os_event')
        print(f"Log-rank p = {results['log_rank_p']:.4f}")
        for name, kmf in results['kmf'].items():
            print(f"  {name}: median OS = {kmf.median_survival_time_:.1f} months")
    """
    _check_lifelines()

    # Filter to patients present in clinical data
    available = clinical_df.index
    filtered_groups = {
        name: [s for s in samples if s in available]
        for name, samples in groups.items()
    }

    kmf_dict = {}
    for name, samples in filtered_groups.items():
        if not samples:
            logger.warning(f"No clinical data for group '{name}', skipping")
            continue

        subset = clinical_df.loc[samples]
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=subset[time_col],
            event_observed=subset[event_col],
            label=name,
            alpha=alpha,
        )
        kmf_dict[name] = kmf

    # Log-rank test (pairwise for 2 groups, multivariate for >2)
    group_names = list(filtered_groups.keys())
    if len(group_names) == 2:
        g1, g2 = group_names
        s1 = clinical_df.loc[filtered_groups[g1]]
        s2 = clinical_df.loc[filtered_groups[g2]]
        lr = logrank_test(
            s1[time_col], s2[time_col],
            event_observed_A=s1[event_col],
            event_observed_B=s2[event_col],
        )
        log_rank_p = lr.p_value
        log_rank_stat = lr.test_statistic
    elif len(group_names) > 2:
        # Build combined frame with group labels
        frames = []
        for name, samples in filtered_groups.items():
            sub = clinical_df.loc[samples, [time_col, event_col]].copy()
            sub['group'] = name
            frames.append(sub)
        combined = pd.concat(frames)
        lr = multivariate_logrank_test(
            combined[time_col], combined['group'], combined[event_col],
        )
        log_rank_p = lr.p_value
        log_rank_stat = lr.test_statistic
    else:
        log_rank_p = np.nan
        log_rank_stat = np.nan

    return {
        'kmf': kmf_dict,
        'log_rank_p': log_rank_p,
        'log_rank_stat': log_rank_stat,
        'median_survival': {
            name: kmf.median_survival_time_ for name, kmf in kmf_dict.items()
        },
        'n_per_group': {name: len(s) for name, s in filtered_groups.items()},
    }


def run_cox_regression(
    clinical_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    time_col: str = 'os_months',
    event_col: str = 'os_event',
    covariates: Optional[List[str]] = None,
    reference_group: str = 'low',
) -> Dict[str, Any]:
    """Cox proportional hazards regression.

    Encodes group membership as dummy variables and fits a Cox model,
    optionally adjusting for additional covariates.

    Args:
        clinical_df: Clinical data indexed by patient/sample ID.
        groups: Dictionary of group_name -> list of sample IDs.
        time_col: Column name for time-to-event.
        event_col: Column name for event indicator.
        covariates: Additional covariate columns to include in the model.
        reference_group: Group to use as baseline (dummy-coded reference).

    Returns:
        Dictionary with:
            - 'cph': Fitted CoxPHFitter object
            - 'summary': Model summary DataFrame
            - 'hazard_ratios': Dict of group_name -> HR vs reference
            - 'concordance': C-index

    Example::

        groups = stratify_patients(expression_df, 'CDH6')
        cox = run_cox_regression(
            clinical_df, groups, 'os_months', 'os_event',
            covariates=['age', 'stage'],
        )
        print(cox['summary'])
        print(f"C-index: {cox['concordance']:.3f}")
    """
    _check_lifelines()

    # Build analysis DataFrame
    frames = []
    for name, samples in groups.items():
        available = [s for s in samples if s in clinical_df.index]
        if not available:
            continue
        sub = clinical_df.loc[available, [time_col, event_col]].copy()
        if covariates:
            for cov in covariates:
                if cov in clinical_df.columns:
                    sub[cov] = clinical_df.loc[available, cov]
        sub['group'] = name
        frames.append(sub)

    if not frames:
        raise ValueError("No samples found in clinical data")

    analysis_df = pd.concat(frames)

    # Dummy-encode groups with reference
    dummies = pd.get_dummies(analysis_df['group'], prefix='group', dtype=float)
    ref_col = f'group_{reference_group}'
    if ref_col in dummies.columns:
        dummies = dummies.drop(columns=[ref_col])
    analysis_df = pd.concat([analysis_df.drop(columns=['group']), dummies], axis=1)

    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(analysis_df, duration_col=time_col, event_col=event_col)

    # Extract hazard ratios for group variables
    hazard_ratios = {}
    for col in dummies.columns:
        if col in cph.summary.index:
            hazard_ratios[col.replace('group_', '')] = cph.summary.loc[col, 'exp(coef)']

    return {
        'cph': cph,
        'summary': cph.summary,
        'hazard_ratios': hazard_ratios,
        'concordance': cph.concordance_index_,
    }


def run_four_way_survival(
    clinical_df: pd.DataFrame,
    gene_groups: Dict[str, List[str]],
    cell_groups: Dict[str, List[str]],
    time_col: str = 'os_months',
    event_col: str = 'os_event',
) -> Dict[str, Any]:
    """4-way Kaplan-Meier analysis for gene x cell type stratification.

    Creates 4 groups by crossing gene expression (high/low) with cell type
    score (high/low) and runs survival analysis on all 4 groups.

    Args:
        clinical_df: Clinical data indexed by patient/sample ID.
        gene_groups: Dict with 'high' and 'low' keys from gene stratification.
        cell_groups: Dict with 'high' and 'low' keys from cell type stratification.
        time_col: Column name for time-to-event.
        event_col: Column name for event indicator.

    Returns:
        Dictionary with:
            - 'km_results': Output of run_km_analysis() with 4 groups
            - 'four_way_groups': The constructed 4-way group dict
            - 'pairwise_p': Dict of pairwise log-rank p-values

    Example::

        gene_groups = stratify_patients(expression_df, 'CDH6')
        cell_groups = stratify_by_cell_type(xcell_df, 'CD8+ T-cells')
        results = run_four_way_survival(
            clinical_df, gene_groups, cell_groups,
            time_col='os_months', event_col='os_event',
        )
        print(f"Overall log-rank p = {results['km_results']['log_rank_p']:.4f}")
    """
    _check_lifelines()

    gene_high = set(gene_groups['high'])
    gene_low = set(gene_groups['low'])
    cell_high = set(cell_groups['high'])
    cell_low = set(cell_groups['low'])

    four_way = {
        'gene_high_cell_high': sorted(gene_high & cell_high),
        'gene_high_cell_low': sorted(gene_high & cell_low),
        'gene_low_cell_high': sorted(gene_low & cell_high),
        'gene_low_cell_low': sorted(gene_low & cell_low),
    }

    # Overall KM + multivariate log-rank
    km_results = run_km_analysis(clinical_df, four_way, time_col, event_col)

    # Pairwise log-rank tests
    pairwise_p = {}
    group_names = list(four_way.keys())
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1, g2 = group_names[i], group_names[j]
            pair = {g1: four_way[g1], g2: four_way[g2]}
            pair_result = run_km_analysis(clinical_df, pair, time_col, event_col)
            pairwise_p[f"{g1} vs {g2}"] = pair_result['log_rank_p']

    return {
        'km_results': km_results,
        'four_way_groups': four_way,
        'pairwise_p': pairwise_p,
    }


def plot_km_curves(
    km_results: Dict[str, Any],
    title: str = 'Kaplan-Meier Survival Curves',
    xlabel: str = 'Time (months)',
    ylabel: str = 'Survival Probability',
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (8, 6),
) -> None:
    """Plot Kaplan-Meier curves with risk table annotation.

    Args:
        km_results: Output of run_km_analysis() (must contain 'kmf' dict).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Path to save figure (PDF/PNG). None to skip.
        show: Whether to display the plot interactively.
        figsize: Figure size in inches.

    Example::

        groups = stratify_patients(expression_df, 'CDH6')
        km = run_km_analysis(clinical_df, groups, 'os_months', 'os_event')
        plot_km_curves(
            km, title='CDH6 High vs Low - Overall Survival',
            save_path='results/CDH6_km.pdf',
        )
    """
    _check_lifelines()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for name, kmf in km_results['kmf'].items():
        n = km_results['n_per_group'].get(name, '?')
        median = km_results['median_survival'].get(name, np.nan)
        label = f"{name} (n={n}, median={median:.1f})"
        kmf.plot_survival_function(ax=ax, label=label)

    p_val = km_results.get('log_rank_p', np.nan)
    if not np.isnan(p_val):
        p_text = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        ax.text(0.95, 0.95, f"Log-rank {p_text}",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved KM plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

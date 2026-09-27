"""Patient stratification and group comparisons."""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of a statistical comparison."""
    cell_type: str
    cohort: str
    gene: str
    t_statistic: float
    p_value: float
    n_high: int
    n_low: int
    mean_high: float
    mean_low: float
    direction: str  # 'increased' or 'decreased' in high group
    significant: bool


def stratify_patients(
    expression_df: pd.DataFrame,
    gene: str,
    method: Literal['median', 'tertiles', 'quartiles'] = 'median'
) -> Dict[str, List[str]]:
    """Stratify patients into groups based on gene expression.
    
    Args:
        expression_df: Expression matrix (genes × samples)
        gene: Gene symbol to stratify by
        method: Stratification method
        
    Returns:
        Dictionary with 'high', 'low', and optionally 'mid' patient lists
    """
    from genolib.utils.gene_symbols import gene_mapper
    
    # Find gene in matrix (handling aliases)
    gene_match = gene_mapper.find_gene_in_matrix(expression_df, gene)
    
    if gene_match is None:
        raise ValueError(f"Gene {gene} not found in expression matrix")
    
    # Get expression values for this gene
    gene_expr = expression_df.loc[gene_match]
    
    if method == 'median':
        threshold = gene_expr.median()
        groups = {
            'high': gene_expr[gene_expr > threshold].index.tolist(),
            'low': gene_expr[gene_expr <= threshold].index.tolist()
        }
        
    elif method == 'tertiles':
        q33 = gene_expr.quantile(1/3)
        q66 = gene_expr.quantile(2/3)
        groups = {
            'high': gene_expr[gene_expr > q66].index.tolist(),
            'mid': gene_expr[(gene_expr > q33) & (gene_expr <= q66)].index.tolist(),
            'low': gene_expr[gene_expr <= q33].index.tolist()
        }
        
    elif method == 'quartiles':
        q25 = gene_expr.quantile(0.25)
        q75 = gene_expr.quantile(0.75)
        groups = {
            'high': gene_expr[gene_expr > q75].index.tolist(),
            'mid': gene_expr[(gene_expr > q25) & (gene_expr <= q75)].index.tolist(),
            'low': gene_expr[gene_expr <= q25].index.tolist()
        }
    else:
        raise ValueError(f"Unknown stratification method: {method}")
    
    logger.info(f"Stratified {len(gene_expr)} patients by {gene}: "
                f"high={len(groups['high'])}, low={len(groups['low'])}")

    return groups


def stratify_by_variable(
    values: pd.Series,
    method: Literal['median', 'tertiles', 'quartiles'] = 'median',
) -> Dict[str, List[str]]:
    """Stratify samples based on any continuous variable.

    Generic stratification that works with gene expression, xCell scores,
    or any numeric Series indexed by sample IDs.

    Args:
        values: Series of numeric values indexed by sample IDs
        method: 'median' (high/low), 'tertiles' (high/mid/low at 1/3, 2/3),
                or 'quartiles' (high/mid/low at 0.25, 0.75)

    Returns:
        Dictionary with 'high', 'low', and optionally 'mid' sample lists
    """
    values = values.dropna()

    if method == 'median':
        threshold = values.median()
        groups = {
            'high': values[values > threshold].index.tolist(),
            'low': values[values <= threshold].index.tolist(),
        }
    elif method == 'tertiles':
        q33 = values.quantile(1 / 3)
        q66 = values.quantile(2 / 3)
        groups = {
            'high': values[values > q66].index.tolist(),
            'mid': values[(values > q33) & (values <= q66)].index.tolist(),
            'low': values[values <= q33].index.tolist(),
        }
    elif method == 'quartiles':
        q25 = values.quantile(0.25)
        q75 = values.quantile(0.75)
        groups = {
            'high': values[values > q75].index.tolist(),
            'mid': values[(values > q25) & (values <= q75)].index.tolist(),
            'low': values[values <= q25].index.tolist(),
        }
    else:
        raise ValueError(f"Unknown stratification method: {method}")

    logger.info(
        f"Stratified {len(values)} samples by {values.name or 'variable'}: "
        + ", ".join(f"{k}={len(v)}" for k, v in groups.items())
    )
    return groups


def stratify_by_cell_type(
    xcell_df: pd.DataFrame,
    cell_type: str,
    method: Literal['median', 'tertiles', 'quartiles'] = 'median',
) -> Dict[str, List[str]]:
    """Stratify patients by xCell cell type enrichment score.

    Args:
        xcell_df: xCell results (cell types x samples)
        cell_type: Cell type name (must match an xCell row)
        method: Stratification method

    Returns:
        Dictionary with 'high', 'low', and optionally 'mid' sample lists

    Raises:
        ValueError: If cell_type is not found in xcell_df
    """
    # Case-insensitive lookup
    matching = [ct for ct in xcell_df.index if ct.lower() == cell_type.lower()]
    if not matching:
        raise ValueError(
            f"Cell type '{cell_type}' not found in xCell results. "
            f"Available: {xcell_df.index.tolist()}"
        )
    row_name = matching[0]
    values = xcell_df.loc[row_name].astype(float)
    values.name = cell_type
    return stratify_by_variable(values, method=method)


def create_four_way_groups(
    expression_df: pd.DataFrame,
    xcell_df: pd.DataFrame,
    gene: str,
    cell_type: str,
    method: Literal['median', 'tertiles', 'quartiles'] = 'median',
) -> Dict[str, List[str]]:
    """Create 4-way patient groups by crossing gene expression and cell type score.

    Mirrors the R approach: paste0(gene, "-", gene_cat, ", ", cell_type, "-", cell_type_cat)

    Args:
        expression_df: Expression matrix (genes x samples)
        xcell_df: xCell results (cell types x samples)
        gene: Gene symbol for expression stratification
        cell_type: Cell type for xCell score stratification
        method: Stratification method (only median supported for 4-way)

    Returns:
        Dictionary with keys: gene_high_cell_high, gene_high_cell_low,
        gene_low_cell_high, gene_low_cell_low
    """
    gene_groups = stratify_patients(expression_df, gene, method=method)
    cell_groups = stratify_by_cell_type(xcell_df, cell_type, method=method)

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

    logger.info(
        f"4-way groups ({gene} x {cell_type}): "
        + ", ".join(f"{k}={len(v)}" for k, v in four_way.items())
    )
    return four_way


def compare_pairwise(
    xcell_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    reference_group: str,
    gene: str,
    cohort: str,
    test: Literal['ttest', 'wilcoxon'] = 'ttest',
    alpha: float = 0.05,
) -> List[ComparisonResult]:
    """Compare each group against a reference group for all cell types.

    Args:
        xcell_df: xCell results (cell types x samples)
        groups: Dictionary of group_name -> sample IDs (e.g. from create_four_way_groups)
        reference_group: Key in groups to use as reference
        gene: Gene label for results
        cohort: Cohort name
        test: Statistical test
        alpha: Significance threshold

    Returns:
        List of ComparisonResult, one per (cell_type, comparison) pair.
        The cohort field encodes the comparison (e.g. "GSE62254: high_high vs high_low").
    """
    if reference_group not in groups:
        raise ValueError(
            f"Reference group '{reference_group}' not in groups: {list(groups.keys())}"
        )

    all_results = []
    for group_name, group_samples in groups.items():
        if group_name == reference_group:
            continue

        comparison_groups = {
            'high': groups[reference_group],
            'low': group_samples,
        }
        comparison_label = f"{cohort}: {reference_group} vs {group_name}"

        results = compare_cell_types(
            xcell_df, comparison_groups, gene, comparison_label, test=test, alpha=alpha,
        )
        all_results.extend(results)

    return all_results


def compare_categorical_variable(
    xcell_df: pd.DataFrame,
    variable_series: pd.Series,
    reference_level: str,
    cohort: str,
    test: Literal['ttest', 'wilcoxon'] = 'ttest',
    alpha: float = 0.05,
) -> List[ComparisonResult]:
    """Compare cell type abundances across levels of a categorical variable.

    For each level != reference, runs the specified test on all cell types.

    Args:
        xcell_df: xCell results (cell types x samples)
        variable_series: Series mapping sample IDs to category levels
        reference_level: The reference category (e.g. "0" for ECOG)
        cohort: Cohort name
        test: Statistical test
        alpha: Significance threshold

    Returns:
        List of ComparisonResult with cohort encoding the comparison
    """
    variable_name = variable_series.name or 'variable'
    variable_series = variable_series.dropna().astype(str)

    levels = sorted(variable_series.unique())
    if reference_level not in levels:
        raise ValueError(
            f"Reference level '{reference_level}' not found. Available: {levels}"
        )

    ref_samples = variable_series[variable_series == reference_level].index.tolist()

    all_results = []
    for level in levels:
        if level == reference_level:
            continue
        level_samples = variable_series[variable_series == level].index.tolist()

        groups = {'high': ref_samples, 'low': level_samples}
        comparison_label = f"{cohort}: {variable_name} {reference_level} vs {level}"

        results = compare_cell_types(
            xcell_df, groups, variable_name, comparison_label, test=test, alpha=alpha,
        )
        all_results.extend(results)

    return all_results


def run_cell_type_stratified_analysis(
    expression_dfs: Dict[str, pd.DataFrame],
    xcell_dfs: Dict[str, pd.DataFrame],
    gene: str,
    cell_types: List[str],
    method: str = 'median',
    test: str = 'ttest',
) -> pd.DataFrame:
    """Run 4-way (gene x cell type) stratified analysis across cohorts.

    For each cohort and cell type combination, creates 4-way groups and
    runs pairwise comparisons against the gene_high_cell_high reference.

    Args:
        expression_dfs: Dict of cohort_name -> expression DataFrame
        xcell_dfs: Dict of cohort_name -> xCell DataFrame
        gene: Gene to stratify by
        cell_types: Cell types to cross with gene expression
        method: Stratification method
        test: Statistical test

    Returns:
        Results DataFrame with extra columns: stratify_gene, stratify_cell_type,
        group_comparison
    """
    all_results = []

    for cohort_name in expression_dfs:
        if cohort_name not in xcell_dfs:
            logger.warning(f"No xCell data for {cohort_name}, skipping")
            continue

        expr_df = expression_dfs[cohort_name]
        xcell_df = xcell_dfs[cohort_name]
        n_samples = expr_df.shape[1]
        cohort_display = f"{cohort_name} (n={n_samples})"

        for cell_type in cell_types:
            try:
                four_way = create_four_way_groups(
                    expr_df, xcell_df, gene, cell_type, method=method,
                )

                results = compare_pairwise(
                    xcell_df, four_way,
                    reference_group='gene_high_cell_high',
                    gene=gene,
                    cohort=cohort_display,
                    test=test,
                )

                for r in results:
                    all_results.append({
                        'cell_type': r.cell_type,
                        'cohort': r.cohort,
                        'gene': r.gene,
                        'stratify_gene': gene,
                        'stratify_cell_type': cell_type,
                        'group_comparison': r.cohort.split(': ', 1)[-1] if ': ' in r.cohort else r.cohort,
                        't_statistic': r.t_statistic,
                        'p_value': r.p_value,
                        'n_high': r.n_high,
                        'n_low': r.n_low,
                        'mean_high': r.mean_high,
                        'mean_low': r.mean_low,
                        'direction': r.direction,
                        'significant': r.significant,
                    })
            except Exception as e:
                logger.error(f"Error in 4-way analysis for {cohort_name}/{cell_type}: {e}")

    if not all_results:
        return pd.DataFrame()

    combined = pd.DataFrame(all_results)

    # FDR correction across all results
    if len(combined) > 1:
        try:
            from statsmodels.stats.multitest import multipletests
            _, pvals_corrected, _, _ = multipletests(
                combined['p_value'].values, method='fdr_bh',
            )
            combined['p_adjusted'] = pvals_corrected
            combined['fdr_significant'] = combined['p_adjusted'] < 0.05
        except ImportError:
            combined['p_adjusted'] = combined['p_value']
            combined['fdr_significant'] = combined['significant']

    return combined


def compare_cell_types(
    xcell_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    gene: str,
    cohort: str,
    test: Literal['ttest', 'wilcoxon'] = 'ttest',
    alpha: float = 0.05
) -> List[ComparisonResult]:
    """Compare cell type abundances between patient groups.
    
    Args:
        xcell_df: xCell results (cell types × samples)
        groups: Patient groups from stratify_patients()
        gene: Gene used for stratification
        cohort: Cohort name
        test: Statistical test to use
        alpha: Significance threshold
        
    Returns:
        List of ComparisonResult objects
    """
    results = []
    
    high_patients = [p for p in groups['high'] if p in xcell_df.columns]
    low_patients = [p for p in groups['low'] if p in xcell_df.columns]
    
    if len(high_patients) < 3 or len(low_patients) < 3:
        logger.warning(f"Insufficient samples for comparison: high={len(high_patients)}, low={len(low_patients)}")
        return results
    
    # Exclude summary scores from cell type comparisons
    exclude_rows = ['immune score', 'stroma score', 'microenvironment score']
    cell_types = [ct for ct in xcell_df.index if ct.lower() not in [e.lower() for e in exclude_rows]]
    
    for cell_type in cell_types:
        high_values = xcell_df.loc[cell_type, high_patients].values.astype(float)
        low_values = xcell_df.loc[cell_type, low_patients].values.astype(float)
        
        # Remove NaN values
        high_values = high_values[~np.isnan(high_values)]
        low_values = low_values[~np.isnan(low_values)]
        
        if len(high_values) < 3 or len(low_values) < 3:
            continue
        
        # Statistical test
        if test == 'ttest':
            t_stat, p_val = stats.ttest_ind(high_values, low_values)
        elif test == 'wilcoxon':
            t_stat, p_val = stats.mannwhitneyu(high_values, low_values, alternative='two-sided')
            # Convert U to approximate z-score for consistency
            n1, n2 = len(high_values), len(low_values)
            mean_u = n1 * n2 / 2
            std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            t_stat = (t_stat - mean_u) / std_u
        else:
            raise ValueError(f"Unknown test: {test}")
        
        # Determine direction
        mean_high = np.mean(high_values)
        mean_low = np.mean(low_values)
        direction = 'increased' if mean_high > mean_low else 'decreased'
        
        results.append(ComparisonResult(
            cell_type=cell_type,
            cohort=cohort,
            gene=gene,
            t_statistic=t_stat,
            p_value=p_val,
            n_high=len(high_values),
            n_low=len(low_values),
            mean_high=mean_high,
            mean_low=mean_low,
            direction=direction,
            significant=p_val < alpha
        ))
    
    return results


def aggregate_results(
    results: List[ComparisonResult],
    fdr_correction: bool = True
) -> pd.DataFrame:
    """Aggregate comparison results into a summary DataFrame.
    
    Args:
        results: List of ComparisonResult objects
        fdr_correction: Apply FDR correction to p-values
        
    Returns:
        Summary DataFrame
    """
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame([
        {
            'cell_type': r.cell_type,
            'cohort': r.cohort,
            'gene': r.gene,
            't_statistic': r.t_statistic,
            'p_value': r.p_value,
            'n_high': r.n_high,
            'n_low': r.n_low,
            'mean_high': r.mean_high,
            'mean_low': r.mean_low,
            'direction': r.direction,
            'significant': r.significant
        }
        for r in results
    ])
    
    if fdr_correction and len(df) > 1:
        try:
            from statsmodels.stats.multitest import multipletests
            _, pvals_corrected, _, _ = multipletests(df['p_value'].values, method='fdr_bh')
            df['p_adjusted'] = pvals_corrected
            df['fdr_significant'] = df['p_adjusted'] < 0.05
        except ImportError:
            logger.warning("statsmodels not available, skipping FDR correction")
            df['p_adjusted'] = df['p_value']
            df['fdr_significant'] = df['significant']
    
    return df


def run_analysis(
    expression_df: pd.DataFrame,
    xcell_df: pd.DataFrame,
    gene: str,
    cohort: str,
    stratification: str = 'median',
    test: str = 'ttest'
) -> pd.DataFrame:
    """Run full analysis pipeline for a single gene and cohort.
    
    Args:
        expression_df: Normalized expression matrix
        xcell_df: xCell deconvolution results
        gene: Gene to analyze
        cohort: Cohort name
        stratification: Stratification method
        test: Statistical test
        
    Returns:
        Results DataFrame
    """
    # Stratify patients
    groups = stratify_patients(expression_df, gene, method=stratification)
    
    # Compare cell types
    results = compare_cell_types(xcell_df, groups, gene, cohort, test=test)
    
    # Aggregate
    return aggregate_results(results, fdr_correction=False)


def run_multi_cohort_analysis(
    expression_dfs: Dict[str, pd.DataFrame],
    xcell_dfs: Dict[str, pd.DataFrame],
    gene: str,
    stratification: str = 'median',
    test: str = 'ttest'
) -> pd.DataFrame:
    """Run analysis across multiple cohorts.
    
    Args:
        expression_dfs: Dict of cohort_name → expression DataFrame
        xcell_dfs: Dict of cohort_name → xCell DataFrame
        gene: Gene to analyze
        stratification: Stratification method
        test: Statistical test
        
    Returns:
        Combined results DataFrame
    """
    all_results = []
    
    for cohort_name in expression_dfs.keys():
        if cohort_name not in xcell_dfs:
            logger.warning(f"No xCell data for cohort {cohort_name}, skipping")
            continue
        
        try:
            expr_df = expression_dfs[cohort_name]
            xcell_df = xcell_dfs[cohort_name]
            
            # Add sample count to cohort name for display
            n_samples = expr_df.shape[1]
            cohort_display = f"{cohort_name} (n={n_samples})"
            
            result_df = run_analysis(
                expr_df, xcell_df, gene, cohort_display,
                stratification=stratification, test=test
            )
            
            if not result_df.empty:
                all_results.append(result_df)
                
        except Exception as e:
            logger.error(f"Error analyzing {cohort_name}: {e}")
            continue
    
    if not all_results:
        return pd.DataFrame()
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # Apply FDR correction across all results
    if len(combined) > 1:
        try:
            from statsmodels.stats.multitest import multipletests
            _, pvals_corrected, _, _ = multipletests(combined['p_value'].values, method='fdr_bh')
            combined['p_adjusted'] = pvals_corrected
            combined['fdr_significant'] = combined['p_adjusted'] < 0.05
        except ImportError:
            combined['p_adjusted'] = combined['p_value']
            combined['fdr_significant'] = combined['significant']
    
    return combined


def create_significance_summary(
    results_df: pd.DataFrame,
    gene: str,
    cell_filter: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Count how many cohorts show significant results per cell type.

    Args:
        results_df: Results from run_multi_cohort_analysis()
        gene: Gene name for the header
        cell_filter: Optional list of cell types to include
        save_path: Path to save the summary text file

    Returns:
        Summary DataFrame sorted by number of significant cohorts
    """
    if results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()
    if cell_filter:
        df = df[df['cell_type'].isin(cell_filter)]

    df['significant'] = df['p_value'] < 0.05

    summary = df.groupby('cell_type').agg(
        n_cohorts=('cohort', 'count'),
        n_significant=('significant', 'sum'),
        n_increased=('direction', lambda x: (
            (x == 'increased') & df.loc[x.index, 'significant']
        ).sum()),
        n_decreased=('direction', lambda x: (
            (x == 'decreased') & df.loc[x.index, 'significant']
        ).sum()),
        mean_t_stat=('t_statistic', 'mean'),
    ).reset_index()

    summary['summary'] = summary.apply(
        lambda row: f"{int(row['n_significant'])}/{int(row['n_cohorts'])} cohorts (p<0.05)",
        axis=1,
    )
    summary = summary.sort_values('n_significant', ascending=False)

    # Build text output
    lines = [
        f"\n{'=' * 60}",
        f"SIGNIFICANCE SUMMARY: {gene}",
        f"{'=' * 60}\n",
    ]
    for _, row in summary.iterrows():
        arrow = "decreased" if row['mean_t_stat'] < 0 else "increased"
        lines.append(f"{row['cell_type']}: {row['summary']} - {arrow}")

    text = '\n'.join(lines)
    logger.info(text)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Saved significance summary to {save_path}")

    return summary

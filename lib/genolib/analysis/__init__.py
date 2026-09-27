"""Analysis modules for statistical comparisons."""

from .compare import (
    ComparisonResult,
    stratify_patients,
    stratify_by_variable,
    stratify_by_cell_type,
    create_four_way_groups,
    compare_cell_types,
    compare_pairwise,
    compare_categorical_variable,
    aggregate_results,
    run_analysis,
    run_multi_cohort_analysis,
    run_cell_type_stratified_analysis,
    create_significance_summary,
)

__all__ = [
    'ComparisonResult',
    'stratify_patients',
    'stratify_by_variable',
    'stratify_by_cell_type',
    'create_four_way_groups',
    'compare_cell_types',
    'compare_pairwise',
    'compare_categorical_variable',
    'aggregate_results',
    'run_analysis',
    'run_multi_cohort_analysis',
    'run_cell_type_stratified_analysis',
    'create_significance_summary',
]

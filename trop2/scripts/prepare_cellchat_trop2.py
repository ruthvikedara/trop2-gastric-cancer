"""Prepare CellChat inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

from genolib import PROJECT_ROOT


DATASETS = {
    "Kumar": PROJECT_ROOT / "data/processed/scrna/gastric/kumar_tumor.h5ad",
    "Sathe": PROJECT_ROOT / "data/processed/scrna/gastric/sathe_tumor.h5ad",
}
OUT = PROJECT_ROOT / "trop2/results/cellchat"

# Marker sets are deliberately compact and canonical.  Scores are the mean
# within-lineage z-score of log-normalized expression.  Hierarchical annotation
# prevents, for example, cytotoxic NK cells from being called CD8 T cells.
MARKERS = {
    "T_cell": ["CD3D", "CD3E", "TRAC", "CD247", "LTB"],
    "NK": ["NKG7", "GNLY", "KLRD1", "FCER1G", "TYROBP"],
    "CD4 T": ["CD4", "IL7R", "CCR7", "LTB", "MAL", "TCF7"],
    "CD8 T": ["CD8A", "CD8B", "CCL5", "GZMK", "CST7"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TIGIT", "TNFRSF4"],
    "B cell": ["MS4A1", "CD79A", "CD79B", "CD19", "CD37", "CD74", "CD22"],
    "Plasma": ["MZB1", "JCHAIN", "SDC1", "TNFRSF17", "IGHG1", "IGKC"],
    "Macrophage": ["C1QA", "C1QB", "C1QC", "APOE", "CD68", "CD163", "MSR1", "MRC1"],
    "Monocyte": ["S100A8", "S100A9", "FCN1", "CTSS", "LILRB1", "CD14"],
    "Dendritic": ["FCER1A", "CD1C", "CLEC10A", "CST3", "LILRA4", "GZMB"],
}
ANNOTATION_TYPES = ["Epithelial", "B cell", "CD4 T", "CD8 T", "Macrophage", "Treg"]
TARGET_TYPES = ["Epithelial", "CD8 T", "Macrophage", "Treg"]


def dense_gene_matrix(adata: ad.AnnData, genes: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return cells x present genes from normalized X without densifying the atlas."""
    present = [gene for gene in genes if gene in adata.var_names]
    x = adata[:, present].X
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32), present


def zscore_columns(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd == 0] = 1
    return np.clip((x - mean) / sd, -5, 5)


def signature_scores(adata: ad.AnnData, indices: np.ndarray,
                     signatures: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = sorted({g for name in signatures for g in MARKERS[name]})
    x, present = dense_gene_matrix(adata[indices], genes)
    z = zscore_columns(x)
    gene_to_col = {gene: i for i, gene in enumerate(present)}
    scores = {}
    detected = {}
    for name in signatures:
        cols = [gene_to_col[g] for g in MARKERS[name] if g in gene_to_col]
        if not cols:
            raise ValueError(f"No markers found for {name}")
        scores[name] = z[:, cols].mean(axis=1)
        detected[name] = (x[:, cols] > 0).sum(axis=1)
    index = adata.obs_names[indices]
    return pd.DataFrame(scores, index=index), pd.DataFrame(detected, index=index)


def annotate_target_types(adata: ad.AnnData) -> pd.Series:
    labels = pd.Series("Other", index=adata.obs_names, dtype="object")
    labels.loc[adata.obs["lineage"].eq("Epithelial")] = "Epithelial"

    # B cells: separate antibody-secreting plasma cells from conventional B cells.
    idx = np.flatnonzero(adata.obs["lineage"].eq("B/Plasma").to_numpy())
    score, detected = signature_scores(adata, idx, ["B cell", "Plasma"])
    is_b = score["B cell"].ge(score["Plasma"]) & detected["B cell"].ge(1)
    labels.loc[score.index[is_b]] = "B cell"

    # T-cell gate first excludes NK profiles, then separates CD4/CD8/Treg states.
    idx = np.flatnonzero(adata.obs["lineage"].eq("T/NK").to_numpy())
    gate, gate_detected = signature_scores(adata, idx, ["T_cell", "NK"])
    is_t = gate["T_cell"].gt(gate["NK"]) & gate_detected["T_cell"].ge(1)
    t_names = gate.index[is_t]
    t_idx = adata.obs_names.get_indexer(t_names)
    score, detected = signature_scores(adata, t_idx, ["CD4 T", "CD8 T", "Treg"])
    best = score.idxmax(axis=1)
    # Regulatory calls require at least one detected canonical regulatory marker;
    # otherwise assign the stronger conventional CD4/CD8 program.
    weak_treg = best.eq("Treg") & detected["Treg"].eq(0)
    best.loc[weak_treg] = score.loc[weak_treg, ["CD4 T", "CD8 T"]].idxmax(axis=1)
    labels.loc[best.index] = best

    # Retain high-confidence macrophages; monocytes/DCs are excluded.
    idx = np.flatnonzero(adata.obs["lineage"].eq("Myeloid").to_numpy())
    score, detected = signature_scores(adata, idx, ["Macrophage", "Monocyte", "Dendritic"])
    is_macro = score.idxmax(axis=1).eq("Macrophage") & detected["Macrophage"].ge(1)
    labels.loc[score.index[is_macro]] = "Macrophage"
    return pd.Categorical(labels, categories=ANNOTATION_TYPES + ["Other"], ordered=True)


def epithelial_trop2_pseudobulk(adata: ad.AnnData) -> pd.DataFrame:
    counts = adata.layers["counts"]
    gene_idx = adata.var_names.get_loc("TACSTD2")
    rows = []
    for sample, obs in adata.obs.groupby("sample", observed=True):
        idx = adata.obs_names.get_indexer(obs.index)
        idx = idx[np.asarray(adata.obs.iloc[idx]["cell_type"].eq("Epithelial"))]
        if len(idx) == 0:
            continue
        sub = counts[idx]
        total = float(sub.sum())
        trop2 = float(sub[:, gene_idx].sum())
        cpm = (trop2 / total * 1e6) if total > 0 else np.nan
        rows.append({"sample": sample, "n_epithelial": len(idx), "TACSTD2_counts": trop2,
                     "library_counts": total, "TACSTD2_log2_CPM": np.log2(cpm + 1)})
    out = pd.DataFrame(rows)
    median = out["TACSTD2_log2_CPM"].median()
    out["condition"] = np.where(out["TACSTD2_log2_CPM"] >= median,
                                "TROP2-high", "TROP2-low")
    out["dataset_median_log2_CPM"] = median
    return out


def balanced_indices(meta: pd.DataFrame, condition: str, max_per_type: int,
                     max_per_sample_type: int, seed: int) -> np.ndarray:
    """Sample-aware deterministic downsampling for tractable, balanced CellChat runs."""
    rng = np.random.default_rng(seed)
    chosen = []
    eligible = meta[meta["condition"].eq(condition)]
    for cell_type in TARGET_TYPES:
        pool = eligible[eligible["cell_type"].eq(cell_type)]
        staged = []
        for _, sample_pool in pool.groupby("sample", observed=True):
            ids = sample_pool["row_index"].to_numpy()
            if len(ids) > max_per_sample_type:
                ids = rng.choice(ids, max_per_sample_type, replace=False)
            staged.extend(ids.tolist())
        staged = np.asarray(staged, dtype=int)
        if len(staged) > max_per_type:
            staged = rng.choice(staged, max_per_type, replace=False)
        chosen.extend(staged.tolist())
    return np.asarray(sorted(chosen), dtype=int)


def export_condition(adata: ad.AnnData, dataset: str, condition: str,
                     idx: np.ndarray) -> None:
    dest = OUT / "input" / dataset.lower() / condition.lower().replace("-", "_")
    dest.mkdir(parents=True, exist_ok=True)
    x = adata.X[idx]
    if not sparse.issparse(x):
        x = sparse.csr_matrix(x)
    x = x.astype(np.float32).T.tocoo()  # CellChat expects genes x cells.
    mmwrite(dest / "expression.mtx", x, field="real")
    pd.Series(adata.var_names).to_csv(dest / "genes.tsv", index=False, header=False)
    pd.Series(adata.obs_names[idx]).to_csv(dest / "barcodes.tsv", index=False, header=False)
    adata.obs.iloc[idx][["sample", "condition", "cell_type"]].to_csv(
        dest / "metadata.tsv", sep="\t", index_label="cell")


def process_dataset(dataset: str, path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n=== {dataset}: {path} ===")
    adata = ad.read_h5ad(path)
    adata.obs["cell_type"] = annotate_target_types(adata)
    status = epithelial_trop2_pseudobulk(adata)
    status.insert(0, "dataset", dataset)
    condition = status.set_index("sample")["condition"]
    adata.obs["condition"] = adata.obs["sample"].map(condition)
    adata.obs["row_index"] = np.arange(adata.n_obs)

    count = (adata.obs.groupby(["condition", "cell_type"], observed=True)
             .size().rename("n_cells").reset_index())
    count.insert(0, "dataset", dataset)
    print(count.to_string(index=False))
    for i, cond in enumerate(["TROP2-low", "TROP2-high"]):
        idx = balanced_indices(adata.obs, cond, args.max_cells_per_type,
                               args.max_cells_per_sample_type, args.seed + i)
        selected = adata.obs.iloc[idx].groupby("cell_type", observed=True).size()
        missing = [ct for ct in TARGET_TYPES if selected.get(ct, 0) < args.min_cells]
        if missing:
            raise ValueError(f"{dataset} {cond}: fewer than {args.min_cells} cells for {missing}")
        print(f"{cond}: {len(idx)} selected cells\n{selected.to_string()}")
        export_condition(adata, dataset, cond, idx)
    return status, count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-cells-per-type", type=int, default=1000)
    p.add_argument("--max-cells-per-sample-type", type=int, default=150)
    p.add_argument("--min-cells", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260907)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    statuses, counts = [], []
    for dataset, path in DATASETS.items():
        status, count = process_dataset(dataset, path, args)
        statuses.append(status)
        counts.append(count)
    pd.concat(statuses, ignore_index=True).to_csv(OUT / "tumor_trop2_status.csv", index=False)
    pd.concat(counts, ignore_index=True).to_csv(OUT / "cell_annotation_counts.csv", index=False)
    print(f"\nWrote focused CellChat inputs to {OUT}")


if __name__ == "__main__":
    main()

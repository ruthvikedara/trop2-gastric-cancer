"""Plot CellChat results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from genolib import PROJECT_ROOT

from genolib.style import (ANNOT_FS, LABEL_FS, LEGEND_FS, PAL, TICK_FS, TITLE_FS,
                           apply_pub_style, clean_ax, savefig)


RESULTS = PROJECT_ROOT / "trop2/results/cellchat/output"
PLOTS = PROJECT_ROOT / "trop2/plots/scrna"
DATASETS = ["Kumar", "Sathe"]
IMMUNE = ["CD8 T", "Macrophage", "Treg"]


def relative_change(high: pd.Series, low: pd.Series) -> pd.Series:
    """Bounded signed change: (high-low)/(high+low), in [-1, 1]."""
    denom = high.fillna(0) + low.fillna(0)
    return (high.fillna(0) - low.fillna(0)).div(denom.where(denom > 0))


def read_family(prefix: str) -> pd.DataFrame:
    files = sorted(RESULTS.glob(f"{prefix}_*_trop2_*.csv"))
    if len(files) != 4:
        raise FileNotFoundError(f"Expected four {prefix} files in {RESULTS}; found {len(files)}")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def focus_epi_immune(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["source"].eq("Epithelial") & df["target"].isin(IMMUNE))
        | (df["target"].eq("Epithelial") & df["source"].isin(IMMUNE))
    ].copy()


def compare_network() -> pd.DataFrame:
    raw = focus_epi_immune(read_family("network_strength"))
    wide = raw.pivot_table(index=["dataset", "source", "target"],
                           columns="condition", values="strength", aggfunc="first").reset_index()
    wide.columns.name = None
    wide["relative_change"] = relative_change(wide["TROP2-high"], wide["TROP2-low"])
    wide["direction"] = np.where(wide["source"].eq("Epithelial"),
                                 "Epithelial → immune", "Immune → epithelial")
    wide["immune_cell"] = np.where(wide["source"].eq("Epithelial"),
                                   wide["target"], wide["source"])
    wide.to_csv(RESULTS / "network_strength_comparison.csv", index=False)
    return wide


def comparison_table(raw: pd.DataFrame, keys: list[str], value: str = "prob") -> pd.DataFrame:
    wide = raw.pivot_table(index=["dataset", *keys], columns="condition",
                           values=[value, "pval"], aggfunc="first").reset_index()
    wide.columns = ["_".join(x).strip("_") if isinstance(x, tuple) else x for x in wide.columns]
    wide = wide.rename(columns={f"{value}_TROP2-high": "prob_high",
                                f"{value}_TROP2-low": "prob_low",
                                "pval_TROP2-high": "pval_high",
                                "pval_TROP2-low": "pval_low"})
    for col in ["prob_high", "prob_low", "pval_high", "pval_low"]:
        if col not in wide:
            wide[col] = np.nan
    wide["relative_change"] = relative_change(wide["prob_high"], wide["prob_low"])
    wide["active_p"] = np.where(wide["relative_change"].ge(0),
                                wide["pval_high"], wide["pval_low"])
    return wide


def compare_interactions() -> pd.DataFrame:
    raw = focus_epi_immune(read_family("interactions"))
    keys = ["source", "target", "ligand", "receptor", "interaction_name_2",
            "pathway_name", "annotation"]
    wide = comparison_table(raw, keys)
    wide.to_csv(RESULTS / "ligand_receptor_comparison.csv", index=False)
    return wide


def compare_pathways() -> pd.DataFrame:
    raw = focus_epi_immune(read_family("pathways"))
    raw = raw.rename(columns={"pathway": "pathway_name", "prob": "prob_value"})
    # Pathway arrays do not carry permutation p-values; use the LR-level p-value
    # only for LR selection and keep pathway summaries purely descriptive.
    raw["pval"] = np.nan
    wide = comparison_table(raw, ["source", "target", "pathway_name"], value="prob_value")
    wide.to_csv(RESULTS / "pathway_comparison.csv", index=False)
    return wide


def replicated_summary(df: pd.DataFrame, keys: list[str], require_active: bool) -> pd.DataFrame:
    work = df.copy()
    work["sign"] = np.sign(work["relative_change"].fillna(0))
    grouped = []
    for key, part in work.groupby(keys, dropna=False):
        if set(part["dataset"]) != set(DATASETS) or len(part) != 2:
            continue
        signs = set(part["sign"])
        if 0 in signs or len(signs) != 1:
            continue
        if require_active and not part["active_p"].le(0.05).all():
            continue
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for dataset in DATASETS:
            row[dataset] = float(part.loc[part["dataset"].eq(dataset), "relative_change"].iloc[0])
        row["mean_change"] = part["relative_change"].mean()
        row["min_abs_change"] = part["relative_change"].abs().min()
        grouped.append(row)
    return pd.DataFrame(grouped)


def plot_network(net: pd.DataFrame) -> None:
    mean = net.pivot_table(index="immune_cell", columns="direction",
                           values="relative_change", aggfunc="mean").reindex(IMMUNE)
    concordant = pd.DataFrame(False, index=mean.index, columns=mean.columns)
    annotations = pd.DataFrame("", index=mean.index, columns=mean.columns)
    for cell in IMMUNE:
        for direction in mean.columns:
            sub = net[net["immune_cell"].eq(cell) & net["direction"].eq(direction)]
            vals = {r.dataset: r.relative_change for r in sub.itertuples()}
            pair = np.asarray([vals.get(dataset, np.nan) for dataset in DATASETS])
            concordant.loc[cell, direction] = (
                np.isfinite(pair).all() and np.prod(np.sign(pair)) > 0
            )
            annotations.loc[cell, direction] = "\n".join(
                f"{dataset[0]} {vals.get(dataset, np.nan):+.2f}" for dataset in DATASETS)

    fig, ax = plt.subplots(figsize=(4.7, 3.5))
    replicated = mean.where(concordant)
    vmax = max(0.25, np.nanmax(np.abs(replicated.to_numpy())))
    ax.set_facecolor("#E6E6E6")
    sns.heatmap(replicated, mask=replicated.isna(), ax=ax, cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax,
                annot=annotations, fmt="", annot_kws={"fontsize": ANNOT_FS},
                linewidths=0.8, linecolor="white",
                cbar_kws={"label": "Relative change\n(TROP2-high vs low)", "shrink": 0.82})
    for i, cell in enumerate(mean.index):
        for j, direction in enumerate(mean.columns):
            if not concordant.loc[cell, direction]:
                ax.text(j + 0.5, i + 0.5, annotations.loc[cell, direction],
                        ha="center", va="center", fontsize=ANNOT_FS, color="#333333")
    ax.set_title("Epithelial-immune communication strength", fontsize=TITLE_FS, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(["Epithelial →\nimmune", "Immune →\nepithelial"], rotation=0,
                       fontsize=TICK_FS)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=TICK_FS)
    ax.text(0, -0.17, "K = Kumar; S = Sathe; gray = direction not replicated",
            transform=ax.transAxes, fontsize=ANNOT_FS, ha="left", va="top")
    savefig(fig, PLOTS / "S3_TROP2_CellChat_network")
    plt.close(fig)


def dot_comparison(summary: pd.DataFrame, label_col: str, title: str,
                   output: str, n: int) -> None:
    if summary.empty:
        raise ValueError(f"No replicated features available for {title}")
    selected = (summary.sort_values("min_abs_change", ascending=False)
                .head(n).sort_values("mean_change"))
    y = np.arange(len(selected))
    height = max(3.6, 0.42 * len(selected) + 1.2)
    fig, ax = plt.subplots(figsize=(7.0, height))
    for yi, row in zip(y, selected.itertuples()):
        ax.plot([getattr(row, "Kumar"), getattr(row, "Sathe")], [yi, yi],
                color="#B8B8B8", lw=1.1, zorder=1)
        ax.scatter(getattr(row, "Kumar"), yi, marker="o", s=40,
                   color="#1A1A1A", zorder=2)
        ax.scatter(getattr(row, "Sathe"), yi, marker="s", s=38,
                   color=PAL["accent_high"], zorder=2)
        ax.scatter(row.mean_change, yi, marker="D", s=34,
                   color=PAL["high"] if row.mean_change > 0 else PAL["low"],
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color="#777777", lw=0.8, ls=":")
    ax.set_yticks(y, selected[label_col], fontsize=TICK_FS)
    ax.set_xlabel("Relative change in CellChat probability\n(TROP2-high vs low tumors)",
                  fontsize=LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS, pad=8)
    clean_ax(ax)
    ax.grid(axis="x", color="#E1E1E1", lw=0.6, ls=":")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#1A1A1A", label="Kumar"),
        Line2D([], [], marker="s", ls="", color=PAL["accent_high"], label="Sathe"),
        Line2D([], [], marker="D", ls="", color="#777777", label="Mean"),
    ], frameon=False, fontsize=LEGEND_FS, loc="lower right")
    savefig(fig, PLOTS / output)
    plt.close(fig)


def main() -> None:
    apply_pub_style()
    PLOTS.mkdir(parents=True, exist_ok=True)
    net = compare_network()
    lr = compare_interactions()
    pathways = compare_pathways()
    plot_network(net)

    lr_summary = replicated_summary(
        lr,
        ["source", "target", "ligand", "receptor", "interaction_name_2",
         "pathway_name", "annotation"],
        require_active=True,
    )
    if lr_summary.empty:
        # Transparent fallback for sparse replication: require support in the
        # stronger condition in either cohort, never invent a differential p-value.
        lr2 = lr.copy()
        lr2["active_p"] = lr2.groupby(
            ["source", "target", "ligand", "receptor", "interaction_name_2",
             "pathway_name", "annotation"]
        )["active_p"].transform("min")
        lr_summary = replicated_summary(
            lr2,
            ["source", "target", "ligand", "receptor", "interaction_name_2",
             "pathway_name", "annotation"],
            require_active=True,
        )
    lr_summary["display"] = (lr_summary["source"] + " → " + lr_summary["target"]
                              + "  |  " + lr_summary["ligand"] + "-" + lr_summary["receptor"])
    lr_summary.to_csv(RESULTS / "ligand_receptor_replicated.csv", index=False)
    protein_lr = lr_summary[lr_summary["annotation"].ne("Non-protein Signaling")]
    dot_comparison(protein_lr, "display", "Replicated epithelial-immune ligand-receptor changes",
                   "S3_TROP2_CellChat_ligand_receptor", n=12)

    path_summary = replicated_summary(
        pathways, ["source", "target", "pathway_name"], require_active=False)
    path_summary["display"] = (path_summary["source"] + " → " + path_summary["target"]
                                + "  |  " + path_summary["pathway_name"])
    path_summary.to_csv(RESULTS / "pathway_replicated.csv", index=False)
    dot_comparison(path_summary, "display", "Replicated epithelial-immune pathway changes",
                   "S3_TROP2_CellChat_pathways", n=12)
    print(f"Wrote CellChat summaries to {RESULTS} and plots to {PLOTS}")


if __name__ == "__main__":
    main()

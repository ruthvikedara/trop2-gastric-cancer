"""TROP2 expression by Lauren subtype."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from genolib import PROJECT_ROOT
from style import (
    ANNOT_FS,
    PAL,
    apply_pub_style,
    axis_label,
    clean_ax,
    fmt_p,
    savefig,
)


BASE = PROJECT_ROOT
CLIN = BASE / "data" / "clinical" / "gastric" / "processed"
PROC = BASE / "data" / "processed" / "gastric"
OUT = Path(__file__).resolve().parent.parent / "plots" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

GENE = "TACSTD2"
LAUREN3 = ["Intestinal", "Diffuse", "Mixed"]
DATASETS = {
    "TCGA-STAD": dict(
        clinical="TCGA_clinical.csv",
        expr="TCGA_STAD_for_xcell.txt",
        id_col="patient_id",
        filt=None,
    ),
    "GSE66229": dict(
        clinical="GSE66229_clinical.csv",
        expr="ACRG_GSE66229_for_xcell.txt",
        id_col="gsm_id",
        filt=("tissue_type", "Tumor"),
    ),
    "GSE15459": dict(
        clinical="GSE15459_clinical.csv",
        expr="ACRG_GSE15459_for_xcell.txt",
        id_col="gsm_id",
        filt=None,
    ),
    "GSE34942": dict(
        clinical="GSE34942_clinical.csv",
        expr="ACRG_GSE34942_for_xcell.txt",
        id_col="gsm_id",
        filt=None,
    ),
}


def load_gene_row(path, gene):
    with open(path) as handle:
        samples = handle.readline().rstrip("\n").split("\t")[1:]
        for line in handle:
            tab = line.index("\t")
            if line[:tab] == gene:
                values = line[tab + 1:].rstrip("\n").split("\t")
                return pd.Series(
                    [float(v) if v.strip() else np.nan for v in values],
                    index=samples,
                    name="expression",
                )
    raise ValueError(f"{gene} not found in {path}")


def normalize_lauren(value):
    if pd.isna(value):
        return None
    return {
        "intestinal": "Intestinal",
        "diffuse": "Diffuse",
        "mixed": "Mixed",
    }.get(str(value).strip().lower())


def load_data():
    frames = []
    for cohort, cfg in DATASETS.items():
        clinical = pd.read_csv(CLIN / cfg["clinical"])
        if cfg["filt"]:
            column, value = cfg["filt"]
            clinical = clinical[
                clinical[column].astype(str).str.lower() == value.lower()
            ]
        clinical = clinical.set_index(cfg["id_col"])
        expression = load_gene_row(PROC / cfg["expr"], GENE)

        if cohort == "TCGA-STAD":
            clinical.index = clinical.index.astype(str).str[:12]
            expression.index = expression.index.astype(str).str[:12]

        common = clinical.index.intersection(expression.dropna().index)
        frame = pd.DataFrame(
            {
                "cohort": cohort,
                "lauren": clinical.loc[common, "lauren"].map(normalize_lauren),
                "expression": expression.loc[common],
            }
        ).dropna()
        sd = frame["expression"].std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            print(f"[{cohort}] constant/invalid TROP2 expression - skipped")
            continue
        frame["trop2_z"] = (frame["expression"] - frame["expression"].mean()) / sd
        frames.append(frame.reset_index(drop=True))
        counts = frame["lauren"].value_counts()
        print(
            f"[{cohort}] n={len(frame)} | "
            + ", ".join(f"{group}={counts.get(group, 0)}" for group in LAUREN3)
        )
    return pd.concat(frames, ignore_index=True)


def bh_adjust(p_values):
    """Benjamini-Hochberg FDR over the three pairwise Lauren comparisons."""
    from statsmodels.stats.multitest import multipletests
    return multipletests(np.asarray(p_values, dtype=float), method="fdr_bh")[1]


def add_violin_box(ax, groups, colors):
    positions = np.arange(len(groups))
    arrays = [values for _, values in groups]
    violins = ax.violinplot(
        arrays,
        positions=positions,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.78)
    box = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.3},
        whiskerprops={"color": "black", "linewidth": 0.9},
        capprops={"color": "black", "linewidth": 0.9},
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.9},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{label}\n(n={len(values)})" for label, values in groups])
    ax.axhline(0, color="#B8B8B8", linewidth=0.8, linestyle=":")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, linestyle=":", alpha=0.8)
    clean_ax(ax)


def main():
    apply_pub_style()
    data = load_data()

    arrays = {
        group: data.loc[data["lauren"] == group, "trop2_z"].to_numpy()
        for group in LAUREN3
    }
    kw_stat, kw_p = kruskal(*(arrays[group] for group in LAUREN3))

    comparisons = [
        ("Intestinal", "Diffuse"),
        ("Intestinal", "Mixed"),
        ("Diffuse", "Mixed"),
    ]
    pairwise = []
    for first, second in comparisons:
        stat, p_value = mannwhitneyu(
            arrays[first], arrays[second], alternative="two-sided"
        )
        pairwise.append((first, second, stat, p_value))
    adjusted = bh_adjust([row[3] for row in pairwise])

    data["lauren_binary"] = np.where(
        data["lauren"] == "Intestinal", "Intestinal", "Non-intestinal"
    )
    binary = {
        group: data.loc[data["lauren_binary"] == group, "trop2_z"].to_numpy()
        for group in ["Intestinal", "Non-intestinal"]
    }
    binary_stat, binary_p = mannwhitneyu(
        binary["Intestinal"], binary["Non-intestinal"], alternative="two-sided"
    )

    stats_rows = [
        {
            "analysis": "three_class_global",
            "comparison": "Intestinal vs Diffuse vs Mixed",
            "test": "Kruskal-Wallis",
            "statistic": kw_stat,
            "p_value": kw_p,
            "p_adjusted": np.nan,
        }
    ]
    for (first, second, stat, p_value), p_adj in zip(pairwise, adjusted):
        stats_rows.append(
            {
                "analysis": "three_class_pairwise",
                "comparison": f"{first} vs {second}",
                "test": "Mann-Whitney U",
                "statistic": stat,
                "p_value": p_value,
                "p_adjusted": p_adj,
            }
        )
    stats_rows.append(
        {
            "analysis": "binary",
            "comparison": "Intestinal vs Non-intestinal (Diffuse + Mixed)",
            "test": "Mann-Whitney U",
            "statistic": binary_stat,
            "p_value": binary_p,
            "p_adjusted": np.nan,
        }
    )
    for cohort_or_pooled, subset in [
        *[(cohort, data[data["cohort"] == cohort]) for cohort in DATASETS],
        ("Pooled", data),
    ]:
        for group in LAUREN3:
            values = subset.loc[subset["lauren"] == group, "trop2_z"]
            stats_rows.append(
                {
                    "analysis": "summary",
                    "comparison": f"{cohort_or_pooled}: {group}",
                    "test": "descriptive",
                    "statistic": values.median() if len(values) else np.nan,
                    "p_value": np.nan,
                    "p_adjusted": np.nan,
                    "n": len(values),
                    "median_trop2_z": values.median() if len(values) else np.nan,
                }
            )
    pd.DataFrame(stats_rows).to_csv(
        OUT / "F1C_TROP2_by_Lauren_stats.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.9), sharey=True)
    add_violin_box(
        axes[0],
        [(group, arrays[group]) for group in LAUREN3],
        [PAL[group] for group in LAUREN3],
    )
    axes[0].set_title("Lauren classes")
    axes[0].set_ylabel(
        axis_label("TROP2 expression", transform="z-score within cohort")
    )
    axes[0].text(
        0.5,
        0.98,
        f"Kruskal-Wallis {fmt_p(kw_p)}",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=ANNOT_FS,
    )

    add_violin_box(
        axes[1],
        [(group, binary[group]) for group in ["Intestinal", "Non-intestinal"]],
        [PAL["Intestinal"], PAL["accent_low"]],
    )
    axes[1].set_title("Binary clinical grouping")
    axes[1].text(
        0.5,
        0.98,
        f"Mann-Whitney {fmt_p(binary_p)}",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=ANNOT_FS,
    )

    fig.suptitle("TROP2 expression by Lauren histologic subtype", fontweight="bold")
    fig.text(
        0.5,
        0.005,
        "Four cohorts pooled after within-cohort standardization; non-intestinal = diffuse + mixed",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=1.2)
    savefig(fig, OUT / "F1C_TROP2_by_Lauren")

    print(f"Three-class: H={kw_stat:.2f}, {fmt_p(kw_p)}")
    for (first, second, _, p_value), p_adj in zip(pairwise, adjusted):
        print(f"  {first} vs {second}: raw {fmt_p(p_value)}, BH FDR={p_adj:.3g}")
    print(f"Binary: U={binary_stat:.1f}, {fmt_p(binary_p)}")


if __name__ == "__main__":
    main()

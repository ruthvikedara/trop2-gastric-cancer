"""Build pathway_sets.yaml and putative_genes.csv from the supplementary tables."""
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).parent
SRC = HERE / "geac_ebv_suptables_v1.1.xlsx"

# Table S2 -> pathway_sets.yaml
df = pd.read_excel(SRC, sheet_name="Table S2 Pathway gene sets")
sections: dict[str, list[dict]] = {}
section = None
for _, r in df.iterrows():
    name, src = r["Pathway / Signature"], r["Genes / Source ID"]
    if pd.isna(name):
        continue  # spacer row
    if pd.isna(src):  # section header row
        section = str(name).strip()
        sections[section] = []
        continue
    entry: dict = {"name": str(name).strip()}
    src = str(src).strip()
    if src.startswith("KEGG"):
        entry["source"] = "kegg"
        entry["kegg_id"] = src.split()[-1]  # e.g. hsa04630
    elif src.startswith("MSigDB Hallmark"):
        entry["source"] = "hallmark"
        entry["hallmark_name"] = src.split(":", 1)[1].strip()
    else:
        entry["source"] = "manual"
        entry["genes"] = [g.strip() for g in src.split(",")]
    entry["citation"] = str(r["Full citation"]).strip()
    entry["doi"] = str(r["DOI"]).strip()
    sections[section].append(entry)

doc = {
    "version": "1.1",
    "species": "Homo sapiens",
    "provenance": (
        "Standard pathway gene sets, generated from Table S2 by parse_suptables.py"
    ),
    "source_types": {
        "manual": "explicit curated gene list (literature signature)",
        "kegg": "KEGG pathway id; R: msigdbr(category='C2', subcategory='CP:KEGG_LEGACY'); "
                "Python: fetch gene list by hsa id (KEGG REST / gseapy / bioservices)",
        "hallmark": "MSigDB Hallmark set; R: msigdbr(category='H')",
    },
    "sections": sections,
}
(HERE / "pathway_sets.yaml").write_text(
    yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120)
)

# Table S1 -> putative_genes.csv
g = pd.read_excel(SRC, sheet_name="Table S1 Putative genes")
rows, cat = [], None
for _, r in g.iterrows():
    if pd.isna(r["Gene symbol"]):  # category header / spacer / footer note
        c = r["Category"]
        if isinstance(c, str) and not c.startswith("Panel comprises"):
            cat = c.strip()
        continue
    rows.append(
        {
            "category": cat,
            "gene_symbol": str(r["Gene symbol"]).strip(),
            "gene_name": str(r["Gene name"]).strip() if isinstance(r["Gene name"], str) else "",
            "relevance": (
                str(r["Relevance to this study"]).strip()
                if isinstance(r["Relevance to this study"], str)
                else ""
            ),
        }
    )
out = pd.DataFrame(rows)
out.to_csv(HERE / "putative_genes.csv", index=False)

n_sets = sum(len(v) for v in sections.values())
n_manual = sum(1 for v in sections.values() for e in v if e["source"] == "manual")
print(f"pathway_sets.yaml : {len(sections)} sections, {n_sets} sets ({n_manual} manual)")
print(f"putative_genes.csv: {len(out)} genes in {out['category'].nunique()} categories")

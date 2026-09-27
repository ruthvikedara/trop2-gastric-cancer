# Standard pathway gene-set definitions. Requires msigdbr, dplyr, stringr.

## Pull KEGG_LEGACY and Hallmark gene sets from msigdbr
kegg_sets <- msigdbr(species = "Homo sapiens", category = "C2", subcategory = "CP:KEGG_LEGACY")
hallmark_sets <- msigdbr(species = "Homo sapiens", category = "H")
get_kegg_geneset <- function(kegg_sets, pattern) {
  kegg_sets %>% filter(str_detect(gs_name, pattern)) %>% pull(gene_symbol) %>% unique()
}
get_hallmark_geneset <- function(hallmark_sets, pattern) {
  hallmark_sets %>% filter(str_detect(gs_name, pattern)) %>% pull(gene_symbol) %>% unique()
}
pathway_list <- list()
## 1. IDO1 mechanism pathways
pathway_list[["IFN-gamma response"]] <- c("CXCL9", "CXCL10", "IFNG", "IDO1", "HLA-DRA", "STAT1")
pathway_list[["JAK-STAT signaling"]] <- get_kegg_geneset(kegg_sets, "JAK_STAT_SIGNALING_PATHWAY")
pathway_list[["Tryptophan metabolism"]] <- get_kegg_geneset(kegg_sets, "^KEGG_TRYPTOPHAN_METABOLISM$")
pathway_list[["AHR signaling"]] <- c("AHR", "ARNT", "CYP1A1", "CYP1B1", "NQO1", "TIPARP")
## 2. CLDN18 / epithelial-junction mechanism pathways
pathway_list[["Hippo-YAP/TEAD signaling"]] <- get_kegg_geneset(kegg_sets, "^KEGG_HIPPO_SIGNALING_PATHWAY$")
pathway_list[["Tight junction"]] <- get_kegg_geneset(kegg_sets, "^KEGG_TIGHT_JUNCTION$")
pathway_list[["EMT"]] <- get_hallmark_geneset(hallmark_sets, "EPITHELIAL_MESENCHYMAL_TRANSITION")
pathway_list[["Wnt signaling"]] <- get_kegg_geneset(kegg_sets, "^KEGG_WNT_SIGNALING_PATHWAY$")
## 3. Broader oncogenic signaling pathways
pathway_list[["NF-kB signaling"]] <- get_kegg_geneset(kegg_sets, "^KEGG_NF_KAPPA_B_SIGNALING_PATHWAY$")
pathway_list[["PI3K-Akt signaling"]] <- get_kegg_geneset(kegg_sets, "^KEGG_PI3K_AKT_SIGNALING_PATHWAY$")
pathway_list[["mTOR signaling"]] <- get_kegg_geneset(kegg_sets, "^KEGG_MTOR_SIGNALING_PATHWAY$")
pathway_list[["TNF signaling"]] <- get_kegg_geneset(kegg_sets, "TNF_SIGNALING_PATHWAY")
pathway_list[["VEGF signaling"]] <- get_kegg_geneset(kegg_sets, "^KEGG_VEGF_SIGNALING_PATHWAY$")
# pathway_list[["Angiogenesis"]] <- get_hallmark_geneset(hallmark_sets, "^HALLMARK_ANGIOGENESIS$")

pathway_list[["MAPK signaling"]] <- get_kegg_geneset(kegg_sets, "^KEGG_MAPK_SIGNALING_PATHWAY$")
pathway_list[["p53 signaling"]] <- get_kegg_geneset(kegg_sets, "^KEGG_P53_SIGNALING_PATHWAY$")
pathway_list[["Cell cycle"]] <- get_kegg_geneset(kegg_sets, "^KEGG_CELL_CYCLE$")
pathway_list[["Apoptosis"]] <- get_kegg_geneset(kegg_sets, "^KEGG_APOPTOSIS$")
pathway_list[["Notch signaling"]] <- get_kegg_geneset(kegg_sets, "^KEGG_NOTCH_SIGNALING_PATHWAY$")
## 4. Broader immune pathway context
pathway_list[["Tumor inflammation signature"]] <- c(
  "CCL5", "CD27", "CD274", "CD276", "CD8A", "CMKLR1", "CXCL9", "CXCR6",
  "HLA-DQA1", "HLA-DRB1", "HLA-E", "IDO1", "LAG3", "NKG7", "PDCD1LG2",
  "PSMB10", "STAT1", "TIGIT"
)
pathway_list[["Cytolytic activity"]] <- c("GZMA", "PRF1")
pathway_list[["Antigen presentation (MHC-I)"]] <- get_kegg_geneset(kegg_sets, "^KEGG_ANTIGEN_PROCESSING_AND_PRESENTATION$")
pathway_list[["NK cell-mediated cytotoxicity"]] <- get_kegg_geneset(kegg_sets, "^KEGG_NATURAL_KILLER_CELL_MEDIATED_CYTOTOXICITY$")
pathway_list[["T cell receptor signaling"]] <- get_kegg_geneset(kegg_sets, "^KEGG_T_CELL_RECEPTOR_SIGNALING_PATHWAY$")
pathway_list[["TGF-beta / immune exclusion"]] <- c("TGFB1", "TGFBR2", "ACTA2", "COL1A1", "COL1A2", "PDGFRB")
pathway_list[["Complement and coagulation"]] <- get_kegg_geneset(kegg_sets, "^KEGG_COMPLEMENT_AND_COAGULATION_CASCADES$")
pathway_list[["cGAS-STING"]] <- get_kegg_geneset(kegg_sets, "^KEGG_CYTOSOLIC_DNA_SENSING_PATHWAY$")
pathway_list[["Chemokine signaling"]] <- get_kegg_geneset(kegg_sets, "^KEGG_CHEMOKINE_SIGNALING_PATHWAY$")
## 5. Metabolic pathway context
pathway_list[["Hypoxia"]] <- get_hallmark_geneset(hallmark_sets, "^HALLMARK_HYPOXIA$")
pathway_list[["Oxidative phosphorylation"]] <- get_kegg_geneset(kegg_sets, "^KEGG_OXIDATIVE_PHOSPHORYLATION$")
pathway_list[["Glycolysis / gluconeogenesis"]] <- get_kegg_geneset(kegg_sets, "^KEGG_GLYCOLYSIS_GLUCONEOGENESIS$")
pathway_list[["Fatty acid metabolism"]] <- get_hallmark_geneset(hallmark_sets, "^HALLMARK_FATTY_ACID_METABOLISM$")
pathway_list[["DNA repair"]] <- get_hallmark_geneset(hallmark_sets, "^HALLMARK_DNA_REPAIR$")

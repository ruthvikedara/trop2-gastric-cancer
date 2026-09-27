#!/usr/bin/env Rscript

# CellChat inference.

project_root <- normalizePath(getwd())
local_lib <- file.path(project_root, "data/cache/R/cellchat")
if (dir.exists(local_lib)) .libPaths(c(local_lib, .libPaths()))

suppressPackageStartupMessages({
  library(CellChat)
  library(Matrix)
})

input_root <- file.path(project_root, "trop2/results/cellchat/input")
output_root <- file.path(project_root, "trop2/results/cellchat/output")
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

target_order <- c("Epithelial", "CD8 T", "Macrophage", "Treg")

read_input <- function(path) {
  message("Reading ", path)
  expression <- as(readMM(file.path(path, "expression.mtx")), "CsparseMatrix")
  genes <- readLines(file.path(path, "genes.tsv"))
  cells <- readLines(file.path(path, "barcodes.tsv"))
  rownames(expression) <- genes
  colnames(expression) <- cells
  metadata <- read.delim(file.path(path, "metadata.tsv"), row.names = 1,
                         check.names = FALSE, stringsAsFactors = FALSE)
  stopifnot(identical(colnames(expression), rownames(metadata)))
  metadata$samples <- factor(metadata$sample)
  metadata$cell_type <- factor(metadata$cell_type, levels = target_order)
  list(expression = expression, metadata = metadata)
}

matrix_to_long <- function(x, value_name) {
  out <- as.data.frame(as.table(x), stringsAsFactors = FALSE)
  names(out) <- c("source", "target", value_name)
  out
}

pathway_to_long <- function(x) {
  if (length(x) == 0L) return(data.frame())
  out <- as.data.frame(as.table(x), stringsAsFactors = FALSE)
  names(out) <- c("source", "target", "pathway", "prob")
  out
}

run_one <- function(dataset, condition) {
  slug <- paste(tolower(dataset), gsub("-", "_", tolower(condition)), sep = "_")
  input_dir <- file.path(input_root, tolower(dataset),
                         gsub("-", "_", tolower(condition)))
  object_file <- file.path(output_root, paste0("cellchat_", slug, ".rds"))
  expected_csv <- file.path(output_root, paste0(
    c("interactions_", "network_strength_", "pathways_"), slug, ".csv"
  ))
  input_mtime <- max(file.info(list.files(input_dir, full.names = TRUE))$mtime)
  cache_current <- file.exists(object_file) && all(file.exists(expected_csv)) &&
    file.info(object_file)$mtime >= input_mtime
  if (cache_current) {
    message("Using existing object: ", object_file)
    return(readRDS(object_file))
  }

  dat <- read_input(input_dir)
  cellchat <- createCellChat(object = dat$expression, meta = dat$metadata,
                             group.by = "cell_type")
  cellchat@DB <- CellChatDB.human
  cellchat <- subsetData(cellchat)
  cellchat <- identifyOverExpressedGenes(cellchat, do.fast = FALSE)
  cellchat <- identifyOverExpressedInteractions(cellchat)
  cellchat <- computeCommunProb(
    cellchat,
    type = "truncatedMean",
    trim = 0.1,
    raw.use = TRUE,
    population.size = FALSE,
    nboot = 100,
    seed.use = 20260907
  )
  cellchat <- filterCommunication(cellchat, min.cells = 30)
  cellchat <- computeCommunProbPathway(cellchat)
  cellchat <- aggregateNet(cellchat)

  interactions <- subsetCommunication(cellchat, thresh = 1)
  interactions$dataset <- dataset
  interactions$condition <- condition
  write.csv(interactions,
            file.path(output_root, paste0("interactions_", slug, ".csv")),
            row.names = FALSE)

  strengths <- merge(matrix_to_long(cellchat@net$count, "n_interactions"),
                     matrix_to_long(cellchat@net$weight, "strength"),
                     by = c("source", "target"), all = TRUE)
  strengths$dataset <- dataset
  strengths$condition <- condition
  write.csv(strengths,
            file.path(output_root, paste0("network_strength_", slug, ".csv")),
            row.names = FALSE)

  pathways <- pathway_to_long(cellchat@netP$prob)
  if (nrow(pathways)) {
    pathways$dataset <- dataset
    pathways$condition <- condition
    write.csv(pathways,
              file.path(output_root, paste0("pathways_", slug, ".csv")),
              row.names = FALSE)
  }
  saveRDS(cellchat, object_file, compress = "xz")
  cellchat
}

set.seed(20260907)
for (dataset in c("Kumar", "Sathe")) {
  for (condition in c("TROP2-low", "TROP2-high")) {
    message("\n=== ", dataset, " / ", condition, " ===")
    run_one(dataset, condition)
    invisible(gc())
  }
}

capture.output(sessionInfo(), file = file.path(output_root, "cellchat_session_info.txt"))
message("\nCellChat results written to ", output_root)

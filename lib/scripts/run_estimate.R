# ESTIMATE stromal, immune and purity scores.

suppressPackageStartupMessages(library(tidyestimate))

PROC <- "data/processed/gastric"
OUT  <- "data/clinical/gastric/processed"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

COHORTS <- c(
  "TCGA-STAD" = "TCGA_STAD_for_xcell.txt",
  "GSE66229"  = "ACRG_GSE66229_for_xcell.txt",
  "GSE15459"  = "ACRG_GSE15459_for_xcell.txt",
  "GSE34942"  = "ACRG_GSE34942_for_xcell.txt",
  "GSE35809"  = "ACRG_GSE35809_for_xcell.txt",
  "GSE51105"  = "ACRG_GSE51105_for_xcell.txt",
  "GSE54129"  = "ACRG_GSE54129_for_xcell.txt",
  "GSE57303"  = "ACRG_GSE57303_for_xcell.txt",
  "GSE84437"  = "ACRG_GSE84437_for_xcell.txt",
  "GSE118916" = "ACRG_GSE118916_for_xcell.txt",
  "GSE26253"  = "ACRG_GSE26253_for_xcell.txt"
)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) COHORTS <- COHORTS[names(COHORTS) %in% args]

for (cohort in names(COHORTS)) {
  path <- file.path(PROC, COHORTS[[cohort]])
  cat(sprintf("\n[%s] %s\n", cohort, path))
  if (!file.exists(path)) { cat("    MISSING - skipped\n"); next }

  m <- read.table(path, header = TRUE, row.names = 1, sep = "\t",
                  check.names = FALSE, stringsAsFactors = FALSE)
  m <- as.matrix(m); mode(m) <- "numeric"
  m <- m[!duplicated(rownames(m)), , drop = FALSE]
  m <- m[stats::complete.cases(m), , drop = FALSE]  # drop NA genes
  cat(sprintf("    %d genes x %d samples\n", nrow(m), ncol(m)))

  res <- tryCatch({
    filt <- tidyestimate::filter_common_genes(m, id = "hgnc_symbol", tidy = FALSE,
                                              tell_missing = FALSE)
    tidyestimate::estimate_score(filt, is_affymetrix = FALSE)
  }, error = function(e) { cat(sprintf("    ESTIMATE failed: %s\n", conditionMessage(e))); NULL })
  if (is.null(res)) next

  res <- as.data.frame(res, stringsAsFactors = FALSE)
  out <- data.frame(
    sample_id      = as.character(res$sample),
    stromal_score  = res$stromal,
    immune_score   = res$immune,
    estimate_score = res$estimate,
    stringsAsFactors = FALSE
  )
  out_path <- file.path(OUT, sprintf("%s_estimate.csv", cohort))
  write.csv(out, out_path, row.names = FALSE)
  cat(sprintf("    -> %s (%d samples)\n", basename(out_path), nrow(out)))
}
cat("\nDONE\n")

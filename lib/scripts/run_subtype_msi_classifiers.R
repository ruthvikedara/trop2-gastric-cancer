# Molecular subtype (GCclassifier) and MSI (PreMSIm) calls from expression.

suppressPackageStartupMessages({
  library(GCclassifier)
  library(PreMSIm)
})

PROC <- "data/processed/gastric"
OUT  <- "data/clinical/gastric/processed"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# cohort -> processed expression matrix filename
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

read_expr <- function(path) {
  m <- read.table(path, header = TRUE, row.names = 1, sep = "\t",
                  check.names = FALSE, stringsAsFactors = FALSE)
  m <- as.matrix(m); mode(m) <- "numeric"
  m[!duplicated(rownames(m)), , drop = FALSE]
}

# classifyGC returns a data.frame with columns: sample, subtype, <per-class probs>.
# Normalize to sample_id + method-prefixed columns so the three systems merge cleanly.
# NOTE: GCclassifier rejects negative inputs; pass it a non-negatively-shifted matrix
# (it z-normalizes per gene internally, so a global shift leaves predictions identical).
run_subtype <- function(expr, method) {
  res <- tryCatch(
    classifyGC(Expr = expr, method = method, idType = "SYMBOL"),
    error = function(e) { cat(sprintf("    %s failed: %s\n", method, conditionMessage(e))); NULL }
  )
  if (is.null(res)) return(NULL)
  res <- as.data.frame(res, stringsAsFactors = FALSE)
  names(res)[names(res) == "sample"] <- "sample_id"
  non_id <- setdiff(names(res), "sample_id")
  names(res)[match(non_id, names(res))] <- paste0(method, "_", non_id)
  res
}

run_msi <- function(path, sample_ids) {
  res <- tryCatch({
    input <- data_pre(path, type = "Symbol")
    msi_pre(input)
  }, error = function(e) { cat(sprintf("    PreMSIm failed: %s\n", conditionMessage(e))); NULL })
  if (is.null(res)) return(NULL)
  res <- as.data.frame(res, stringsAsFactors = FALSE)
  # msi_pre -> columns Sample, MSI_status (1 = MSI-H, 0 = MSS)
  data.frame(
    sample_id   = as.character(res$Sample),
    msi_premsim = ifelse(res$MSI_status == 1, "MSI-H", "MSS"),
    stringsAsFactors = FALSE
  )
}

for (cohort in names(COHORTS)) {
  path <- file.path(PROC, COHORTS[[cohort]])
  cat(sprintf("\n[%s] %s\n", cohort, path))
  if (!file.exists(path)) { cat("    MISSING - skipped\n"); next }
  expr <- read_expr(path)
  cat(sprintf("    %d genes x %d samples\n", nrow(expr), ncol(expr)))

  # GCclassifier requires non-negative, NA-free input. Shift to non-negative
  # (z-norm-invariant) and drop genes with any NA (it imputes missing signature
  # genes itself; TCGA carries ~8% NA cells but 82% of genes are complete).
  expr_pos <- if (min(expr, na.rm = TRUE) < 0) expr - min(expr, na.rm = TRUE) else expr
  expr_pos <- expr_pos[stats::complete.cases(expr_pos), , drop = FALSE]

  # --- subtypes: merge TCGA + ACRG + EMP on sample_id ---
  subs <- Filter(Negate(is.null), lapply(c("TCGA", "ACRG", "EMP"), function(m) run_subtype(expr_pos, m)))
  if (length(subs) > 0) {
    merged <- Reduce(function(a, b) merge(a, b, by = "sample_id", all = TRUE), subs)
    out_sub <- file.path(OUT, sprintf("%s_subtypes_predicted.csv", cohort))
    write.csv(merged, out_sub, row.names = FALSE)
    cat(sprintf("    -> %s (%d samples)\n", basename(out_sub), nrow(merged)))
  }

  # --- MSI ---
  msi <- run_msi(path, colnames(expr))
  if (!is.null(msi)) {
    out_msi <- file.path(OUT, sprintf("%s_msi_predicted.csv", cohort))
    write.csv(msi, out_msi, row.names = FALSE)
    n_h <- sum(msi$msi_premsim == "MSI-H")
    cat(sprintf("    -> %s (%d MSI-H / %d total)\n", basename(out_msi), n_h, nrow(msi)))
  }
}
cat("\nDONE\n")

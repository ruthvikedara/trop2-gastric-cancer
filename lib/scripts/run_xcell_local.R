# Run xCell on an expression matrix.

suppressPackageStartupMessages(library(xCell))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript run_xcell_local.R <input.txt> <output.txt> [rna_seq=FALSE]")
}
input_path  <- args[1]
output_path <- args[2]
rna_seq     <- if (length(args) >= 3) as.logical(args[3]) else FALSE

cat(sprintf("Input:    %s\n", input_path))
cat(sprintf("Output:   %s\n", output_path))
cat(sprintf("RNA-seq:  %s\n", rna_seq))

# Read the expression matrix. Tabs only; preserve original column names (e.g.
# TCGA barcodes with hyphens).
expr <- read.table(input_path, header = TRUE, row.names = 1, sep = "\t",
                   check.names = FALSE, stringsAsFactors = FALSE)
expr <- as.matrix(expr)
mode(expr) <- "numeric"
cat(sprintf("Loaded %d genes x %d samples\n", nrow(expr), ncol(expr)))

# xCell expects gene symbols (HGNC) as rownames, unique. Fail loudly otherwise.
if (any(duplicated(rownames(expr)))) {
  stop("Duplicate gene symbols in input - collapse before running xCell.")
}
if (any(is.na(rownames(expr))) || any(rownames(expr) == "")) {
  stop("Empty/NA rownames in input.")
}

# rnaseq=TRUE is xCell's flag for RNA-seq calibration; FALSE for microarray.
result <- xCellAnalysis(expr, rnaseq = rna_seq)
cat(sprintf("Computed %d cell types x %d samples\n", nrow(result), ncol(result)))

dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
write.table(result, output_path, sep = "\t", quote = FALSE, col.names = NA)
cat(sprintf("Saved to %s\n", output_path))

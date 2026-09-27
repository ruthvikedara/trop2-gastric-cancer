#!/usr/bin/env Rscript

# TROP2 tumor vs normal.

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

root <- normalizePath(getwd())
out_dir <- file.path(root, "trop2", "plots", "phase1")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
stem <- file.path(out_dir, "F1A_TROP2_normal_vs_tumor_three_cohort")

read_gene_row <- function(path, gene = "TACSTD2") {
  con <- file(path, open = "r")
  on.exit(close(con))
  header <- strsplit(readLines(con, n = 1), "\t", fixed = TRUE)[[1]][-1]
  repeat {
    line <- readLines(con, n = 1)
    if (length(line) == 0) stop(gene, " not found in ", path)
    fields <- strsplit(line, "\t", fixed = TRUE)[[1]]
    if (fields[1] == gene) {
      values <- as.numeric(fields[-1])
      names(values) <- header
      return(values)
    }
  }
}

split_ids <- function(path, id_col, mode) {
  clinical <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  ids <- trimws(as.character(clinical[[id_col]]))
  if (mode == "GSE66229") {
    tissue <- tolower(trimws(as.character(clinical$tissue_type)))
    return(list(
      normal = ids[tissue == "normal"], tumor = ids[tissue == "tumor"],
      normal_pair = rep(NA_character_, sum(tissue == "normal")),
      tumor_pair = rep(NA_character_, sum(tissue == "tumor"))
    ))
  }
  title <- tolower(as.character(clinical$sample_title))
  normal_idx <- grepl("normal", title, fixed = TRUE)
  tumor_idx <- grepl("tumor", title, fixed = TRUE)
  pair_id <- sub(".*_rep", "", title)
  list(
    normal = ids[normal_idx], tumor = ids[tumor_idx],
    normal_pair = pair_id[normal_idx], tumor_pair = pair_id[tumor_idx]
  )
}

make_cohort <- function(name, expr, clinical, id_col, title) {
  values <- read_gene_row(file.path(root, expr))
  ids <- split_ids(file.path(root, clinical), id_col, name)
  data.frame(
    cohort = title,
    group = factor(
      c(rep("normal", length(ids$normal)), rep("tumor", length(ids$tumor))),
      levels = c("normal", "tumor")
    ),
    expression = c(values[ids$normal], values[ids$tumor]),
    pair_id = c(ids$normal_pair, ids$tumor_pair),
    stringsAsFactors = FALSE
  )
}

dat <- rbind(
  make_cohort(
    "GSE66229",
    "data/processed/gastric/ACRG_GSE66229_for_xcell.txt",
    "data/clinical/gastric/processed/GSE66229_clinical.csv",
    "gsm_id", "GSE66229\n(300T / 100N)"
  ),
  make_cohort(
    "GSE118916",
    "data/processed/gastric/ACRG_GSE118916_for_xcell.txt",
    "data/clinical/gastric/processed/GSE118916_clinical.csv",
    "gsm_id", "GSE118916\n(15T / 15N paired)"
  )
)

combo <- read.csv(
  file.path(root, "data/processed/gastric/TCGA_STAD_TROP2_normal_vs_tumor_TCGA_GTEx.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
tcga <- data.frame(
  cohort = sprintf(
    "TCGA + GTEx\n(%dT / %dN)",
    sum(tolower(combo$group) == "tumor"),
    sum(tolower(combo$group) == "normal")
  ),
  group = factor(tolower(combo$group), levels = c("normal", "tumor")),
  expression = combo$TACSTD2,
  pair_id = NA_character_
)
dat <- rbind(dat, tcga)
dat <- dat[is.finite(dat$expression), ]

cohort_order <- c(
  unique(dat$cohort[grepl("^GSE66229", dat$cohort)]),
  unique(dat$cohort[grepl("^TCGA", dat$cohort)]),
  unique(dat$cohort[grepl("^GSE118916", dat$cohort)])
)
normal_fill <- "#B0B0B0"
tumor_fill <- "#C9A227"

format_p <- function(p) {
  if (p == 0) return("p < 2.22e-16")
  if (p < 0.001) sprintf("p = %.2e", p) else sprintf("p = %.3f", p)
}

format_delta_p <- function(delta, p) {
  delta_str <- sprintf("Δ=%+.2f", delta)
  if (p == 0) return(paste0(delta_str, ", p < 2.22e-16"))
  p_str <- if (p < 0.001) sprintf("p = %.2e", p) else sprintf("p = %.3f", p)
  paste0(delta_str, ", ", p_str)
}

make_plot <- function(cohort_name, show_y = TRUE) {
  d <- dat[dat$cohort == cohort_name, ]
  n_normal <- sum(d$group == "normal")
  n_tumor <- sum(d$group == "tumor")
  paired <- all(!is.na(d$pair_id))
  if (paired) {
    wide <- reshape(
      d[, c("pair_id", "group", "expression")],
      idvar = "pair_id", timevar = "group", direction = "wide"
    )
    p <- wilcox.test(
      wide$expression.tumor, wide$expression.normal,
      paired = TRUE, exact = FALSE
    )$p.value
    delta <- median(wide$expression.tumor - wide$expression.normal)
  } else {
    p <- wilcox.test(
      d$expression[d$group == "tumor"],
      d$expression[d$group == "normal"],
      exact = FALSE
    )$p.value
    delta <- median(d$expression[d$group == "tumor"]) - median(d$expression[d$group == "normal"])
  }
  limits <- range(d$expression, na.rm = TRUE)
  span <- max(diff(limits), 0.5)
  bracket <- limits[2] + 0.085 * span
  tick <- 0.022 * span
  lower_pad <- if (cohort_name == "GSE118916") 0.16 else 0.06

  ggplot(d, aes(group, expression, fill = group)) +
    geom_violin(
      width = 0.68, trim = FALSE, scale = "width", adjust = 0.72,
      color = NA, alpha = 0.88
    ) +
    geom_boxplot(
      width = 0.12, fill = "white", color = "black", linewidth = 0.55,
      outlier.shape = 16, outlier.size = 1.25, outlier.alpha = 0.55
    ) +
    annotate(
      "segment", x = 1, xend = 1, y = bracket, yend = bracket + tick,
      linewidth = 0.6
    ) +
    annotate(
      "segment", x = 1, xend = 2, y = bracket + tick, yend = bracket + tick,
      linewidth = 0.6
    ) +
    annotate(
      "segment", x = 2, xend = 2, y = bracket + tick, yend = bracket,
      linewidth = 0.6
    ) +
    annotate(
      "text", x = 1.5, y = bracket + 2.4 * tick, label = format_delta_p(delta, p),
      size = 3.9, vjust = 0
    ) +
    scale_fill_manual(values = c(normal = normal_fill, tumor = tumor_fill)) +
    scale_x_discrete(labels = c(
      normal = sprintf("n=%d\nnormal", n_normal),
      tumor = sprintf("n=%d\ntumor", n_tumor)
    )) +
    scale_y_continuous(expand = expansion(mult = c(0, 0))) +
    coord_cartesian(
      ylim = c(limits[1] - lower_pad * span, limits[2] + 0.30 * span),
      clip = "off"
    ) +
    labs(
      title = cohort_name,
      x = NULL,
      y = if (show_y) expression(TROP2~expression~(log[2])) else NULL
    ) +
    theme_classic(base_size = 15) +
    theme(
      legend.position = "none",
      plot.title = element_text(size = 14, face = "bold", hjust = 0, lineheight = 0.95),
      axis.title.y = element_text(size = 13, margin = margin(r = 5)),
      axis.text.x = element_text(size = 12.5, color = "black", lineheight = 0.92),
      axis.text.y = element_text(size = 11.5, color = "black"),
      axis.ticks.x = element_blank(),
      panel.grid.major.y = element_line(
        color = "#B8B8B8", linewidth = 0.42, linetype = "dotted"
      ),
      plot.margin = margin(t = 2, r = 6, b = 2, l = if (show_y) 12 else 4)
    )
}

plots <- lapply(seq_along(cohort_order), function(i) {
  make_plot(cohort_order[i], show_y = TRUE)
})

draw_all <- function() {
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(
    6, 1,
    heights = unit(c(0.055, 1, 0.006, 1, 0.006, 1), c("npc", "null", "npc", "null", "npc", "null"))
  )))
  grid.text(
    "TROP2: tumor vs normal",
    vp = viewport(layout.pos.row = 1, layout.pos.col = 1),
    gp = gpar(fontsize = 15, fontface = "bold")
  )
  plot_rows <- c(2, 4, 6)
  for (i in seq_along(plots)) {
    print(
      plots[[i]],
      vp = viewport(layout.pos.row = plot_rows[i], layout.pos.col = 1),
      newpage = FALSE
    )
  }
}

cairo_pdf(paste0(stem, ".pdf"), width = 3.6, height = 7.82)
draw_all()
dev.off()
png(paste0(stem, ".png"), width = 3.6, height = 7.82, units = "in", res = 300, type = "cairo")
draw_all()
dev.off()

stats <- do.call(rbind, lapply(cohort_order, function(cohort_name) {
  d <- dat[dat$cohort == cohort_name, ]
  normal <- d$expression[d$group == "normal"]
  tumor <- d$expression[d$group == "tumor"]
  paired <- all(!is.na(d$pair_id))
  if (paired) {
    wide <- reshape(
      d[, c("pair_id", "group", "expression")],
      idvar = "pair_id", timevar = "group", direction = "wide"
    )
    delta <- median(wide$expression.tumor - wide$expression.normal)
    p <- wilcox.test(wide$expression.tumor, wide$expression.normal,
                     paired = TRUE, exact = FALSE)$p.value
  } else {
    delta <- median(tumor) - median(normal)
    p <- wilcox.test(tumor, normal, exact = FALSE)$p.value
  }
  data.frame(
    cohort = sub("\n.*$", "", cohort_name),
    normal_n = length(normal),
    tumor_n = length(tumor),
    median_delta_log2 = delta,
    test = if (paired) "paired Wilcoxon signed-rank" else "Mann-Whitney U",
    p = p
  )
}))
write.csv(stats, paste0(stem, "_stats.csv"), row.names = FALSE)
print(stats)

#!/usr/bin/env Rscript
# Figure 6 panel B — descendant tip-consistency forest (CH235 lineage).
# Faithful reuse of BEAST/18_plot_descendant_consistency_forest.R: the original
# forest code is inlined here unchanged and reads the descendant-consistency
# summary from source_data/. Panels A/C/D/E are Python (run Figure6.py).

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

ca <- commandArgs(FALSE)
.here <- sub("^--file=", "", ca[grep("^--file=", ca)])
.here <- if (length(.here) > 0) dirname(normalizePath(.here)) else getwd()
data_dir <- file.path(.here, "source_data")
out_dir  <- file.path(.here, "Figure6_panels")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

fig_xlsx   <- file.path(data_dir, "Figure6.xlsx")  # sheet B_forest
output_pdf <- file.path(out_dir, "panel_B_descendant_consistency.pdf")

ch235_order <- c(
  "UCA", "IA4", "IA3", "IA2", "IA1",
  "CH235", "CH236", "CH239", "CH240", "CH241"
)
ch235_color <- "#D55E00"

p_label <- function(p_value) {
  ifelse(
    is.na(p_value), "NA",
    ifelse(p_value < 0.001, "<0.001", sprintf("%.3f", p_value))
  )
}

fisher_z_ci <- function(rho, n) {
  if (is.na(rho) || is.na(n) || n <= 3) {
    return(c(NA_real_, NA_real_))
  }
  clipped <- min(max(rho, -0.999999), 0.999999)
  z <- atanh(clipped)
  se <- 1 / sqrt(n - 3)
  tanh(c(z - 1.96 * se, z + 1.96 * se))
}

prepare_summary <- function(df) {
  names(df)[names(df) == "Antibody"] <- "antibody"
  names(df)[names(df) == "spearman_descendant_fraction_vs_pred_prob"] <- "spearman_rho"
  names(df)[names(df) == "spearman_permutation_p_greater"] <- "permutation_p"
  names(df)[names(df) == "n_node_antibody_pairs"] <- "n_nodes"
  names(df)[names(df) == "mean_descendant_measured_tips"] <- "mean_descendant_tips"
  df$antibody <- sub("^CH235_", "", df$antibody)

  df <- df[df$antibody %in% ch235_order, , drop = FALSE]
  df <- df[order(match(df$antibody, ch235_order)), , drop = FALSE]
  df$antibody <- factor(df$antibody, levels = rev(ch235_order))

  ci <- t(vapply(
    seq_len(nrow(df)),
    function(i) fisher_z_ci(df$spearman_rho[i], df$n_nodes[i]),
    numeric(2)
  ))
  df$ci_lo <- ci[, 1]
  df$ci_hi <- ci[, 2]
  df$ci_lo_plot <- pmax(df$ci_lo, 0)
  df$ci_hi_plot <- pmin(df$ci_hi, 1)
  df$estimable <- !is.na(df$spearman_rho)
  df$significant <- !is.na(df$permutation_p) & df$permutation_p < 0.05
  df$p_label <- p_label(df$permutation_p)
  df$n_label <- as.character(as.integer(df$n_nodes))
  df$plot_shape <- ifelse(df$significant, "Pperm < 0.05", "Pperm >= 0.05 or NA")
  df
}

plot_forest <- function(df, out_pdf) {
  pdf_width <- 4.7
  pdf_height <- 3.8

  p <- ggplot(df, aes(y = antibody)) +
    annotate(
      "rect", xmin = -0.15, xmax = 0, ymin = -Inf, ymax = Inf,
      fill = "#F2F2F2", color = NA
    ) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.35, color = "#777777") +
    geom_errorbarh(
      data = df[df$estimable, , drop = FALSE],
      aes(xmin = ci_lo_plot, xmax = ci_hi_plot),
      height = 0,
      linewidth = 0.55,
      color = ch235_color,
      alpha = 0.65
    ) +
    geom_point(
      data = df[df$estimable, , drop = FALSE],
      aes(x = spearman_rho, shape = plot_shape),
      size = 2.4,
      stroke = 0.8,
      color = ch235_color,
      fill = ch235_color
    ) +
    geom_text(
      data = df[!df$estimable, , drop = FALSE],
      aes(x = 0.02, label = "NA"),
      hjust = 0,
      color = "#9A9A9A",
      fontface = "italic",
      size = 2.9
    ) +
    geom_text(aes(x = 1.10, label = n_label), size = 2.8, hjust = 0.5) +
    geom_text(aes(x = 1.25, label = p_label), size = 2.8, hjust = 0.5) +
    annotate("text", x = 1.10, y = length(ch235_order) + 0.72, label = "n",
             fontface = "bold", size = 3.0) +
    annotate("text", x = 1.25, y = length(ch235_order) + 0.72, label = "Pperm",
             fontface = "bold", size = 3.0) +
    scale_shape_manual(
      values = c("Pperm < 0.05" = 21, "Pperm >= 0.05 or NA" = 1),
      breaks = c("Pperm < 0.05", "Pperm >= 0.05 or NA"),
      labels = c("Pperm < 0.05", "Pperm >= 0.05 or NA")
    ) +
    scale_x_continuous(
      breaks = c(0, 0.25, 0.5, 0.75, 1.0),
      expand = c(0, 0)
    ) +
    scale_y_discrete(limits = rev(ch235_order)) +
    labs(
      title = "Descendant tip consistency by antibody",
      subtitle = "CH235 lineage only; horizontal bars: approximate 95% Fisher-z CI; NA = not estimable.",
      x = "Spearman rho: descendant measured sensitive fraction\nvs ancestral predicted sensitivity",
      y = NULL,
      shape = NULL
    ) +
    coord_cartesian(xlim = c(-0.15, 1.31), clip = "off") +
    theme_classic(base_size = 9) +
    theme(
      plot.title = element_text(face = "bold", size = 11, hjust = 0.5),
      plot.subtitle = element_text(size = 7.5, color = "#555555", hjust = 0.5),
      axis.text.y = element_text(size = 9, color = "black"),
      axis.text.x = element_text(size = 8.5, color = "black"),
      axis.title.x = element_text(size = 9),
      legend.position = "bottom",
      legend.text = element_text(size = 8),
      legend.key.width = unit(0.35, "cm"),
      panel.grid.major.x = element_line(color = "#DDDDDD", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      plot.margin = margin(6, 30, 6, 6)
    )

  ggsave(out_pdf, p, width = pdf_width, height = pdf_height, units = "in", device = cairo_pdf)
}

raw <- as.data.frame(readxl::read_excel(fig_xlsx, sheet = "B_forest"))
df <- prepare_summary(raw)
plot_forest(df, output_pdf)
cat("wrote:", output_pdf, "\n")

#!/usr/bin/env Rscript
# Figure S4 - Stratified performance of UniCAV on the CH505 single-donor external
# validation set. Plotting logic ported from the original original_analysis scripts
# (plot_ch505_ensemble_confusion_by_antibody.R, plot_ch505_predicted_sensitivity_
# score_distribution.R); panel B reuses the same confusion styling faceted by
# sampling week.
#   A per-antibody confusion matrices (CH103 + CH235 lineages)   -> predictions
#   B confusion matrices by sampling week (Week 0..100)          -> predictions
#   C observed vs predicted sensitive fraction along CH235       -> C_maturation
#     maturation (germline -> mature); Pearson r = 0.647
#   D predicted-score violins by true S/R (All / CH103 / CH235)  -> predictions
# Data: source_data/FigureS4.xlsx. Output -> FigureS4_panels/.

suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr); library(ggplot2); library(scales)
})

ca <- commandArgs(FALSE)
here <- sub("^--file=", "", ca[grep("^--file=", ca)])
here <- if (length(here) > 0) dirname(normalizePath(here)) else getwd()
fig_xlsx <- file.path(here, "source_data", "FigureS4.xlsx")
out_dir  <- file.path(here, "FigureS4_panels")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

base_theme <- theme_classic(base_size = 7, base_family = "Helvetica") +
  theme(axis.line = element_line(linewidth = 0.35, colour = "#1A1A1A"),
        axis.ticks = element_line(linewidth = 0.3, colour = "#1A1A1A"),
        axis.text = element_text(colour = "#1A1A1A"), axis.title = element_text(colour = "#1A1A1A"),
        plot.title = element_text(size = 8.5, face = "bold", hjust = 0),
        strip.background = element_blank(), strip.text = element_text(size = 7, face = "bold"))
theme_set(base_theme)
save_pdf <- function(p, name, w, h)
  ggsave(file.path(out_dir, name), p, width = w / 25.4, height = h / 25.4, device = cairo_pdf, bg = "white")

ic50_to_sensitive <- function(ic50) as.integer(ic50 < 50)
auc_rank <- function(y, s) {
  pos <- s[y == 1]; neg <- s[y == 0]
  if (length(pos) == 0 || length(neg) == 0) return(NA_real_)
  r <- rank(c(pos, neg)); (sum(r[seq_along(pos)]) - length(pos) * (length(pos) + 1) / 2) / (length(pos) * length(neg))
}

pred <- read_excel(fig_xlsx, sheet = "predictions") %>%
  mutate(y_true_num = ic50_to_sensitive(ic50),
         y_true_num = ifelse(is.na(y_true_num), as.integer(y_true), y_true_num),
         score = as.numeric(average_y_prob),
         pred_num = as.integer(average_y_pred_label),
         pred_num = ifelse(is.na(pred_num), as.integer(score >= 0.5), pred_num)) %>%
  filter(!is.na(ic50), !is.na(y_true_num), is.finite(score))

# ---- shared confusion-matrix builder + plotter --------------------------------
build_conf <- function(df, facet_col) {
  df <- df %>%
    mutate(true_label = factor(ifelse(y_true_num == 1, "True Sensitive", "True Resistant"),
                               levels = c("True Sensitive", "True Resistant")),
           pred_label = factor(ifelse(pred_num == 1, "Pred Sensitive", "Pred Resistant"),
                               levels = c("Pred Resistant", "Pred Sensitive")))
  df %>%
    count(.data[[facet_col]], true_label, pred_label, name = "count") %>%
    complete(!!rlang::sym(facet_col),
             true_label = factor(c("True Sensitive", "True Resistant"), levels = c("True Sensitive", "True Resistant")),
             pred_label = factor(c("Pred Resistant", "Pred Sensitive"), levels = c("Pred Resistant", "Pred Sensitive")),
             fill = list(count = 0)) %>%
    group_by(.data[[facet_col]]) %>%
    mutate(panel_n = sum(count), percent = 100 * count / panel_n,
           label = sprintf("%d\n%.1f%%", count, percent),
           text_colour = ifelse(percent >= 50, "white", "#0F172A")) %>%
    ungroup()
}
plot_conf <- function(counts, facet_col, ncol, title) {
  ggplot(counts, aes(x = pred_label, y = true_label, fill = percent)) +
    geom_tile(width = 1, height = 1) +
    geom_text(aes(label = label, colour = text_colour), size = 2.1, fontface = "bold", lineheight = 0.9) +
    facet_wrap(vars(.data[[facet_col]]), ncol = ncol) +
    scale_fill_gradient(low = "#DDEAF4", high = "#0D3B73", limits = c(0, 100)) +
    scale_colour_identity() +
    scale_x_discrete(labels = c("Pred Resistant" = "Pred\nResistant", "Pred Sensitive" = "Pred\nSensitive"), expand = c(0, 0)) +
    scale_y_discrete(labels = c("True Resistant" = "True\nResistant", "True Sensitive" = "True\nSensitive"), expand = c(0, 0)) +
    coord_fixed(expand = FALSE) + labs(title = title, x = NULL, y = NULL) +
    theme(legend.position = "none", axis.text = element_text(size = 5.6),
          strip.text = element_text(size = 6.4, face = "bold"), panel.spacing = grid::unit(1.5, "mm"))
}

# ---- A: per-antibody confusion ------------------------------------------------
ch103_order <- c("CH103", "CH104", "CH106", "CH31")
ch235_order <- c("UCA", "IA4", "IA3", "IA2", "IA1", "CH235", "CH236", "CH239", "CH240", "CH241", "CH31")
predA <- pred %>% mutate(antibody_panel = paste0(lineage, " / ", antibody))
panel_order <- c(paste("CH103", ch103_order, sep = " / "), paste("CH235", ch235_order, sep = " / "))
cA <- build_conf(predA, "antibody_panel")
panel_order <- panel_order[panel_order %in% cA$antibody_panel]
cA$antibody_panel <- factor(cA$antibody_panel, levels = panel_order)
save_pdf(plot_conf(cA, "antibody_panel", 5, "A  Per-antibody confusion (CH103 + CH235 lineages)"), "panelA_confusion_by_antibody.pdf", 178, 95)

# ---- B: confusion by sampling week --------------------------------------------
week_order <- c("Week 0", "Week 4", "Week 14", "Week 20", "Week 30", "Week 53", "Week 78", "Week 100")
cB <- build_conf(pred, "timepoint")
cB$timepoint <- factor(cB$timepoint, levels = week_order[week_order %in% cB$timepoint])
save_pdf(plot_conf(cB, "timepoint", 8, "B  Confusion by sampling week"), "panelB_confusion_by_week.pdf", 178, 42)

# ---- C: CH235 maturation observed vs predicted --------------------------------
mat <- read_excel(fig_xlsx, sheet = "C_maturation") %>% arrange(ab_idx)
mat$antibody <- factor(mat$antibody, levels = mat$antibody[order(mat$ab_idx)])
r_c <- suppressWarnings(cor(mat$Observed, mat$Predicted, method = "pearson"))

maturation_theme <- theme_classic(base_size = 7, base_family = "Helvetica") +
  theme(
    axis.line = element_line(linewidth = 0.35, colour = "#1A1A1A"),
    axis.ticks = element_line(linewidth = 0.3, colour = "#1A1A1A"),
    axis.text = element_text(colour = "#1A1A1A"),
    axis.title = element_text(colour = "#1A1A1A"),
    legend.title = element_blank(),
    legend.text = element_text(size = 6.1),
    legend.key.height = unit(3, "mm"),
    legend.key.width = unit(4, "mm"),
    plot.title = element_text(size = 8, face = "bold", hjust = 0),
    panel.grid.major.y = element_line(linewidth = 0.18, colour = "#E6E9EF"),
    panel.grid.minor = element_blank()
  )

pal_obs_pred <- c(Observed = "#2C3947", Predicted = "#C2A56D")
make_maturation_plot <- function(summary_df, pred_col, y_label, title_r, palette_values) {
  plot_df <- summary_df %>%
    select(antibody, ab_idx, Observed, all_of(pred_col)) %>%
    rename(Predicted_value = all_of(pred_col)) %>%
    pivot_longer(cols = c(Observed, Predicted_value),
                 names_to = "Series", values_to = "Value") %>%
    mutate(
      Series = ifelse(Series == "Predicted_value", pred_col, Series),
      Series = factor(Series, levels = c("Observed", pred_col))
    )

  ggplot(plot_df, aes(x = antibody, y = Value,
                      colour = Series, linetype = Series, shape = Series)) +
    geom_line(aes(group = Series), linewidth = 0.65) +
    geom_point(size = 1.9, stroke = 0.25) +
    scale_colour_manual(values = palette_values) +
    scale_linetype_manual(values = setNames(c("solid", "22"), c("Observed", pred_col))) +
    scale_shape_manual(values = setNames(c(16, 17), c("Observed", pred_col))) +
    scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.25),
      labels = percent_format(accuracy = 1)
    ) +
    labs(
      x = "CH235 lineage antibody (germline to mature)",
      y = y_label,
      title = paste0("CH235 maturation-axis sensitivity (r = ", sprintf("%.3f", title_r), ")")
    ) +
    maturation_theme +
    theme(
      legend.position = "top",
      legend.justification = "left",
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid.major.x = element_blank(),
      plot.margin = margin(4, 4, 4, 4)
    )
}

pC <- make_maturation_plot(
  mat,
  pred_col = "Predicted",
  y_label = "Sensitive fraction",
  title_r = r_c,
  palette_values = pal_obs_pred
)
save_pdf(pC, "panelC_ch235_maturation.pdf", 89, 70)

# ---- D: predicted-score violins by true S/R -----------------------------------
predD <- pred %>% mutate(true_label = factor(ifelse(y_true_num == 1, "True Sensitive", "True Resistant"),
                                             levels = c("True Resistant", "True Sensitive")),
                         lineage = factor(lineage, levels = c("CH103", "CH235")))
plot_df <- bind_rows(mutate(predD, panel = "All"), mutate(predD, panel = as.character(lineage))) %>%
  filter(!is.na(panel)) %>% mutate(panel = factor(panel, levels = c("All", "CH103", "CH235")))
summ <- plot_df %>% group_by(panel) %>%
  summarise(n_resistant = sum(y_true_num == 0), n_sensitive = sum(y_true_num == 1),
            accuracy = mean(y_true_num == pred_num, na.rm = TRUE), .groups = "drop")
n_anno <- bind_rows(
  transmute(summ, panel, true_label = factor("True Resistant", levels = c("True Resistant", "True Sensitive")),
            label = sprintf("n = %d", n_resistant), y = 0.075),
  transmute(summ, panel, true_label = factor("True Sensitive", levels = c("True Resistant", "True Sensitive")),
            label = sprintf("n = %d", n_sensitive), y = 0.075))
acc_anno <- transmute(summ, panel, label = sprintf("Acc = %.3f", accuracy), x = 1.5, y = 0.155)
pal_true <- c("True Resistant" = "#DB6C50", "True Sensitive" = "#2C9688")
set.seed(17)
pD <- ggplot(plot_df, aes(x = true_label, y = score, fill = true_label, colour = true_label)) +
  geom_violin(width = 0.82, linewidth = 0.25, alpha = 0.35, trim = TRUE, scale = "width") +
  geom_boxplot(width = 0.22, outlier.shape = NA, linewidth = 0.35, alpha = 0.85) +
  geom_jitter(width = 0.115, height = 0, size = 0.58, alpha = 0.28, stroke = 0) +
  geom_hline(yintercept = 0.5, linetype = "dashed", linewidth = 0.3, colour = "#777777") +
  geom_text(data = n_anno, aes(x = true_label, y = y, label = label), inherit.aes = FALSE,
            hjust = 0.5, vjust = 0, size = 2.45, fontface = "bold", colour = "#26313D", lineheight = 0.9) +
  geom_text(data = acc_anno, aes(x = x, y = y, label = label), inherit.aes = FALSE,
            hjust = 0.5, vjust = 0, size = 2.45, fontface = "bold", colour = "#26313D") +
  facet_wrap(~ panel, nrow = 1) +
  scale_fill_manual(values = pal_true, guide = "none") + scale_colour_manual(values = pal_true, guide = "none") +
  scale_x_discrete(labels = c("True Resistant" = "True\nResistant", "True Sensitive" = "True\nSensitive")) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25), expand = expansion(mult = c(0.02, 0.03))) +
  coord_cartesian(clip = "off") +
  labs(title = "D  CH505 predicted sensitivity score distribution", x = NULL, y = "Predicted sensitivity score") +
  theme(axis.text = element_text(size = 7), axis.title.y = element_text(size = 8),
        panel.grid.major.y = element_line(linewidth = 0.18, colour = "#E7EBF0"), plot.margin = margin(4, 15, 4, 5))
save_pdf(pD, "panelD_score_violin.pdf", 118, 68)

cat("Figure S4 panels ->", out_dir, "\n")
for (f in list.files(out_dir, pattern = "\\.pdf$")) cat("  ", f, "\n")

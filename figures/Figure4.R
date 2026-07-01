#!/usr/bin/env Rscript
# Figure 4 (R panels) — single consolidated script.
# Each panel is a function reusing the original CH505 plotting code; data is read
# from source_data/ and panel PDFs are written to Figure4_panels/.
# Panel D (regression) is Python: run Figure4.py.

suppressPackageStartupMessages({
  library(dplyr); library(ggplot2); library(readr); library(tidyr)
  library(ape); library(ggtree); library(ggnewscale); library(grid); library(scales); library(stringr); library(forcats)
})

ca <- commandArgs(FALSE)
.here <- sub("^--file=", "", ca[grep("^--file=", ca)])
.here <- if (length(.here) > 0) dirname(normalizePath(.here)) else getwd()
data_dir <- file.path(.here, "source_data")
out_dir  <- file.path(.here, "Figure4_panels")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# shared inputs used by the panel bodies
input_csv      <- file.path(data_dir, "independent_test_predictions_ensemble.csv")
gbm_csv        <- file.path(data_dir, "gbm_predictions.csv")
tree_file      <- file.path(data_dir, "ch505_hky_strict_exp.mcc.tree")
dates_file     <- file.path(data_dir, "ch505_beast_dates.tsv")
pred_file      <- input_csv
full_pred_file <- file.path(data_dir, "full_predictions.csv")
output_dir     <- out_dir
out_pdf            <- file.path(out_dir, "panel_G_tree.pdf")
out_annot          <- file.path(out_dir, ".tree_tip_annotations.csv")
out_tips           <- file.path(out_dir, ".tree_tips.csv")
out_source_summary <- file.path(out_dir, ".tree_source_summary.csv")

panel_B_confusion <- function() {

auc_rank <- function(label, score) {
  ok <- is.finite(label) & is.finite(score)
  label <- label[ok]
  score <- score[ok]
  if (length(unique(label)) < 2) return(NA_real_)
  n_pos <- sum(label == 1)
  n_neg <- sum(label == 0)
  ranks <- rank(score, ties.method = "average")
  (sum(ranks[label == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

ic50_to_sensitive <- function(x) {
  x_chr <- trimws(as.character(x))
  x_num <- suppressWarnings(as.numeric(gsub("[<>~=[:space:]]", "", x_chr)))
  ifelse(is.na(x_num), NA_integer_, ifelse(grepl("^\\s*>", x_chr) | x_num >= 50, 0L, 1L))
}

metric_row <- function(df, group_name) {
  y <- df$y_true_num
  p <- df$pred_num
  prob <- df$prob

  tp <- as.numeric(sum(y == 1 & p == 1, na.rm = TRUE))
  tn <- as.numeric(sum(y == 0 & p == 0, na.rm = TRUE))
  fp <- as.numeric(sum(y == 0 & p == 1, na.rm = TRUE))
  fn <- as.numeric(sum(y == 1 & p == 0, na.rm = TRUE))
  n <- tp + tn + fp + fn

  accuracy <- (tp + tn) / n
  sensitivity <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
  specificity <- ifelse(tn + fp > 0, tn / (tn + fp), NA_real_)
  balanced_acc <- mean(c(sensitivity, specificity), na.rm = TRUE)
  precision <- ifelse(tp + fp > 0, tp / (tp + fp), NA_real_)
  f1 <- ifelse(is.finite(precision + sensitivity) && precision + sensitivity > 0,
               2 * precision * sensitivity / (precision + sensitivity),
               NA_real_)
  mcc_denom <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  mcc <- ifelse(mcc_denom > 0, (tp * tn - fp * fn) / mcc_denom, NA_real_)

  tibble(
    group = group_name,
    n = n,
    true_resistant_pred_resistant = tn,
    true_resistant_pred_sensitive = fp,
    true_sensitive_pred_resistant = fn,
    true_sensitive_pred_sensitive = tp,
    AUC = auc_rank(y, prob),
    Accuracy = accuracy,
    Balanced_Accuracy = balanced_acc,
    Precision_sensitive = precision,
    Recall_sensitive = sensitivity,
    Specificity_resistant = specificity,
    F1_sensitive = f1,
    MCC = mcc
  )
}

theme_set(
  theme_classic(base_size = 7, base_family = "Helvetica") +
    theme(
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      axis.text = element_text(colour = "#1A1A1A"),
      axis.title = element_text(colour = "#1A1A1A"),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 7.5),
      legend.title = element_text(size = 6.5),
      legend.text = element_text(size = 6.2),
      plot.title = element_text(size = 8.5, face = "bold", hjust = 0),
      panel.spacing = unit(5, "mm")
    )
)

raw_pred <- read_csv(input_csv, show_col_types = FALSE)
if (!"binary_label_50" %in% names(raw_pred)) raw_pred$binary_label_50 <- NA_integer_

pred <- raw_pred %>%
  mutate(
    y_true_num = ic50_to_sensitive(ic50),
    y_true_num = ifelse(is.na(y_true_num), as.integer(y_true), y_true_num),
    prob = as.numeric(average_y_prob),
    pred_num = as.integer(average_y_pred_label),
    pred_num = ifelse(is.na(pred_num), as.integer(prob >= 0.5), pred_num),
    true_label = factor(ifelse(y_true_num == 1, "True Sensitive", "True Resistant"), levels = c("True Sensitive", "True Resistant")),
    pred_label = factor(ifelse(pred_num == 1, "Pred Sensitive", "Pred Resistant"), levels = c("Pred Resistant", "Pred Sensitive"))
  ) %>%
  filter(!is.na(ic50), !is.na(y_true_num), is.finite(prob))

plot_df <- bind_rows(
  pred %>% mutate(panel = "All"),
  pred %>% filter(lineage %in% c("CH103", "CH235")) %>% mutate(panel = lineage)
) %>%
  count(panel, true_label, pred_label, name = "count") %>%
  complete(
    panel,
    true_label = factor(c("True Sensitive", "True Resistant"), levels = c("True Sensitive", "True Resistant")),
    pred_label = factor(c("Pred Resistant", "Pred Sensitive"), levels = c("Pred Resistant", "Pred Sensitive")),
    fill = list(count = 0)
  ) %>%
  group_by(panel) %>%
  mutate(
    panel_n = sum(count),
    percent = 100 * count / panel_n,
    label = sprintf("%d\n%.1f%%", count, percent),
    text_colour = ifelse(percent >= 50, "white", "#0F172A")
  ) %>%
  ungroup() %>%
  mutate(panel = factor(panel, levels = c("All", "CH103", "CH235")))

metrics <- bind_rows(
  metric_row(pred, "All"),
  pred %>% filter(lineage == "CH103") %>% metric_row("CH103"),
  pred %>% filter(lineage == "CH235") %>% metric_row("CH235")
)

write_csv(plot_df, file.path(out_dir, "unicav_ch505_confusion_counts.csv"))
write_csv(metrics, file.path(out_dir, "unicav_ch505_confusion_metrics.csv"))

label_mismatch <- pred %>%
  mutate(binary_label_50_num = suppressWarnings(as.integer(as.numeric(binary_label_50)))) %>%
  summarise(
    n = n(),
    y_true_1 = sum(y_true_num == 1, na.rm = TRUE),
    y_true_0 = sum(y_true_num == 0, na.rm = TRUE),
    binary_label_50_1 = sum(binary_label_50_num == 1, na.rm = TRUE),
    binary_label_50_0 = sum(binary_label_50_num == 0, na.rm = TRUE),
    same_y_true_and_binary_label_50 = sum(y_true_num == binary_label_50_num, na.rm = TRUE),
    different_y_true_and_binary_label_50 = sum(y_true_num != binary_label_50_num, na.rm = TRUE)
  )
write_csv(label_mismatch, file.path(out_dir, "unicav_ch505_label_column_check.csv"))

p <- ggplot(plot_df, aes(x = pred_label, y = true_label, fill = percent)) +
  geom_tile(colour = NA, linewidth = 0, width = 1, height = 1) +
  geom_text(aes(label = label, colour = text_colour), size = 3.15, fontface = "bold", lineheight = 0.88) +
  facet_wrap(~ panel, nrow = 1) +
  scale_fill_gradient(
    low = "#DDEAF4",
    high = "#0D3B73",
    limits = c(0, 100)
  ) +
  scale_colour_identity() +
  scale_x_discrete(
    labels = c("Pred Resistant" = "Pred\nResistant", "Pred Sensitive" = "Pred\nSensitive"),
    expand = c(0, 0)
  ) +
  scale_y_discrete(
    labels = c("True Resistant" = "True\nResistant", "True Sensitive" = "True\nSensitive"),
    expand = c(0, 0)
  ) +
  coord_fixed(expand = FALSE) +
  labs(
    title = "UniCAV model predictions on CH505 validation set",
    x = NULL,
    y = NULL
  ) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 7.2),
    axis.text.y = element_text(size = 7.2),
    strip.text = element_text(face = "bold", size = 8.8),
    panel.border = element_blank(),
    plot.margin = margin(4, 6, 4, 4)
  )

ggsave(
  file.path(out_dir, "unicav_ch505_confusion_matrices.pdf"),
  p,
  width = 145 / 25.4,
  height = 54 / 25.4,
  device = cairo_pdf,
  bg = "white"
)

message("Saved: ", file.path(out_dir, "unicav_ch505_confusion_matrices.pdf"))
message("Saved: ", file.path(out_dir, "unicav_ch505_confusion_metrics.csv"))
message("Saved: ", file.path(out_dir, "unicav_ch505_confusion_counts.csv"))

}

panel_C_metrics <- function() {

auc_rank <- function(label, score) {
  ok <- is.finite(label) & is.finite(score)
  label <- label[ok]
  score <- score[ok]
  if (length(unique(label)) < 2) return(NA_real_)
  n_pos <- sum(label == 1)
  n_neg <- sum(label == 0)
  ranks <- rank(score, ties.method = "average")
  (sum(ranks[label == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

ic50_to_sensitive <- function(x) {
  x_chr <- trimws(as.character(x))
  x_num <- suppressWarnings(as.numeric(gsub("[<>~=[:space:]]", "", x_chr)))
  ifelse(is.na(x_num), NA_integer_, ifelse(grepl("^\\s*>", x_chr) | x_num >= 50, 0L, 1L))
}

metric_row <- function(df, fold_name) {
  y <- df$y_true_num
  p <- df$pred_num
  prob <- df$prob

  tp <- as.numeric(sum(y == 1 & p == 1, na.rm = TRUE))
  tn <- as.numeric(sum(y == 0 & p == 0, na.rm = TRUE))
  fp <- as.numeric(sum(y == 0 & p == 1, na.rm = TRUE))
  fn <- as.numeric(sum(y == 1 & p == 0, na.rm = TRUE))
  n <- tp + tn + fp + fn

  recall_s <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
  specificity_r <- ifelse(tn + fp > 0, tn / (tn + fp), NA_real_)
  precision_s <- ifelse(tp + fp > 0, tp / (tp + fp), NA_real_)
  f1_s <- ifelse(is.finite(precision_s + recall_s) && precision_s + recall_s > 0,
                 2 * precision_s * recall_s / (precision_s + recall_s),
                 NA_real_)
  mcc_denom <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  mcc <- ifelse(mcc_denom > 0, (tp * tn - fp * fn) / mcc_denom, NA_real_)

  tibble(
    fold = fold_name,
    n = n,
    true_resistant_pred_resistant = tn,
    true_resistant_pred_sensitive = fp,
    true_sensitive_pred_resistant = fn,
    true_sensitive_pred_sensitive = tp,
    AUC = auc_rank(y, prob),
    Accuracy = (tp + tn) / n,
    Balanced_Accuracy = mean(c(recall_s, specificity_r), na.rm = TRUE),
    Precision_sensitive = precision_s,
    Recall_sensitive = recall_s,
    Specificity_resistant = specificity_r,
    F1_sensitive = f1_s,
    MCC = mcc
  )
}

theme_set(
  theme_classic(base_size = 7, base_family = "Helvetica") +
    theme(
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      axis.text = element_text(colour = "#1A1A1A"),
      axis.title = element_text(colour = "#1A1A1A"),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 7),
      legend.title = element_text(size = 6.5),
      legend.text = element_text(size = 6.2),
      plot.title = element_text(size = 8.5, face = "bold", hjust = 0),
      panel.spacing = unit(4, "mm")
    )
)

pred <- read_csv(input_csv, show_col_types = FALSE) %>%
  mutate(
    y_true_num = ic50_to_sensitive(ic50),
    y_true_num = ifelse(is.na(y_true_num), as.integer(y_true), y_true_num)
  ) %>%
  filter(!is.na(ic50), !is.na(y_true_num))

fold_long <- bind_rows(lapply(1:5, function(i) {
  prob_col <- paste0("fold", i, "_prob")
  pred %>%
    transmute(
      virus,
      antibody,
      lineage,
      fold = paste0("Fold ", i),
      y_true_num,
      prob = as.numeric(.data[[prob_col]]),
      pred_num = as.integer(.data[[paste0("fold", i, "_label")]]),
      pred_num = ifelse(is.na(pred_num), as.integer(prob >= 0.5), pred_num)
    )
}), pred %>%
  transmute(
    virus,
    antibody,
    lineage,
    fold = "Ensemble",
    y_true_num,
    prob = as.numeric(average_y_prob),
    pred_num = as.integer(average_y_pred_label),
    pred_num = ifelse(is.na(pred_num), as.integer(prob >= 0.5), pred_num)
  )
) %>%
  mutate(
    fold = factor(fold, levels = c(paste0("Fold ", 1:5), "Ensemble")),
    panel_group = factor("All", levels = c("All", "CH103", "CH235")),
    true_label = factor(ifelse(y_true_num == 1, "True Sensitive", "True Resistant"), levels = c("True Sensitive", "True Resistant")),
    pred_label = factor(ifelse(pred_num == 1, "Pred Sensitive", "Pred Resistant"), levels = c("Pred Resistant", "Pred Sensitive"))
  )

metrics <- bind_rows(lapply(levels(fold_long$fold), function(f) {
  metric_row(filter(fold_long, fold == f), f)
}))

counts <- bind_rows(
  fold_long %>% mutate(panel_group = factor("All", levels = c("All", "CH103", "CH235"))),
  fold_long %>%
    filter(lineage %in% c("CH103", "CH235")) %>%
    mutate(panel_group = factor(lineage, levels = c("All", "CH103", "CH235")))
) %>%
  count(panel_group, fold, true_label, pred_label, name = "count") %>%
  complete(
    panel_group,
    fold,
    true_label = factor(c("True Sensitive", "True Resistant"), levels = c("True Sensitive", "True Resistant")),
    pred_label = factor(c("Pred Resistant", "Pred Sensitive"), levels = c("Pred Resistant", "Pred Sensitive")),
    fill = list(count = 0)
  ) %>%
  group_by(panel_group, fold) %>%
  mutate(
    panel_n = sum(count),
    percent = 100 * count / panel_n,
    label = sprintf("%d\n%.1f%%", count, percent),
    text_colour = ifelse(percent >= 50, "white", "#0F172A")
  ) %>%
  ungroup()

write_csv(metrics, file.path(out_dir, "unicav_ch505_5fold_metrics.csv"))
write_csv(counts, file.path(out_dir, "unicav_ch505_5fold_confusion_counts.csv"))

conf_plot <- ggplot(counts, aes(x = pred_label, y = true_label, fill = percent)) +
  geom_tile(colour = NA, linewidth = 0, width = 1, height = 1) +
  geom_text(aes(label = label, colour = text_colour), size = 1.9, fontface = "bold", lineheight = 0.9) +
  facet_grid(panel_group ~ fold) +
  scale_fill_gradient(
    low = "#DDEAF4",
    high = "#0D3B73",
    limits = c(0, 100)
  ) +
  scale_colour_identity() +
  scale_x_discrete(
    labels = c("Pred Resistant" = "Pred\nResistant", "Pred Sensitive" = "Pred\nSensitive"),
    expand = c(0, 0)
  ) +
  scale_y_discrete(
    labels = c("True Resistant" = "True\nResistant", "True Sensitive" = "True\nSensitive"),
    expand = c(0, 0)
  ) +
  coord_fixed(expand = FALSE) +
  labs(
    title = "UniCAV CH505 five-fold predictions",
    x = NULL,
    y = NULL
  ) +
  theme(
    legend.position = "none",
    panel.border = element_blank(),
    axis.text.x = element_text(size = 5.8),
    axis.text.y = element_text(size = 5.8),
    strip.text.x = element_text(size = 6.5),
    strip.text.y = element_text(size = 6.5, angle = 0),
    plot.margin = margin(4, 6, 4, 4)
  )

metric_long <- metrics %>%
  select(fold, AUC, Accuracy, Balanced_Accuracy, F1_sensitive, MCC) %>%
  pivot_longer(-fold, names_to = "metric", values_to = "value") %>%
  mutate(
    metric = recode(
      metric,
      Balanced_Accuracy = "Balanced acc.",
      F1_sensitive = "F1 score"
    ),
    metric = factor(metric, levels = c("AUC", "Accuracy", "Balanced acc.", "F1 score", "MCC"))
  )

ensemble_points <- metric_long %>%
  filter(fold == "Ensemble") %>%
  transmute(metric, value)

pal_metric <- c(
  AUC = "#9BB3E5",
  Accuracy = "#F2C94C",
  `Balanced acc.` = "#FF8A65",
  `F1 score` = "#66BFA3",
  MCC = "#547A95"
)

metric_plot <- ggplot(ensemble_points, aes(x = metric, y = value, fill = metric)) +
  geom_col(width = 0.82, colour = "white", linewidth = 0.18) +
  geom_text(
    aes(label = sprintf("%.3f", value)),
    vjust = -0.45,
    size = 2.15,
    colour = "#2C3947",
    fontface = "bold"
  ) +
  scale_fill_manual(values = pal_metric) +
  scale_y_continuous(limits = c(0, 1), breaks = c(0.2, 0.4, 0.6, 0.8), expand = expansion(mult = c(0, 0.02))) +
  labs(
    title = "UniCAV CH505 ensemble performance",
    x = NULL,
    y = "Metric value",
    fill = NULL
  ) +
  theme(
    axis.line = element_line(linewidth = 0.35, colour = "#1A1A1A"),
    axis.ticks = element_line(linewidth = 0.3, colour = "#1A1A1A"),
    axis.text.x = element_text(angle = 25, hjust = 1),
    legend.position = "none",
    panel.grid.major.y = element_line(linewidth = 0.18, colour = "#E7EBF0")
  )

write_csv(ensemble_points, file.path(out_dir, "unicav_ch505_ensemble_metric_summary.csv"))

ggsave(
  file.path(out_dir, "unicav_ch505_5fold_confusion_matrices.pdf"),
  conf_plot,
  width = 190 / 25.4,
  height = 118 / 25.4,
  device = cairo_pdf,
  bg = "white"
)

ggsave(
  file.path(out_dir, "unicav_ch505_5fold_metrics.pdf"),
  metric_plot,
  width = 72 / 25.4,
  height = 62 / 25.4,
  device = cairo_pdf,
  bg = "white"
)

message("Saved: ", file.path(out_dir, "unicav_ch505_5fold_confusion_matrices.pdf"))
message("Saved: ", file.path(out_dir, "unicav_ch505_5fold_metrics.pdf"))
message("Saved: ", file.path(out_dir, "unicav_ch505_5fold_metrics.csv"))

}

panel_E_heatmap <- function() {

auc_rank <- function(label, score) {
  ok <- is.finite(label) & is.finite(score)
  label <- label[ok]
  score <- score[ok]
  if (length(unique(label)) < 2) return(NA_real_)
  n_pos <- sum(label == 1)
  n_neg <- sum(label == 0)
  ranks <- rank(score, ties.method = "average")
  (sum(ranks[label == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

ic50_to_sensitive <- function(x) {
  x_chr <- trimws(as.character(x))
  x_num <- suppressWarnings(as.numeric(gsub("[<>~=[:space:]]", "", x_chr)))
  ifelse(is.na(x_num), NA_integer_, ifelse(grepl("^\\s*>", x_chr) | x_num >= 50, 0L, 1L))
}

f1_sensitive <- function(label, pred) {
  tp <- sum(label == 1 & pred == 1, na.rm = TRUE)
  fp <- sum(label == 0 & pred == 1, na.rm = TRUE)
  fn <- sum(label == 1 & pred == 0, na.rm = TRUE)
  precision <- ifelse(tp + fp > 0, tp / (tp + fp), NA_real_)
  recall <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
  ifelse(is.finite(precision + recall) && precision + recall > 0,
         2 * precision * recall / (precision + recall),
         NA_real_)
}

balanced_acc <- function(label, pred) {
  tp <- sum(label == 1 & pred == 1, na.rm = TRUE)
  tn <- sum(label == 0 & pred == 0, na.rm = TRUE)
  fp <- sum(label == 0 & pred == 1, na.rm = TRUE)
  fn <- sum(label == 1 & pred == 0, na.rm = TRUE)
  recall_s <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
  specificity_r <- ifelse(tn + fp > 0, tn / (tn + fp), NA_real_)
  mean(c(recall_s, specificity_r), na.rm = TRUE)
}

mcc_score <- function(label, pred) {
  tp <- sum(label == 1 & pred == 1, na.rm = TRUE)
  tn <- sum(label == 0 & pred == 0, na.rm = TRUE)
  fp <- sum(label == 0 & pred == 1, na.rm = TRUE)
  fn <- sum(label == 1 & pred == 0, na.rm = TRUE)
  denom <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  ifelse(denom > 0, (tp * tn - fp * fn) / denom, NA_real_)
}

fmt_num <- function(x) {
  ifelse(is.na(x), "NA", sprintf("%.2f", x))
}

theme_set(
  theme_classic(base_size = 7, base_family = "Helvetica") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "#1A1A1A"),
      axis.ticks = element_line(linewidth = 0.3, colour = "#1A1A1A"),
    axis.text = element_text(colour = "#1A1A1A"),
    axis.title = element_text(colour = "#1A1A1A"),
    legend.position = "right",
    legend.justification = "center",
    legend.title = element_text(size = 5.8),
    legend.text = element_text(size = 5.4),
    plot.title = element_text(size = 9, face = "bold", hjust = 0),
    panel.grid = element_blank()
    )
)

pred <- read_csv(input_csv, show_col_types = FALSE) %>%
  mutate(
    y_true_num = ic50_to_sensitive(ic50),
    y_true_num = ifelse(is.na(y_true_num), as.integer(y_true), y_true_num),
    prob = as.numeric(average_y_prob),
    pred_num = as.integer(average_y_pred_label),
    pred_num = ifelse(is.na(pred_num), as.integer(prob >= 0.5), pred_num),
    row_label = paste(lineage, antibody, sep = " | ")
  ) %>%
  filter(!is.na(ic50), !is.na(y_true_num), is.finite(prob))

per_ab <- pred %>%
  group_by(lineage, antibody, row_label) %>%
  summarise(
    n = n(),
    AUC = auc_rank(y_true_num, prob),
    Accuracy = mean(pred_num == y_true_num, na.rm = TRUE),
    `Balanced acc.` = balanced_acc(y_true_num, pred_num),
    `F1 score` = f1_sensitive(y_true_num, pred_num),
    MCC = mcc_score(y_true_num, pred_num),
    .groups = "drop"
  )

row_order <- per_ab %>%
  mutate(
    lineage_rank = match(lineage, c("CH103", "CH235")),
    antibody_rank = case_when(
      lineage == "CH103" ~ match(antibody, c("CH103", "CH104", "CH106", "CH31")),
      lineage == "CH235" ~ match(antibody, c("CH235", "CH236", "CH239", "CH240", "CH241", "CH31", "IA1", "IA2", "IA3", "IA4", "UCA")),
      TRUE ~ NA_integer_
    )
  ) %>%
  arrange(lineage_rank, antibody_rank, antibody) %>%
  pull(row_label)

heat_df <- per_ab %>%
  select(row_label, AUC, Accuracy, `Balanced acc.`, `F1 score`, MCC) %>%
  pivot_longer(cols = c(AUC, Accuracy, `Balanced acc.`, `F1 score`, MCC),
               names_to = "Metric", values_to = "Value") %>%
  mutate(
    row_label = factor(row_label, levels = rev(row_order)),
    Metric = factor(Metric, levels = c("AUC", "Accuracy", "Balanced acc.", "F1 score", "MCC")),
    fill_value = pmin(1, pmax(0, Value)),
    text_col = ifelse(!is.na(fill_value) & fill_value >= 0.72, "white", "#1A1A1A"),
    label = fmt_num(Value)
  )

write_csv(per_ab, file.path(out_dir, "unicav_ch505_per_antibody_performance_metrics.csv"))

p <- ggplot(heat_df, aes(x = Metric, y = row_label, fill = fill_value)) +
  geom_tile(colour = "white", linewidth = 0.35, width = 1, height = 1) +
  geom_text(aes(label = label, colour = text_col), size = 2.05, fontface = "bold") +
  scale_colour_identity() +
  scale_fill_gradientn(
    colours = c("#F4F0EA", "#D7E5E8", "#7FAEBC", "#2C3947"),
    limits = c(0, 1),
    na.value = "#EFEFEF",
    breaks = c(0, 0.5, 1),
    labels = c("0", "0.5", "1"),
    name = "Metric value"
  ) +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  coord_fixed(expand = FALSE) +
  labs(
    x = NULL,
    y = NULL,
    title = "Per-antibody validation performance"
  ) +
  theme(
    legend.position = "right",
    legend.direction = "vertical",
    legend.key.width = unit(2.2, "mm"),
    legend.key.height = unit(9, "mm"),
    axis.text.x = element_text(angle = 30, hjust = 1, size = 8.4),
    axis.text.y = element_text(size = 6.8),
    axis.ticks = element_blank(),
    axis.line = element_line(linewidth = 0.35, colour = "#1A1A1A"),
    plot.margin = margin(4, 5, 4, 4)
  )

ggsave(
  file.path(out_dir, "unicav_ch505_per_antibody_performance_heatmap.pdf"),
  p,
  width = 260 / 25.4,
  height = 96 / 25.4,
  device = cairo_pdf,
  bg = "white"
)

message("Saved: ", file.path(out_dir, "unicav_ch505_per_antibody_performance_heatmap.pdf"))
message("Saved: ", file.path(out_dir, "unicav_ch505_per_antibody_performance_metrics.csv"))

}

panel_F_longitudinal <- function() {

week_num <- function(x) as.numeric(gsub("[^0-9]", "", x))

ic50_to_sensitive <- function(x) {
  x_chr <- trimws(as.character(x))
  x_num <- suppressWarnings(as.numeric(gsub("[<>~=[:space:]]", "", x_chr)))
  ifelse(is.na(x_num), NA_integer_, ifelse(grepl("^\\s*>", x_chr) | x_num >= 50, 0L, 1L))
}

safe_cor <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]
  y <- y[ok]
  if (length(x) < 3 || sd(x) == 0 || sd(y) == 0) return(NA_real_)
  cor(x, y)
}

theme_set(
  theme_classic(base_size = 7, base_family = "Helvetica") +
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
      strip.background = element_blank(),
      strip.text = element_text(size = 7, face = "bold"),
      panel.grid.major.y = element_line(linewidth = 0.18, colour = "#E6E9EF"),
      panel.grid.minor = element_blank()
    )
)

raw_pred <- read_csv(input_csv, show_col_types = FALSE)
if (!"binary_label_50" %in% names(raw_pred)) raw_pred$binary_label_50 <- NA_integer_

pred <- raw_pred %>%
  mutate(
    virus = as.character(virus),
    antibody = as.character(antibody),
    lineage = factor(as.character(lineage), levels = c("CH103", "CH235")),
    week = week_num(timepoint),
    unicav_y_true = as.integer(y_true),
    unicav_binary_label_50 = as.integer(binary_label_50),
    prob_sensitive = as.numeric(average_y_prob),
    pred_sensitive = as.integer(average_y_pred_label),
    pred_sensitive = ifelse(is.na(pred_sensitive), as.integer(prob_sensitive >= 0.5), pred_sensitive)
  ) %>%
  mutate(
    observed_sensitive = ic50_to_sensitive(ic50),
    observed_sensitive = ifelse(is.na(observed_sensitive), as.integer(y_true), observed_sensitive)
  ) %>%
  filter(!is.na(ic50), !is.na(lineage), is.finite(week), is.finite(observed_sensitive), is.finite(prob_sensitive))

trend_df <- pred %>%
  group_by(lineage, week, timepoint) %>%
  summarise(
    n = n(),
    Observed = mean(observed_sensitive == 1, na.rm = TRUE),
    Predicted = mean(pred_sensitive == 1, na.rm = TRUE),
    `Mean P(sensitive)` = mean(prob_sensitive, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(lineage, week)

trend_metrics <- trend_df %>%
  group_by(lineage) %>%
  summarise(
    n_timepoints = n(),
    Pearson_r = safe_cor(Observed, Predicted),
    MAE = mean(abs(Observed - Predicted), na.rm = TRUE),
    observed_drop = first(Observed[order(week)]) - last(Observed[order(week)]),
    predicted_drop = first(Predicted[order(week)]) - last(Predicted[order(week)]),
    .groups = "drop"
  )

lineage_labels <- trend_metrics %>%
  mutate(
    lineage_label = ifelse(
      is.finite(Pearson_r),
      paste0(lineage, " (r = ", sprintf("%.3f", Pearson_r), ")"),
      paste0(lineage, " (r = NA)")
    )
  )
lineage_labels <- setNames(lineage_labels$lineage_label, lineage_labels$lineage)

trend_long <- trend_df %>%
  pivot_longer(cols = c(Observed, Predicted), names_to = "Series", values_to = "Sensitive_fraction") %>%
  mutate(Series = factor(Series, levels = c("Observed", "Predicted")))

pal_obs_pred <- c(Observed = "#2C3947", Predicted = "#C2A56D")

p <- ggplot(trend_long, aes(x = week, y = Sensitive_fraction, colour = Series, linetype = Series)) +
  geom_line(linewidth = 0.58) +
  geom_point(aes(shape = Series), size = 1.75, stroke = 0.25) +
  facet_wrap(~ lineage, ncol = 1, labeller = labeller(lineage = lineage_labels)) +
  scale_colour_manual(values = pal_obs_pred) +
  scale_linetype_manual(values = c(Observed = "solid", Predicted = "22")) +
  scale_shape_manual(values = c(Observed = 16, Predicted = 17)) +
  scale_x_continuous(breaks = sort(unique(trend_long$week))) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25),
    labels = percent_format(accuracy = 1)
  ) +
  labs(
    x = "Sampling week",
    y = "Sensitive fraction",
    title = "UniCAV longitudinal sensitivity dynamics"
  ) +
  theme(
    legend.position = "top",
    legend.justification = "left",
    panel.grid.major.x = element_blank(),
    plot.margin = margin(5, 5, 5, 5)
  )

write_csv(trend_df, file.path(out_dir, "unicav_figure4F_longitudinal_sensitive_fraction.csv"))
write_csv(trend_metrics, file.path(out_dir, "unicav_figure4F_longitudinal_sensitive_fraction_metrics.csv"))

ggsave(
  file.path(out_dir, "unicav_figure4F_longitudinal_sensitive_fraction.pdf"),
  p,
  width = 105 / 25.4,
  height = 88 / 25.4,
  device = cairo_pdf,
  bg = "white"
)

message("Saved: ", file.path(out_dir, "unicav_figure4F_longitudinal_sensitive_fraction.pdf"))
message("Saved: ", file.path(out_dir, "unicav_figure4F_longitudinal_sensitive_fraction.csv"))
message("Saved: ", file.path(out_dir, "unicav_figure4F_longitudinal_sensitive_fraction_metrics.csv"))

}

panel_F_ch235 <- function() {

ch235_order <- c("UCA", "IA4", "IA3", "IA2", "IA1",
                 "CH235", "CH236", "CH239", "CH240", "CH241")

week_num <- function(x) as.numeric(gsub("[^0-9]", "", x))

ic50_to_sensitive <- function(x) {
  x_chr <- trimws(as.character(x))
  x_num <- suppressWarnings(as.numeric(gsub("[<>~=[:space:]]", "", x_chr)))
  ifelse(is.na(x_num), NA_integer_, ifelse(grepl("^\\s*>", x_chr) | x_num >= 50, 0L, 1L))
}

safe_cor <- function(x, y, method = "pearson") {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]
  y <- y[ok]
  if (length(x) < 3 || sd(x) == 0 || sd(y) == 0) return(NA_real_)
  cor(x, y, method = method)
}

theme_set(
  theme_classic(base_size = 7, base_family = "Helvetica") +
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
)

pred <- read_csv(input_csv, show_col_types = FALSE) %>%
  mutate(
    virus = as.character(virus),
    antibody = as.character(antibody),
    lineage = as.character(lineage),
    timepoint = as.character(timepoint),
    week = week_num(timepoint),
    observed_sensitive = ic50_to_sensitive(ic50),
    observed_sensitive = ifelse(is.na(observed_sensitive), as.integer(y_true), observed_sensitive),
    prob_sensitive = as.numeric(average_y_prob),
    pred_sensitive = as.integer(average_y_pred_label),
    pred_sensitive = ifelse(is.na(pred_sensitive), as.integer(prob_sensitive >= 0.5), pred_sensitive),
    antibody = factor(antibody, levels = ch235_order),
    ab_idx = match(as.character(antibody), ch235_order)
  ) %>%
  filter(
    !is.na(ic50),
    lineage == "CH235",
    !is.na(antibody),
    is.finite(ab_idx),
    is.finite(observed_sensitive),
    is.finite(pred_sensitive)
  )

summary_df <- pred %>%
  group_by(antibody, ab_idx) %>%
  summarise(
    n = n(),
    n_viruses = n_distinct(virus),
    Observed = mean(observed_sensitive == 1, na.rm = TRUE),
    Predicted = mean(pred_sensitive == 1, na.rm = TRUE),
    `Predicted score` = mean(prob_sensitive, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(ab_idx)

metrics_df <- tibble(
  lineage = "CH235",
  n_antibodies = nrow(summary_df),
  n_viruses = n_distinct(pred$virus),
  Pearson_r_observed_vs_predicted = safe_cor(summary_df$Observed, summary_df$Predicted, "pearson"),
  Spearman_r_observed_vs_predicted = safe_cor(summary_df$Observed, summary_df$Predicted, "spearman"),
  MAE_observed_vs_predicted = mean(abs(summary_df$Observed - summary_df$Predicted), na.rm = TRUE),
  Pearson_r_observed_vs_predicted_score = safe_cor(summary_df$Observed, summary_df$`Predicted score`, "pearson"),
  Spearman_r_observed_vs_predicted_score = safe_cor(summary_df$Observed, summary_df$`Predicted score`, "spearman"),
  MAE_observed_vs_predicted_score = mean(abs(summary_df$Observed - summary_df$`Predicted score`), na.rm = TRUE)
)

pal_obs_pred <- c(Observed = "#2C3947", Predicted = "#C2A56D")
pal_obs_score <- c(Observed = "#2C3947", `Predicted score` = "#C2A56D")

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
    theme(
      legend.position = "top",
      legend.justification = "left",
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid.major.x = element_blank(),
      plot.margin = margin(4, 4, 4, 4)
    )
}

p <- make_maturation_plot(
  summary_df,
  pred_col = "Predicted",
  y_label = "Sensitive fraction",
  title_r = metrics_df$Pearson_r_observed_vs_predicted,
  palette_values = pal_obs_pred
)

p_score <- make_maturation_plot(
  summary_df,
  pred_col = "Predicted score",
  y_label = "Sensitive fraction / score",
  title_r = metrics_df$Pearson_r_observed_vs_predicted_score,
  palette_values = pal_obs_score
)

out_base <- file.path(out_dir, "unicav_ch235_maturation_sensitive_fraction")
write_csv(summary_df, paste0(out_base, ".csv"))
write_csv(metrics_df, paste0(out_base, "_metrics.csv"))

ggsave(
  paste0(out_base, ".pdf"),
  p,
  width = 89 / 25.4,
  height = 70 / 25.4,
  device = cairo_pdf,
  bg = "white"
)

ggsave(
  paste0(out_base, "_score.pdf"),
  p_score,
  width = 89 / 25.4,
  height = 70 / 25.4,
  device = cairo_pdf,
  bg = "white"
)

message("Saved: ", paste0(out_base, ".pdf"))
message("Saved: ", paste0(out_base, "_score.pdf"))
message("Saved: ", paste0(out_base, ".csv"))
message("Saved: ", paste0(out_base, "_metrics.csv"))

}

panel_G_tree <- function() {

tree <- read.nexus(tree_file)
if (inherits(tree, "multiPhylo")) tree <- tree[[1]]
tree <- ladderize(tree, right = TRUE)

old_opt <- options(ignore.negative.edge = TRUE)
on.exit(options(old_opt), add = TRUE)

dates <- read.delim(dates_file, stringsAsFactors = FALSE, check.names = FALSE)
pred <- read.csv(pred_file, stringsAsFactors = FALSE)

tip_to_short_virus <- function(label) {
  core <- sub("^703010505_", "", label)
  core <- sub("_KC[0-9]+$", "", core)
  if (core == "TF") return("T/F")

  parts <- strsplit(core, "_", fixed = TRUE)[[1]]
  if (length(parts) < 2) return(gsub("_", ".", core))

  week <- parts[1]
  clone <- paste(parts[-1], collapse = "_")
  m <- regexec("^([A-Za-z]?)([0-9]+)$", clone)
  reg <- regmatches(clone, m)[[1]]
  if (length(reg) == 0) return(gsub("_", ".", core))

  prefix <- reg[2]
  number <- as.character(as.integer(reg[3]))
  clone_norm <- if (nzchar(prefix)) paste0(toupper(prefix), number) else number
  paste0(week, ".", clone_norm)
}

extract_timepoint <- function(label) {
  if (grepl("_TF$", label)) return("TF")
  matched <- sub("^.*_(w[0-9]+).*$", "\\1", label)
  if (identical(matched, label)) return("unknown")
  matched
}

bin_ic50 <- function(x) {
  cut(
    x,
    breaks = c(-Inf, 0.2, 1, 2, 10, 20, 50, Inf),
    labels = c("<=0.2", "0.2-1", "1-2", "2-10", "10-20", "20-50", ">=50"),
    right = TRUE,
    ordered_result = TRUE
  )
}

parse_ic50_numeric <- function(x) {
  suppressWarnings(as.numeric(gsub("[^0-9.]", "", as.character(x))))
}

bin_ic50_with_censor <- function(value, is_greater_than) {
  binned <- as.character(bin_ic50(value))
  binned[is_greater_than & !is.na(value) & value >= 50] <- ">=50"
  binned
}

# Same tip metadata as the original MCC IC50 tree; only the IC50 columns are replaced.
tip_meta <- data.frame(
  taxon = tree$tip.label,
  stringsAsFactors = FALSE
)
tip_meta$short_virus <- vapply(tip_meta$taxon, tip_to_short_virus, character(1))
tip_meta <- merge(tip_meta, dates, by = "taxon", all.x = TRUE, sort = FALSE)
tip_meta$tree_timepoint <- vapply(tip_meta$taxon, extract_timepoint, character(1))
tip_meta <- tip_meta[match(tree$tip.label, tip_meta$taxon), ]
if (anyNA(tip_meta$time_years)) {
  missing_taxa <- tip_meta$taxon[is.na(tip_meta$time_years)]
  stop("Missing time_years for tips: ", paste(head(missing_taxa, 10), collapse = ", "))
}

full_pred <- read.csv(full_pred_file, stringsAsFactors = FALSE)
prediction_source_label <- function(df, fallback) {
  if ("prediction_source" %in% names(df)) {
    vals <- unique(na.omit(as.character(df$prediction_source)))
    if (length(vals) > 0) return(vals[1])
  }
  fallback
}
measured_source_label <- prediction_source_label(pred, "measured_ensemble")
full_source_label <- prediction_source_label(full_pred, "full_ensemble")

unicav_ch235 <- pred[
  pred$lineage == "CH235" & pred$antibody == "CH235" & !is.na(pred$ic50),
  c("virus", "ic50", "average_y_prob", "average_y_pred_label")
]
unicav_ch235$measured_ch235_ic50 <- parse_ic50_numeric(unicav_ch235$ic50)
unicav_ch235$measured_ch235_ic50_gt <- grepl("^\\s*>", as.character(unicav_ch235$ic50))

measured_ch235 <- merge(
  aggregate(measured_ch235_ic50 ~ virus, unicav_ch235, mean, na.rm = TRUE),
  aggregate(measured_ch235_ic50_gt ~ virus, unicav_ch235, any),
  by = "virus",
  all = TRUE
)
names(measured_ch235)[names(measured_ch235) == "virus"] <- "short_virus"
measured_ch235$measured_ch235_ic50_bin <- bin_ic50_with_censor(
  measured_ch235$measured_ch235_ic50,
  measured_ch235$measured_ch235_ic50_gt
)

measured_pred_ch235 <- merge(
  aggregate(average_y_prob ~ virus, unicav_ch235, mean, na.rm = TRUE),
  aggregate(average_y_pred_label ~ virus, unicav_ch235, mean, na.rm = TRUE),
  by = "virus",
  all = TRUE
)
names(measured_pred_ch235) <- c(
  "short_virus",
  "predicted_ch235_prob_sensitive",
  "predicted_ch235_label_mean"
)
measured_pred_ch235$predicted_ch235_source <- "measured_ensemble"
measured_pred_ch235$predicted_ch235_source <- measured_source_label

full_ch235 <- full_pred[
  full_pred$lineage == "CH235" & full_pred$antibody == "CH235",
  c("virus", "average_y_prob", "average_y_pred_label")
]
full_ch235$average_y_prob <- suppressWarnings(as.numeric(full_ch235$average_y_prob))
full_ch235$average_y_pred_label <- suppressWarnings(as.numeric(full_ch235$average_y_pred_label))
full_pred_ch235 <- merge(
  aggregate(average_y_prob ~ virus, full_ch235, mean, na.rm = TRUE),
  aggregate(average_y_pred_label ~ virus, full_ch235, mean, na.rm = TRUE),
  by = "virus",
  all = TRUE
)
names(full_pred_ch235) <- c(
  "short_virus",
  "predicted_ch235_prob_sensitive",
  "predicted_ch235_label_mean"
)
full_pred_ch235$predicted_ch235_source <- "full_ensemble"
full_pred_ch235$predicted_ch235_source <- full_source_label
full_pred_ch235 <- full_pred_ch235[
  !(full_pred_ch235$short_virus %in% measured_pred_ch235$short_virus),
]

pred_ch235 <- rbind(measured_pred_ch235, full_pred_ch235)
pred_ch235$predicted_ch235_label <- ifelse(
  pred_ch235$predicted_ch235_label_mean >= 0.5,
  "Sensitive",
  "Resistant"
)
pred_ch235$predicted_ch235_ic50_bin <- ifelse(
  pred_ch235$predicted_ch235_label_mean >= 0.5,
  "Pred <50",
  "Pred >=50"
)

tip_meta <- merge(tip_meta, measured_ch235, by = "short_virus", all.x = TRUE, sort = FALSE)
tip_meta <- tip_meta[match(tree$tip.label, tip_meta$taxon), ]
tip_meta <- merge(tip_meta, pred_ch235, by = "short_virus", all.x = TRUE, sort = FALSE)
tip_meta <- tip_meta[match(tree$tip.label, tip_meta$taxon), ]
tip_meta$has_ch235_measured <- !is.na(tip_meta$measured_ch235_ic50)
tip_meta$has_ch235_predicted <- !is.na(tip_meta$predicted_ch235_label_mean)

write.csv(tip_meta, out_annot, row.names = FALSE)
write.csv(tip_meta[tip_meta$has_ch235_measured | tip_meta$has_ch235_predicted, ], out_tips, row.names = FALSE)
source_summary <- data.frame(
  source = c(measured_source_label, full_source_label),
  source_file = c(pred_file, full_pred_file),
  used_for = c(
    "CH235 tips with measured IC50",
    "CH235 tips without measured IC50"
  ),
  n_tips = c(
    sum(tip_meta$short_virus %in% measured_pred_ch235$short_virus, na.rm = TRUE),
    sum(tip_meta$short_virus %in% full_pred_ch235$short_virus, na.rm = TRUE)
  ),
  stringsAsFactors = FALSE
)
write.csv(source_summary, out_source_summary, row.names = FALSE)

week_levels <- unique(tip_meta$tree_timepoint[grepl("^w[0-9]+$", tip_meta$tree_timepoint)])
week_levels <- week_levels[order(as.integer(sub("^w", "", week_levels)))]
timepoint_levels <- unique(c("TF", week_levels, setdiff(unique(tip_meta$tree_timepoint), c("TF", week_levels))))
tip_meta$tree_timepoint <- factor(tip_meta$tree_timepoint, levels = timepoint_levels)

n_groups <- nlevels(tip_meta$tree_timepoint)
preferred_palette <- c(
  "TF" = "#d73027",
  "w4" = "#FFF6F6",
  "w14" = "#F6FFDC",
  "w20" = "#ebe6ba",
  "w22" = "#e8c69f",
  "w30" = "#dc8274",
  "w53" = "#c36075",
  "w78" = "#9e4e7a",
  "w100" = "#7f3f7a",
  "w136" = "#483063",
  "w160" = "#1e1030"
)
fallback_palette <- c("#e41a1c", "#ff7f00", "#4daf4a", "#00a087", "#377eb8", "#984ea3")
palette_values <- setNames(rep(fallback_palette, length.out = n_groups), levels(tip_meta$tree_timepoint))
palette_values[names(preferred_palette)[names(preferred_palette) %in% levels(tip_meta$tree_timepoint)]] <-
  preferred_palette[names(preferred_palette) %in% levels(tip_meta$tree_timepoint)]

tip_depths <- node.depth.edgelength(tree)[seq_len(Ntip(tree))]
root_offset <- min(tip_depths[tip_meta$time_years == min(tip_meta$time_years)])
max_time_years <- max(tip_meta$time_years, na.rm = TRUE)
tick_years <- seq(0, ceiling(max_time_years * 2) / 2, by = 0.5)
tick_positions <- root_offset + tick_years
tick_weeks <- round(tick_years * 52)
tree_max_depth <- max(tip_depths)
axis_df <- data.frame(
  x = tick_positions,
  label = tick_weeks
)
axis_y <- -18
axis_tick_y0 <- -18
axis_tick_y1 <- -15.5
axis_label_y <- -24
axis_title_y <- -32

measured_bin_levels <- c(">=50", "20-50", "10-20", "2-10", "1-2", "0.2-1", "<=0.2", "NT")
measured_colors <- c(
  ">=50" = "grey70",
  "20-50" = "#fff1b8",
  "10-20" = "#ffd166",
  "2-10" = "#f8961e",
  "1-2" = "#f3722c",
  "0.2-1" = "#d73027",
  "<=0.2" = "#7f0000",
  "NT" = "white"
)
predicted_bin_levels <- c("Pred <50", "Pred >=50", "NT")
predicted_colors <- c(
  "Pred <50" = "#d73027",
  "Pred >=50" = "grey70",
  "NT" = "white"
)

p_base <- ggtree(tree, size = 0.25)
td <- p_base$data
tip_plot_df <- merge(
  td[td$isTip, c("label", "x", "y")],
  tip_meta,
  by.x = "label", by.y = "taxon", all.x = TRUE
)
tip_plot_df$measured_bin <- ifelse(
  is.na(tip_plot_df$measured_ch235_ic50_bin),
  "NT",
  tip_plot_df$measured_ch235_ic50_bin
)
tip_plot_df$predicted_bin <- ifelse(
  is.na(tip_plot_df$predicted_ch235_ic50_bin),
  "NT",
  tip_plot_df$predicted_ch235_ic50_bin
)
tip_plot_df$measured_bin <- factor(tip_plot_df$measured_bin, levels = measured_bin_levels)
tip_plot_df$predicted_bin <- factor(tip_plot_df$predicted_bin, levels = predicted_bin_levels)
legend_measured_df <- data.frame(
  x = -1e6,
  y = seq_along(measured_bin_levels),
  measured_bin = factor(measured_bin_levels, levels = measured_bin_levels)
)
legend_predicted_df <- data.frame(
  x = -1e6,
  y = seq_along(predicted_bin_levels),
  predicted_bin = factor(predicted_bin_levels, levels = predicted_bin_levels)
)

tf_tip_df <- tip_plot_df[tip_plot_df$tree_timepoint == "TF", ]
non_tf_tip_df <- tip_plot_df[tip_plot_df$tree_timepoint != "TF", ]

bar_w <- tree_max_depth * 0.05
bar_gap <- tree_max_depth * 0.015
bar_x1 <- tree_max_depth * 1.04
bar_x2 <- bar_x1 + bar_w + bar_gap

legend_x <- tree_max_depth * 1.20
legend_key_w <- tree_max_depth * 0.034
legend_text_x <- legend_x + legend_key_w + tree_max_depth * 0.016
legend_key_h <- 3.15
legend_step <- 3.6
legend_top <- max(tip_plot_df$y, na.rm = TRUE) - 8

measured_legend <- data.frame(
  label = c(">=50", "20-50", "10-20", "2-10", "1-2", "0.2-1", "<=0.2", "Not Test"),
  fill = unname(measured_colors[measured_bin_levels]),
  y = legend_top - 13 - (seq_along(measured_bin_levels) - 1) * legend_step
)
measured_legend$xmin <- legend_x
measured_legend$xmax <- legend_x + legend_key_w
measured_legend$ymin <- measured_legend$y - legend_key_h / 2
measured_legend$ymax <- measured_legend$y + legend_key_h / 2

predicted_legend <- data.frame(
  label = c("Resistant", "Sensitive"),
  fill = c(predicted_colors[["Pred >=50"]], predicted_colors[["Pred <50"]]),
  y = legend_top - 75 - c(0, legend_step)
)
predicted_legend$xmin <- legend_x
predicted_legend$xmax <- legend_x + legend_key_w
predicted_legend$ymin <- predicted_legend$y - legend_key_h / 2
predicted_legend$ymax <- predicted_legend$y + legend_key_h / 2

timepoint_label <- vapply(levels(tip_meta$tree_timepoint), function(tp) {
  if (tp == "TF") "T/F"
  else if (grepl("^w[0-9]+$", tp)) paste0("Week ", sub("^w", "", tp))
  else tp
}, character(1))
timepoint_legend <- data.frame(
  label = timepoint_label,
  color = unname(palette_values[levels(tip_meta$tree_timepoint)]),
  y = legend_top - 118 - (seq_along(timepoint_label) - 1) * 3.35
)
timepoint_legend$x <- legend_x + legend_key_w / 2

legend_title_df <- data.frame(
  label = c("Measured\nIC50(ug/ml)", "Model\npredicted", "Timepoint"),
  x = c(legend_x, legend_x, legend_x),
  y = c(legend_top, legend_top - 62, legend_top - 105)
)

common_theme <- theme_tree2() +
  theme(
    plot.title = element_blank(),
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.line.x = element_blank(),
    axis.ticks.x = element_blank(),
    legend.position = "none",
    legend.title = element_text(size = 3.8, lineheight = 0.78),
    legend.text = element_text(size = 3.45),
    legend.key.height = unit(0.075, "cm"),
    legend.key.width = unit(0.13, "cm"),
    legend.spacing.y = unit(0, "cm"),
    legend.margin = margin(0, 0, 0, 0),
    legend.box = "vertical",
    legend.box.margin = margin(0, 0, 0, 1),
    legend.box.spacing = unit(0.02, "cm"),
    plot.margin = margin(4, 6, 5, 5)
  )

ic50_plot <- p_base +
  geom_point(
    data = non_tf_tip_df,
    aes(x = x, y = y, color = tree_timepoint),
    size = 1.0, shape = 16, inherit.aes = FALSE
  ) +
  geom_point(
    data = tf_tip_df,
    aes(x = x, y = y, color = tree_timepoint),
    size = 1.0, shape = 16, inherit.aes = FALSE
  ) +
  scale_color_manual(
    values = palette_values,
    breaks = levels(tip_meta$tree_timepoint),
    labels = vapply(levels(tip_meta$tree_timepoint), function(tp) {
      if (tp == "TF") "Transmitted/Founder"
      else if (grepl("^w[0-9]+$", tp)) paste0("Week ", sub("^w", "", tp))
      else tp
    }, character(1)),
    drop = FALSE,
    guide = "none"
  ) +
  geom_tile(
    data = tip_plot_df,
    aes(x = bar_x1 + 0.05, y = y, fill = measured_bin),
    width = bar_w, height = 0.8,
    inherit.aes = FALSE, show.legend = FALSE
  ) +
  scale_fill_manual(
    values = measured_colors,
    breaks = measured_bin_levels,
    limits = measured_bin_levels,
    labels = c(">=50", "20-50", "10-20", "2-10", "1-2", "0.2-1", "<=0.2", "Not Test"),
    drop = FALSE,
    guide = "none"
  ) +
  new_scale_fill() +
  geom_tile(
    data = tip_plot_df,
    aes(x = bar_x2 + 0.05, y = y, fill = predicted_bin),
    width = bar_w, height = 0.8,
    inherit.aes = FALSE, show.legend = FALSE
  ) +
  scale_fill_manual(
    values = predicted_colors,
    breaks = predicted_bin_levels,
    limits = predicted_bin_levels,
    labels = c("Sensitive", "Resistant", "Not tested"),
    drop = FALSE,
    guide = "none"
  ) +
  geom_text(
    data = tf_tip_df,
    aes(x = x, y = y, label = "T/F"),
    hjust = -0.3, size = 2.2, inherit.aes = FALSE
  ) +
  annotate(
    "segment",
    x = min(axis_df$x), xend = max(axis_df$x),
    y = axis_y, yend = axis_y,
    linewidth = 0.3, color = "black"
  ) +
  geom_segment(
    data = axis_df,
    aes(x = x, xend = x, y = axis_tick_y0, yend = axis_tick_y1),
    linewidth = 0.3, color = "black", inherit.aes = FALSE
  ) +
  geom_text(
    data = axis_df,
    aes(x = x, y = axis_label_y, label = label),
    size = 2.4, color = "black", inherit.aes = FALSE
  ) +
  annotate(
    "text",
    x = mean(range(axis_df$x)), y = axis_title_y,
    label = "Sampling Time (Weeks)",
    size = 2.8, color = "black"
  ) +
  annotate("text", x = bar_x1 + 0.05, y = -5, label = "measured", size = 1.8, hjust = 0.5, angle = 90) +
  annotate("text", x = bar_x2 + 0.05, y = -5, label = "predicted", size = 1.8, hjust = 0.5, angle = 90) +
  geom_text(
    data = legend_title_df,
    aes(x = x, y = y, label = label),
    size = 1.82, hjust = 0, lineheight = 0.82, inherit.aes = FALSE
  ) +
  geom_rect(
    data = measured_legend,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    fill = measured_legend$fill, color = "grey55", linewidth = 0.08, inherit.aes = FALSE
  ) +
  geom_text(
    data = measured_legend,
    aes(x = legend_text_x, y = y, label = label),
    size = 1.62, hjust = 0, inherit.aes = FALSE
  ) +
  geom_rect(
    data = predicted_legend,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    fill = predicted_legend$fill, color = "grey55", linewidth = 0.08, inherit.aes = FALSE
  ) +
  geom_text(
    data = predicted_legend,
    aes(x = legend_text_x, y = y, label = label),
    size = 1.62, hjust = 0, inherit.aes = FALSE
  ) +
  geom_point(
    data = timepoint_legend,
    aes(x = x, y = y),
    color = timepoint_legend$color, size = 0.98, inherit.aes = FALSE
  ) +
  geom_text(
    data = timepoint_legend,
    aes(x = legend_text_x, y = y, label = label),
    size = 1.54, hjust = 0, inherit.aes = FALSE
  ) +
  scale_x_continuous(
    breaks = tick_positions,
    labels = tick_weeks,
    expand = expansion(mult = c(0.005, 0.015))
  ) +
  labs(x = NULL, y = NULL) +
  common_theme +
  coord_cartesian(xlim = c(0, tree_max_depth * 1.48), clip = "off")

ggsave(out_pdf, plot = ic50_plot, width = 4.6, height = 5.2, units = "in", device = cairo_pdf)

cat("Wrote:", out_pdf, "\n")
cat("Wrote:", out_annot, "\n")
cat("Wrote:", out_tips, "\n")
cat("Wrote:", out_source_summary, "\n")
cat("Tips:", Ntip(tree), "\n")
cat("Measured CH235 tips:", sum(tip_meta$has_ch235_measured, na.rm = TRUE), "\n")
cat("Predicted CH235 tips:", sum(tip_meta$has_ch235_predicted, na.rm = TRUE), "\n")
cat("Full no-IC50 prediction source:", full_source_label, "\n")

}

for (fn in list(panel_B_confusion, panel_C_metrics, panel_E_heatmap,
                panel_F_longitudinal, panel_G_tree)) {
  res <- try(fn(), silent = FALSE)
  if (inherits(res, "try-error")) message("[panel FAILED]")
}

# rename panel PDFs to panel_<letter>_<desc>.pdf and drop non-panel extras
ren <- c(
  "unicav_ch505_confusion_matrices.pdf"            = "panel_B_confusion.pdf",
  "unicav_ch505_5fold_metrics.pdf"                 = "panel_C_metrics.pdf",
  "unicav_ch505_per_antibody_performance_heatmap.pdf" = "panel_E_per_antibody_heatmap.pdf",
  "unicav_figure4F_longitudinal_sensitive_fraction.pdf" = "panel_F_longitudinal.pdf"
)
for (o in names(ren)) {
  op <- file.path(out_dir, o)
  if (file.exists(op)) file.rename(op, file.path(out_dir, ren[[o]]))
}
# keep only the Figure-4 panel PDFs (drop intermediate CSVs and non-panel plots)
keep <- c(unname(ren), "panel_G_tree.pdf", "panel_D_regression_ch505.pdf")
for (ff in list.files(out_dir, full.names = TRUE)) {
  if (!(basename(ff) %in% keep)) unlink(ff)
}
message("Figure 4 R panels -> ", out_dir)

#!/usr/bin/env Rscript
# Figure S1 - Construction and characterization of the CATNAP virus-antibody
# dataset. Panels A-G are reproduced from source_data/FigureS1.xlsx. The
# plotting code is ported verbatim from the original scripts under
# original_analysis/ (plot_clean_json_cleaning_statistics.R, CATNAP_matrix/
# plot_catnap_measurement_sparsity.R, plot_clean_json_dataset_distributions.R,
# plot_clean_json_sequence_lengths.R):
#   A data-cleaning waterfall      (sheet A_waterfall)
#   B neutralization matrix        (sheet B_matrix; CATNAP assay.txt source)
#   C pairs by epitope class       (sheet C_epitope)
#   D pairs by virus subtype/CRF   (sheet D_subtype)
#   E classification target        (sheet E_class)
#   F regression IC50 distribution (sheet F_ic50)
#   G Env/VH/VL sequence lengths   (sheet G_seqlen)
# Output -> FigureS1_panels/.

suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr); library(ggplot2); library(scales); library(patchwork)
})

ca <- commandArgs(FALSE)
here <- sub("^--file=", "", ca[grep("^--file=", ca)])
here <- if (length(here) > 0) dirname(normalizePath(here)) else getwd()
fig_xlsx <- file.path(here, "source_data", "FigureS1.xlsx")
out_dir  <- file.path(here, "FigureS1_panels")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# palettes (verbatim from originals)
pal_stage    <- c(Input = "#D8D3CC", Classification = "#8AB8D5", Regression = "#D7B89C", Final = "#2F5F73")
pal_dataset  <- c(Classification = "#547A95", Regression = "#C2A56D")
pal_target   <- c(Sensitive = "#547A95", Resistant = "#D8745A")
pal_single   <- "#547A95"
pal_regression <- "#C2A56D"
palette_seq  <- c(Env = "#c9777a", VH = "#7099bf", VL = "#7caf89")
pal_log_ic50 <- c("#081D58", "#173B8F", "#2166AC", "#4292C6", "#9ECAE1")

theme_set(
  theme_classic(base_size = 7, base_family = "Helvetica") +
    theme(axis.line = element_line(linewidth = 0.35, colour = "#1A1A1A"),
          axis.ticks = element_line(linewidth = 0.3, colour = "#1A1A1A"),
          axis.text = element_text(colour = "#1A1A1A"), axis.title = element_text(colour = "#1A1A1A"),
          legend.title = element_blank(), legend.text = element_text(size = 6),
          plot.title = element_text(size = 8.5, face = "bold", hjust = 0),
          panel.grid.major.y = element_line(linewidth = 0.18, colour = "#E6E9EF"),
          panel.grid.minor = element_blank()))

save_pdf <- function(p, name, w_mm, h_mm)
  ggsave(file.path(out_dir, name), p, width = w_mm / 25.4, height = h_mm / 25.4, device = cairo_pdf, bg = "white")

# ---------------------------------------------------------------- A: waterfall
A <- as.data.frame(read_excel(fig_xlsx, sheet = "A_waterfall")) %>%
  group_by(task) %>% arrange(stage_order, .by_group = TRUE) %>%
  mutate(stage_key = paste(task, stage_order, stage, sep = "__"),
         stage_display = paste0(stage_order, ". ", stage), row_label = comma(rows),
         removed_label = ifelse(is.na(removed_from_previous), "", paste0("-", comma(removed_from_previous)))) %>%
  ungroup()
stage_levels <- A %>% group_by(task) %>% arrange(desc(stage_order), .by_group = TRUE) %>% pull(stage_key)
stage_labels <- setNames(A$stage_display, A$stage_key)
pA <- ggplot(A, aes(x = rows, y = factor(stage_key, levels = stage_levels), fill = stage_type)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = row_label), hjust = -0.08, size = 2.05, colour = "#1A1A1A") +
  geom_text(data = filter(A, !is.na(removed_from_previous)), aes(label = removed_label),
            hjust = 1.08, size = 1.9, colour = "#1A1A1A", fontface = "bold") +
  facet_wrap(~task, nrow = 1, scales = "free_y") +
  scale_fill_manual(values = pal_stage) + scale_y_discrete(labels = stage_labels) +
  scale_x_continuous(limits = c(0, max(A$rows) * 1.18), labels = comma,
                     breaks = breaks_pretty(n = 4), expand = expansion(mult = c(0, 0))) +
  labs(x = "Records", y = NULL) +
  theme(legend.position = "none", panel.grid.major.y = element_blank(),
        panel.spacing.x = grid::unit(8, "mm"), strip.background = element_blank(),
        strip.text = element_text(size = 8.8, face = "bold", hjust = 0))
save_pdf(pA, "panelA_waterfall.pdf", 178, 70)

# --------------------------------------------------------------- B: matrix (assay.txt)
B <- as.data.frame(read_excel(fig_xlsx, sheet = "B_matrix"))
v_rank <- B %>% count(Virus, name = "n") %>% arrange(desc(n)) %>% mutate(virus_rank = row_number())
a_rank <- B %>% count(Antibody, name = "n") %>% arrange(desc(n)) %>% mutate(antibody_rank = row_number())
Bp <- B %>% left_join(v_rank[, c("Virus", "virus_rank")], by = "Virus") %>%
  left_join(a_rank[, c("Antibody", "antibody_rank")], by = "Antibody")
pB <- ggplot(Bp, aes(x = virus_rank, y = antibody_rank, fill = log_ic50)) +
  geom_raster() +
  scale_fill_gradientn(colours = pal_log_ic50, name = expression(log[10] * IC[50])) +
  scale_y_reverse(expand = c(0, 0)) + scale_x_continuous(expand = c(0, 0)) +
  labs(x = "Viruses (ranked by coverage)", y = "Antibodies (ranked by coverage)", title = "Neutralization matrix") +
  theme(panel.grid = element_blank(), panel.background = element_rect(fill = "#E9E9E9", colour = NA),
        legend.position = "right", legend.title = element_text(size = 6),
        legend.key.width = grid::unit(3, "mm"), legend.key.height = grid::unit(6, "mm"))
save_pdf(pB, "panelB_matrix.pdf", 92, 66)

# ==== original helper: lump to top_n categories + Other, add fraction ========
lump_top <- function(counts, top_n = 12) {
  top_categories <- counts %>% group_by(category) %>%
    summarise(total_records = sum(records), .groups = "drop") %>%
    arrange(desc(total_records)) %>% slice_head(n = top_n) %>% pull(category)
  counts %>%
    mutate(category = if_else(category %in% top_categories, category, "Other")) %>%
    group_by(dataset, category) %>% summarise(records = sum(records), .groups = "drop") %>%
    group_by(dataset) %>% mutate(fraction = records / sum(records)) %>% ungroup()
}

# ==== original: plot_grouped_bar (verbatim) ==================================
plot_grouped_bar <- function(counts, title) {
  category_levels <- counts %>% group_by(category) %>%
    summarise(total_records = sum(records), .groups = "drop") %>%
    arrange(total_records) %>% pull(category)
  counts <- counts %>%
    mutate(dataset = factor(dataset, levels = c("Classification", "Regression")),
           category = factor(category, levels = category_levels))
  ggplot(counts, aes(x = category, y = records, fill = dataset)) +
    geom_col(position = position_dodge(width = 0.84), width = 0.8, colour = NA) +
    geom_text(aes(label = comma(records), group = dataset), position = position_dodge(width = 0.84),
              hjust = -0.08, size = 1.7, colour = "#1A1A1A") +
    coord_flip(clip = "off") + scale_fill_manual(values = pal_dataset) +
    scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.22))) +
    labs(title = title, x = NULL, y = "Virus-antibody pairs") +
    theme(legend.position = "right", legend.text = element_text(size = 5.2),
          legend.key.size = unit(3.2, "mm"), legend.key.width = unit(3.2, "mm"),
          legend.spacing.y = unit(1, "mm"), panel.grid.major.y = element_blank())
}

# --------------------------------------------------------------- C: epitope
C <- lump_top(as.data.frame(read_excel(fig_xlsx, sheet = "C_epitope")), top_n = 12)
save_pdf(plot_grouped_bar(C, "Antibody epitope distribution"), "panelC_epitope.pdf", 92, 70)

# --------------------------------------------------------------- D: subtype
D <- lump_top(as.data.frame(read_excel(fig_xlsx, sheet = "D_subtype")), top_n = 12)
save_pdf(plot_grouped_bar(D, "Virus subtype distribution"), "panelD_subtype.pdf", 92, 70)

# --------------------------------------------------------------- E: class target (donut)
E <- as.data.frame(read_excel(fig_xlsx, sheet = "E_class")) %>%
  mutate(label = factor(label, levels = c("Resistant", "Sensitive")),
         fraction = records / sum(records),
         text = paste0(as.character(label), "\n", comma(records), "\n", percent(fraction, accuracy = 0.1)))
pE <- ggplot(E, aes(x = 1, y = records, fill = label)) +
  geom_col(width = 1, colour = "white", linewidth = 0.35) +
  geom_text(aes(label = text), position = position_stack(vjust = 0.5), size = 2.25,
            lineheight = 0.88, colour = "#1A1A1A") +
  coord_polar(theta = "y") + scale_fill_manual(values = pal_target) +
  labs(title = "Classification target distribution") +
  theme(axis.line = element_blank(), axis.ticks = element_blank(), axis.text = element_blank(),
        axis.title = element_blank(), panel.grid = element_blank(), legend.position = "none")
save_pdf(pE, "panelE_class_balance.pdf", 72, 68)

# --------------------------------------------------------------- F: regression target
F <- as.data.frame(read_excel(fig_xlsx, sheet = "F_ic50")) %>% filter(!is.na(log_ic50_exact))
pF <- ggplot(F, aes(x = log_ic50_exact)) +
  geom_histogram(bins = 50, fill = pal_regression, colour = "white", linewidth = 0.12, alpha = 0.92) +
  geom_vline(xintercept = log10(50), linetype = "22", linewidth = 0.35, colour = "#D8745A") +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.08))) +
  labs(title = "Regression target distribution", x = expression(log[10] ~ IC[50]), y = "Virus-antibody pairs")
save_pdf(pF, "panelF_ic50.pdf", 82, 68)

# --------------------------------------------------------------- G: sequence lengths
G_raw <- as.data.frame(read_excel(fig_xlsx, sheet = "G_seqlen")) %>%
  tidyr::uncount(records) %>%
  mutate(dataset = factor(dataset, levels = c("Classification", "Regression")),
         chain = factor(chain, levels = c("Env", "VH", "VL")))
plot_length_panel <- function(lengths, dataset_label, chain_label) {
  x_start <- c(Env = 650, VH = 80, VL = 90)[[chain_label]]
  x_step  <- c(Env = 50, VH = 10, VL = 10)[[chain_label]]
  panel_data <- lengths %>% filter(dataset == dataset_label, chain == chain_label)
  x_end <- ceiling(max(panel_data$length, na.rm = TRUE) / x_step) * x_step
  x_breaks <- seq(x_start, x_end, by = x_step)
  ggplot(panel_data, aes(x = length, fill = chain)) +
    geom_histogram(bins = 42, colour = "white", linewidth = 0.12, alpha = 0.92) +
    { if (chain_label == "VL") annotate("text", x = x_end, y = Inf, label = dataset_label, angle = 270,
                                        hjust = -0.08, vjust = 0.5, fontface = "bold", size = 2.7, colour = "#1A1A1A") } +
    scale_fill_manual(values = palette_seq) +
    scale_x_continuous(breaks = x_breaks, labels = label_number(accuracy = 1), expand = expansion(mult = c(0, 0.02))) +
    coord_cartesian(xlim = c(x_start, x_end), clip = "off") +
    scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.08))) +
    labs(title = chain_label, x = "Sequence length (aa)", y = "Number of records") +
    theme(panel.grid.major.x = element_blank(),
          panel.grid.major.y = element_line(linewidth = 0.18, colour = "#E6E9EF"),
          legend.position = "none",
          plot.margin = margin(2, if (chain_label == "VL") 24 else 2, 2, 2))
}
chains <- c("Env", "VH", "VL")
class_panels <- lapply(chains, function(ch) plot_length_panel(G_raw, "Classification", ch))
reg_panels   <- lapply(chains, function(ch) plot_length_panel(G_raw, "Regression", ch))
pG <- wrap_plots(class_panels, nrow = 1) / wrap_plots(reg_panels, nrow = 1) &
  theme(plot.margin = margin(1.5, 1.5, 1.5, 1.5))
save_pdf(pG, "panelG_seqlen.pdf", 178, 88)

cat("Figure S1 panels ->", out_dir, "\n")
for (f in list.files(out_dir, pattern = "\\.pdf$")) cat("  ", f, "\n")

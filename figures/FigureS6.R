#!/usr/bin/env Rscript
# Figure S6 (panels A-C) - Selection pressure across CH505 Env. Ported from
# BEAST/selection_pressure/02_plot_selection_pressure.R (FEL/MEME Manhattan +
# dN-dS/branches) and the CD4bs enrichment plot.
#   A FEL site-wise selection (Manhattan + dN-dS)        -> A_selection/env_regions
#   B MEME episodic selection (Manhattan + n branches)   -> A_selection/env_regions
#   C enrichment of selection at CD4bs contact sites     -> C_enrichment
# Panel D (CD4bs escape link) is Python (run FigureS6.py).
# Data: source_data/FigureS6.xlsx. Output -> FigureS6_panels/.

suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr); library(ggplot2); library(patchwork); library(ggrepel)
})

ca <- commandArgs(FALSE)
here <- sub("^--file=", "", ca[grep("^--file=", ca)])
here <- if (length(here) > 0) dirname(normalizePath(here)) else getwd()
fig_xlsx <- file.path(here, "source_data", "FigureS6.xlsx")
out_dir  <- file.path(here, "FigureS6_panels")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

axis_col <- "#1A1A1A"; grid_col <- "#E7EBF0"; text_col <- "#26313D"; threshold_col <- "#777777"
pal_selection <- c("Positive" = "#DB6C50", "Negative" = "#0D3B73", "Neutral" = "#ACB4BF")
pal_meme <- c("TRUE" = "#DB6C50", "FALSE" = "#ACB4BF")
sig_threshold <- -log10(0.1)
theme_set(theme_classic(base_size = 7, base_family = "Helvetica") +
  theme(axis.line = element_line(linewidth = 0.35, colour = "#1A1A1A"),
        axis.ticks = element_line(linewidth = 0.3, colour = "#1A1A1A"),
        plot.title = element_text(size = 8.8, face = "bold", hjust = 0), panel.spacing = unit(4, "mm")))

sel <- as.data.frame(read_excel(fig_xlsx, sheet = "A_selection")) %>%
  mutate(hxb2_aa_position = suppressWarnings(as.numeric(hxb2_position)),
         FEL_selection = tools::toTitleCase(tolower(as.character(FEL_selection))),
         significant = as.character(MEME_significant) %in% c("True", "TRUE", "1"),
         fel_log10p = -log10(pmax(FEL_pvalue, 1e-20)),
         meme_log10p = -log10(pmax(MEME_pvalue, 1e-20)),
         dnds = FEL_beta - FEL_alpha,
         n_branches = as.numeric(MEME_n_branches)) %>%
  filter(!is.na(hxb2_aa_position))

env <- as.data.frame(read_excel(fig_xlsx, sheet = "env_regions"))
env$start <- env$hxb2_start; env$end <- env$hxb2_end; env$mid <- (env$start + env$end) / 2
region_colors <- c("#D3E2F2", "#FFFFFF")[(seq_len(nrow(env)) - 1) %% 2 + 1]
hxb2_max <- max(sel$hxb2_aa_position, na.rm = TRUE)

cd4bs_hxb2 <- data.frame(hxb2_start = c(124, 196, 198, 279, 364, 425, 455, 469, 471),
                         hxb2_end   = c(127, 196, 198, 283, 374, 432, 461, 469, 477))
add_cd4bs_bands <- function()
  geom_rect(data = cd4bs_hxb2, aes(xmin = hxb2_start - 0.5, xmax = hxb2_end + 0.5, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "#F4D06F", alpha = 0.32)
region_layer <- function()
  list(annotate("rect", xmin = env$start - 0.5, xmax = env$end + 0.5, ymin = -Inf, ymax = Inf, fill = region_colors, alpha = 0.7),
       annotate("text", x = env$mid, y = Inf, label = env$region, vjust = 1.7, size = 1.9, fontface = "bold", colour = text_col))
xscale <- scale_x_continuous(breaks = seq(0, hxb2_max, by = 100), expand = expansion(mult = c(0.01, 0.01)))

fel_pos <- sel %>% filter(FEL_selection == "Positive")
meme_sig <- sel %>% filter(significant)

# ---- A: FEL Manhattan + dN-dS ----
p_fel <- ggplot(sel, aes(hxb2_aa_position, fel_log10p)) + region_layer() + add_cd4bs_bands() +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = threshold_col, linewidth = 0.3) +
  geom_point(aes(color = FEL_selection), size = 0.72, alpha = 0.82) +
  geom_text_repel(data = fel_pos, aes(label = hxb2_aa_position), size = 2.0, color = "#8B5E00",
                  fontface = "bold", max.overlaps = Inf, nudge_y = 0.35, segment.size = 0.16,
                  segment.color = "#C39125", min.segment.length = 0) +
  scale_color_manual(values = pal_selection, name = "Selection") + xscale +
  labs(title = "FEL site-wise selection across CH505 Env", x = "HXB2 Env residue position",
       y = expression(-log[10](p - value))) +
  guides(color = guide_legend(override.aes = list(size = 1.8, alpha = 1), nrow = 1)) +
  theme(panel.grid.major.y = element_line(linewidth = 0.18, colour = grid_col), legend.position = "top",
        legend.justification = "left", legend.margin = margin(b = -5))
cap <- quantile(abs(sel$dnds), 0.99, na.rm = TRUE) * 1.5
sel$dnds_capped <- pmax(pmin(sel$dnds, cap), -cap)
p_dnds <- ggplot(sel, aes(hxb2_aa_position, dnds_capped)) + region_layer() + add_cd4bs_bands() +
  geom_hline(yintercept = 0, color = threshold_col, linewidth = 0.3) +
  geom_segment(aes(xend = hxb2_aa_position, y = 0, yend = dnds_capped, color = FEL_selection), linewidth = 0.25, alpha = 0.8) +
  geom_point(aes(color = FEL_selection), size = 0.52, alpha = 0.85) +
  scale_color_manual(values = pal_selection, guide = "none") + xscale +
  labs(title = "FEL rate difference per site", x = "HXB2 Env residue position", y = expression(dN - dS)) +
  theme(panel.grid.major.y = element_line(linewidth = 0.18, colour = grid_col))
ggsave(file.path(out_dir, "panelA_FEL.pdf"), p_fel / p_dnds + plot_layout(heights = c(1, 1)),
       width = 6.6, height = 4.35, units = "in", device = cairo_pdf, bg = "white")

# ---- B: MEME Manhattan + n branches ----
p_meme <- ggplot(sel, aes(hxb2_aa_position, meme_log10p)) + region_layer() + add_cd4bs_bands() +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = threshold_col, linewidth = 0.3) +
  geom_point(aes(color = significant), size = 0.72, alpha = 0.82) +
  scale_color_manual(values = pal_meme, labels = c("TRUE" = "p < 0.1", "FALSE" = "p >= 0.1"), name = "Episodic Selection") +
  geom_text_repel(data = meme_sig, aes(label = hxb2_aa_position), size = 2.0, color = "#DB6C50",
                  max.overlaps = 20, nudge_y = 0.5, segment.size = 0.16, segment.color = "#A8B7C8", min.segment.length = 0.1) +
  xscale + labs(title = "MEME episodic selection across CH505 Env", x = "HXB2 Env residue position",
                y = expression(-log[10](p - value))) +
  guides(color = guide_legend(override.aes = list(size = 1.8, alpha = 1), nrow = 1)) +
  theme(panel.grid.major.y = element_line(linewidth = 0.18, colour = grid_col), legend.position = "top",
        legend.justification = "left", legend.margin = margin(b = -5))
p_branches <- ggplot(meme_sig, aes(hxb2_aa_position, n_branches)) + region_layer() + add_cd4bs_bands() +
  geom_segment(aes(xend = hxb2_aa_position, y = 0, yend = n_branches), color = "#DB6C50", linewidth = 0.42, alpha = 0.85) +
  geom_point(color = "#DB6C50", size = 0.95) +
  geom_text_repel(data = meme_sig, aes(label = hxb2_aa_position), size = 2.0, color = "#DB6C50",
                  max.overlaps = 20, segment.size = 0.16, segment.color = "#A8B7C8", min.segment.length = 0.1) +
  xscale + labs(title = "MEME branches under selection", x = "HXB2 Env residue position", y = "# branches") +
  theme(panel.grid.major.y = element_line(linewidth = 0.18, colour = grid_col))
ggsave(file.path(out_dir, "panelB_MEME.pdf"), p_meme / p_branches + plot_layout(heights = c(1, 1)),
       width = 6.6, height = 4.35, units = "in", device = cairo_pdf, bg = "white")

# ---- C: CD4bs enrichment ----
enr <- as.data.frame(read_excel(fig_xlsx, sheet = "C_enrichment"))
signal_defs <- data.frame(metric = as.character(enr$metric), stringsAsFactors = FALSE)
enrichment_summary <- enr
enrichment_summary$metric <- factor(enrichment_summary$metric, levels = signal_defs$metric)

enrichment_plot_df <- rbind(
  data.frame(
    metric = enrichment_summary$metric,
    region = "CD4bs contact sites",
    selected = enrichment_summary$cd4bs_selected,
    total = enrichment_summary$cd4bs_total,
    percent = enrichment_summary$cd4bs_percent,
    stringsAsFactors = FALSE
  ),
  data.frame(
    metric = enrichment_summary$metric,
    region = "Other HXB2 sites",
    selected = enrichment_summary$non_cd4bs_selected,
    total = enrichment_summary$non_cd4bs_total,
    percent = enrichment_summary$non_cd4bs_percent,
    stringsAsFactors = FALSE
  )
)
enrichment_plot_df$metric <- factor(enrichment_plot_df$metric, levels = signal_defs$metric)
enrichment_plot_df$region <- factor(
  enrichment_plot_df$region,
  levels = c("Other HXB2 sites", "CD4bs contact sites")
)
enrichment_plot_df$count_label <- paste0(enrichment_plot_df$selected, "/", enrichment_plot_df$total)

region_cols <- c("Other HXB2 sites" = "#D4E5F4", "CD4bs contact sites" = "#0A6DAF")

enrichment_plot_df$metric <- factor(enrichment_plot_df$metric, levels = signal_defs$metric)
enrichment_plot_df$region <- factor(
  enrichment_plot_df$region,
  levels = c("Other HXB2 sites", "CD4bs contact sites")
)
enrichment_plot_df$metric_index <- as.numeric(enrichment_plot_df$metric)
enrichment_plot_df$x_pos <- enrichment_plot_df$metric_index +
  ifelse(enrichment_plot_df$region == "Other HXB2 sites", -0.135, 0.135)

bracket_df <- enrichment_summary
bracket_df$metric <- factor(bracket_df$metric, levels = signal_defs$metric)
bracket_df$metric_index <- as.numeric(bracket_df$metric)
bracket_df$x_left <- bracket_df$metric_index - 0.135
bracket_df$x_right <- bracket_df$metric_index + 0.135
bracket_df$y <- pmax(bracket_df$cd4bs_percent, bracket_df$non_cd4bs_percent) + 4.0
bracket_df$y_tick <- bracket_df$y - 1.2
bracket_df$y_label <- bracket_df$y + 1.0

pC <- ggplot(enrichment_plot_df, aes(x = x_pos, y = percent, fill = region)) +
  geom_col(width = 0.27, color = "white", linewidth = 0.25) +
  geom_segment(
    data = bracket_df,
    aes(x = x_left, xend = x_right, y = y, yend = y),
    inherit.aes = FALSE,
    linewidth = 0.32,
    color = axis_col
  ) +
  geom_segment(
    data = bracket_df,
    aes(x = x_left, xend = x_left, y = y_tick, yend = y),
    inherit.aes = FALSE,
    linewidth = 0.32,
    color = axis_col
  ) +
  geom_segment(
    data = bracket_df,
    aes(x = x_right, xend = x_right, y = y_tick, yend = y),
    inherit.aes = FALSE,
    linewidth = 0.32,
    color = axis_col
  ) +
  geom_text(
    data = bracket_df,
    aes(x = metric_index, y = y_label, label = p_label),
    inherit.aes = FALSE,
    size = 2.2,
    color = text_col
  ) +
  scale_fill_manual(values = region_cols, name = NULL) +
  scale_x_continuous(
    breaks = seq_along(signal_defs$metric),
    labels = signal_defs$metric,
    expand = expansion(add = 0.25)
  ) +
  scale_y_continuous(
    limits = c(0, max(bracket_df$y_label, enrichment_plot_df$percent) + 3),
    breaks = seq(0, 40, by = 10),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "Selection signals are enriched at CD4bs contact sites",
    x = NULL,
    y = "Selected HXB2-mapped sites (%)"
  ) +
  theme(
    axis.text.x = element_text(angle = 32, hjust = 1, vjust = 1, color = axis_col, size = 6.3),
    axis.text.y = element_text(color = axis_col, size = 7),
    axis.title.y = element_text(color = axis_col, size = 8),
    panel.grid.major.y = element_line(linewidth = 0.18, colour = grid_col),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "center",
    legend.text = element_text(size = 6.7),
    legend.key.size = unit(0.28, "cm"),
    legend.margin = margin(b = -3),
    plot.title = element_text(face = "bold", size = 8.8, hjust = 0.5, margin = margin(b = 1)),
    plot.margin = margin(3, 5, 3, 4)
  )
ggsave(file.path(out_dir, "panelC_cd4bs_enrichment.pdf"), plot = pC, width = 3.85, height = 2.55,
       units = "in", device = cairo_pdf)

cat("Figure S6 panels A-C ->", out_dir, "  [panel D: run FigureS6.py]\n")
for (f in list.files(out_dir, pattern = "\\.pdf$")) cat("  ", f, "\n")

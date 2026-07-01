#!/usr/bin/env Rscript
# Figure 5 (R panels D/E) — single consolidated script.
# Attention-vs-selection: violin of attention at positively-selected sites +
# attention~selection scatter. Reuses the original plotting code; reads
# source_data/ and writes the panel PDF to Figure5_panels/.
# Python panels A/B/C: run Figure5.py.

suppressPackageStartupMessages({
  library(dplyr); library(ggplot2); library(readr); library(tidyr)
  library(patchwork); library(scales); library(ggrepel)
})

ca <- commandArgs(FALSE)
.here <- sub("^--file=", "", ca[grep("^--file=", ca)])
.here <- if (length(.here) > 0) dirname(normalizePath(.here)) else getwd()
data_dir <- file.path(.here, "source_data")
out_dir  <- file.path(.here, "Figure5_panels")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
fig_xlsx <- file.path(data_dir, "Figure5.xlsx")  # sheets profile / selection / env_regions
out_pdf  <- file.path(out_dir, "panel_DE_attention_vs_selection.pdf")

panel_DE <- function() {

# --- Style (consistent with selection-pressure figures) ---
theme_set(
  theme_classic(base_size = 7, base_family = "Helvetica") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "#1A1A1A"),
      axis.ticks = element_line(linewidth = 0.3, colour = "#1A1A1A"),
      axis.text = element_text(colour = "#1A1A1A"),
      axis.title = element_text(colour = "#1A1A1A"),
      plot.title = element_text(size = 8, face = "bold", hjust = 0),
      plot.tag = element_text(size = 10, face = "bold"),
      legend.title = element_text(size = 6.3),
      legend.text = element_text(size = 6.0),
      legend.key.size = unit(0.28, "cm"),
      panel.grid.major.y = element_line(linewidth = 0.18, colour = "#E7EBF0"),
      panel.grid.minor = element_blank()
    )
)
col_attn <- "#0A6DAF"   # attention
col_pos  <- "#DB6C50"   # positive/episodic selection
col_grey <- "#ACB4BF"   # non-selected
loop_fill <- "#D3E2F2"  # gp120 variable-loop shading

# --- Load data ---
attn <- as.data.frame(readxl::read_excel(fig_xlsx, sheet = "profile"))
sel  <- as.data.frame(readxl::read_excel(fig_xlsx, sheet = "selection"))

attn$hxb2_pos <- suppressWarnings(as.numeric(attn$hxb2_pos))
sel$hxb2_position <- suppressWarnings(as.numeric(sel$hxb2_position))

# Per-HXB2-position selection status (positive OR episodic) + region
sel_pos_flag <- (tolower(as.character(sel$FEL_selection)) == "positive") |
  (tolower(as.character(sel$MEME_significant)) %in% c("true", "1"))
sel_site <- sel[!is.na(sel$hxb2_position), ]
sel_site$is_pos <- sel_pos_flag[!is.na(sel$hxb2_position)]
sel_site <- sel_site %>%
  group_by(hxb2_position) %>%
  summarise(region = dplyr::first(region),
            is_pos = any(is_pos), .groups = "drop")

df <- attn %>%
  left_join(sel_site, by = c("hxb2_pos" = "hxb2_position"))
df$is_pos[is.na(df$is_pos)] <- FALSE

hxb2_max <- max(df$hxb2_pos, na.rm = TRUE)

# --- Env regions + CD4bs annotation (same as ch505_FEL/MEME figures) ---
env_regions <- as.data.frame(readxl::read_excel(fig_xlsx, sheet = "env_regions"))
env_regions$start <- env_regions$hxb2_start
env_regions$end   <- env_regions$hxb2_end
env_regions$mid   <- (env_regions$start + env_regions$end) / 2
palette2 <- c("#D3E2F2", "#FFFFFF")
region_colors <- palette2[(seq_len(nrow(env_regions)) - 1) %% 2 + 1]
text_col <- "#26313D"

# CD4bs contact intervals (Env_HXB2_feature.xlsx, Type = CD4 contact site)
cd4bs_hxb2 <- data.frame(
  hxb2_start = c(124, 196, 198, 279, 364, 425, 455, 469, 471),
  hxb2_end   = c(127, 196, 198, 283, 374, 432, 461, 469, 477)
)
cd4bs_fill <- "#F4C95D"

add_region_bg <- function() {
  annotate("rect",
           xmin = env_regions$start - 0.5, xmax = env_regions$end + 0.5,
           ymin = -Inf, ymax = Inf, fill = region_colors, alpha = 0.7)
}
add_region_labels <- function() {
  annotate("text", x = env_regions$mid, y = Inf, label = env_regions$region,
           vjust = 1.7, size = 1.9, fontface = "bold", colour = text_col)
}
add_cd4bs_bg <- function() {
  annotate("rect",
           xmin = cd4bs_hxb2$hxb2_start - 0.5, xmax = cd4bs_hxb2$hxb2_end + 0.5,
           ymin = -Inf, ymax = Inf, fill = cd4bs_fill, alpha = 0.18)
}

# --- Statistics ---
wt <- wilcox.test(attention_over_mean ~ is_pos, data = df, alternative = "less")
# 'less': non-pos < pos  (one-sided, selected sites higher)
p_mw <- wt$p.value
med_pos <- median(df$attention_over_mean[df$is_pos], na.rm = TRUE)
med_non <- median(df$attention_over_mean[!df$is_pos], na.rm = TRUE)

region_levels <- c("SP","C1","V1","V2","C2","V3","C3","V4","C4","V5","C5","gp41")
reg <- df %>%
  filter(!is.na(region) & region %in% region_levels) %>%
  group_by(region) %>%
  summarise(mean_attn = mean(attention_over_mean, na.rm = TRUE),
            pos_pct = 100 * mean(is_pos), n = dplyr::n(), .groups = "drop")
reg$region <- factor(reg$region, levels = region_levels)
r_region <- cor(reg$mean_attn, reg$pos_pct, use = "complete.obs")

cat(sprintf("Mann-Whitney (one-sided, selected>other) p = %.2e\n", p_mw))
cat(sprintf("median attention: selected = %.3f, other = %.3f\n", med_pos, med_non))
cat(sprintf("region-level Pearson r (mean_attn vs pos_pct) = %.3f\n", r_region))

fmt_p <- function(p) if (p < 1e-3) "P < 0.001" else sprintf("P = %.3f", p)

# --- Panel A top: attention track ---
pA_top <- ggplot(df, aes(x = hxb2_pos, y = attention_over_mean_smoothed)) +
  add_region_bg() + add_region_labels() + add_cd4bs_bg() +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "#777777", linewidth = 0.3) +
  geom_line(colour = col_attn, linewidth = 0.5) +
  scale_x_continuous(limits = c(1, hxb2_max), breaks = seq(0, hxb2_max, 100),
                     expand = expansion(mult = c(0.01, 0.01))) +
  scale_y_continuous(expand = expansion(mult = c(0.03, 0.12))) +
  labs(title = "Cross-attention and positive selection converge on gp120 variable regions",
       x = NULL, y = "Attention / mean", tag = "A") +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        plot.margin = margin(3, 6, 0, 5))

# --- Panel A bottom: positive-selection sites ---
pos_sites <- df[df$is_pos, ]
pA_bot <- ggplot(pos_sites, aes(x = hxb2_pos)) +
  add_region_bg() + add_cd4bs_bg() +
  geom_segment(aes(xend = hxb2_pos, y = 0, yend = 1), colour = col_pos, linewidth = 0.4) +
  scale_x_continuous(limits = c(1, hxb2_max), breaks = seq(0, hxb2_max, 100),
                     expand = expansion(mult = c(0.01, 0.01))) +
  scale_y_continuous(limits = c(0, 1), breaks = NULL) +
  labs(x = "HXB2 Env residue position", y = "Positively-\nselected sites", tag = "") +
  theme(panel.grid.major.y = element_blank(),
        axis.title.y = element_text(size = 6),
        plot.margin = margin(0, 6, 3, 5))

# --- Panel B: attention at selected vs other ---
n_other <- sum(!df$is_pos, na.rm = TRUE)
n_pos   <- sum(df$is_pos,  na.rm = TRUE)
lev <- c("Other", "Positively\nselected")
df$grp <- factor(ifelse(df$is_pos, lev[2], lev[1]), levels = lev)
ymax_b <- quantile(df$attention_over_mean, 0.995, na.rm = TRUE)
ymin_b <- min(df$attention_over_mean, na.rm = TRUE)
med_df <- data.frame(grp = factor(lev, levels = lev),
                     med = c(med_non, med_pos),
                     col = c("#5B6770", col_pos))
pB <- ggplot(df, aes(x = grp, y = attention_over_mean, fill = grp)) +
  geom_violin(linewidth = 0.25, colour = "white", width = 0.85, alpha = 0.9,
              scale = "width") +
  geom_boxplot(width = 0.15, outlier.shape = NA, linewidth = 0.3, fill = "white") +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "#777777", linewidth = 0.3) +
  geom_text(data = med_df, aes(x = grp, y = med, label = sprintf("%.2f", med)),
            inherit.aes = FALSE, nudge_x = -0.40, hjust = 1, size = 2.0,
            colour = med_df$col) +
  annotate("segment", x = 1, xend = 2, y = ymax_b * 1.03, yend = ymax_b * 1.03,
           linewidth = 0.3, colour = "#444444") +
  annotate("text", x = 1.5, y = ymax_b * 1.05, label = fmt_p(p_mw),
           size = 2.4, vjust = 0) +
  scale_fill_manual(values = setNames(c(col_grey, col_pos), lev), guide = "none") +
  scale_x_discrete(labels = c(sprintf("Other\n(n = %d)", n_other),
                              sprintf("Positively\nselected\n(n = %d)", n_pos))) +
  coord_cartesian(ylim = c(ymin_b, ymax_b * 1.12)) +
  labs(title = "Site-level", x = NULL, y = "Attention / mean", tag = "B") +
  theme(plot.margin = margin(3, 6, 3, 5))

# --- Panel C: region-level ---
v_loops <- c("V1", "V2", "V3", "V4", "V5")
reg$class <- ifelse(reg$region %in% v_loops, "Variable loop", "Conserved / other")
ct <- suppressWarnings(cor.test(reg$mean_attn, reg$pos_pct))
pC <- ggplot(reg, aes(x = mean_attn, y = pos_pct)) +
  geom_smooth(method = "lm", se = TRUE, colour = "#9AA0A6", fill = "#E3E7EC",
              linewidth = 0.4, linetype = "dashed", alpha = 0.6) +
  geom_point(aes(colour = class), size = 1.9, alpha = 0.95) +
  ggrepel::geom_text_repel(aes(label = region, colour = class), size = 2.1,
                           max.overlaps = Inf, segment.size = 0.15,
                           min.segment.length = 0, show.legend = FALSE) +
  annotate("text", x = -Inf, y = Inf, hjust = -0.12, vjust = 1.5,
           label = sprintf("Pearson r = %.2f, %s", r_region, fmt_p(ct$p.value)),
           size = 2.4) +
  scale_colour_manual(values = c("Variable loop" = col_pos,
                                 "Conserved / other" = col_attn), name = NULL) +
  labs(title = "Region-level", x = "Mean attention / mean",
       y = "Positively-selected sites (%)", tag = "C") +
  theme(plot.margin = margin(3, 6, 3, 5),
        legend.position = c(0.99, 0.02), legend.justification = c(1, 0),
        legend.key.size = unit(0.26, "cm"), legend.text = element_text(size = 5.6),
        legend.background = element_rect(
          fill = grDevices::adjustcolor("white", 0.6), colour = NA))

# --- Compose (panels D + E only; the attention track is Figure 5A, panel_A_attention) ---
fig <- (pB | pC)

ggsave(out_pdf, plot = fig, width = 183 / 25.4, height = 74 / 25.4,
       units = "in", device = cairo_pdf)
cat("Wrote:", out_pdf, "\n")

}

res <- try(panel_DE(), silent=FALSE)
if (inherits(res,'try-error')) message('[panel DE FAILED]')
for (ff in list.files(out_dir, full.names=TRUE)) if (!grepl('\\.pdf$', ff)) unlink(ff)
message('Figure 5 R panel -> ', out_dir)

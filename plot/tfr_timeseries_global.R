###############################################################################
# Author: Emily Goren based on some code from Julian Chalek
# Date: 8/09/2019
# TFR time series panel plots
###############################################################################

rm(list=ls())
library(tidyverse)
library(mortdb, lib.loc = "/ihme/mortality/shared/r/")
source("xarray_to_r.R")

# Paths
FBD_DIR  <- "/ihme/forecasting/data/5/future"
PLT_DIR  <- "/ihme/forecasting/plot/5/future"
WPP_DIR  <- "/home/j/WORK/02_fertilitypop/fertility/gbd_2019/forecasting/data"
save_version <- "20190806_141418_fix_draw_bound_ccfx_to2110"
base <- "20190806_141418_fix_draw_bound_ccfx_to2110_combined"
sdg <- "20190807_164000_fix_draw_bound_ccfx_sdg_to2110_scen_swapped_combined"
fastest <- "20190807_163915_fix_draw_bound_ccfx_99th_to2110_combined"
forecast_start <- 2018

# location data
locs <- mortdb::get_locations(level = "all", gbd_year = 2017) %>%
  filter(level <= 3 ) %>%
  select(c("location_id", "location_name", "ihme_loc_id"))


# Base scenarios
get_tfr_data <- function(version, labs) {
  tfrdat <- "tfr_combined.nc"
  #tfrmean <- "tfr_agg_mean.nc"
  #tfrq <- "tfr_agg_quantile.nc"
  dat <- xarray_nc_to_R(sprintf("%s/tfr/%s/%s", FBD_DIR, version, tfrdat))
  est <- dat[quantile == "mean"] #xarray_nc_to_R(sprintf("%s/tfr/%s/%s", FBD_DIR, version, tfrmean))
  UI <- dat[quantile != "mean"] #xarray_nc_to_R(sprintf("%s/tfr/%s/%s", FBD_DIR, version, tfrq))
  UI <- UI %>% spread(quantile, value)
  all <- merge(est, UI, by = c("scenario", "year_id", "location_id"))
  all <- all %>%
    mutate(Scenario = recode(scenario, `-1` = labs$`-1`, `0` = labs$`0`, `1` = labs$`1`))
  return(all)
}

labs_base <- list(
  `-1` = "Slower Met Need and Education Pace",
  `0` = "Reference with 95% UI",
  `1` =  "Faster Met Need and Education Pace")
labs_sdg  <- list(
  `-1` = "SDG Met Need and Education Pace",
  `0` = "DROP",
  `1` =  "DROP")
labs_fastest  <-  list(
  `-1` = "DROP",
  `0` = "DROP",
  `1` =  "Fastest Met Need and Education Pace")
d_base <- get_tfr_data(base, labs_base)
d_sdg <- get_tfr_data(sdg, labs_sdg)
d_fastest <- get_tfr_data(fastest, labs_fastest)
d_future <- rbind(d_base, d_sdg, d_fastest) %>% #d_sdg, d_fastest) %>%
  filter(Scenario != "DROP") %>%
  rename(mean = value) %>%
  left_join(locs, by = c("location_id"))

library(data.table)
d_past <- fread("/ihme/fertilitypop/fertility/gbd_2017/modeling/asfr/va84/loop2/results/gpr/compiled_summary_gpr.csv")
d_past <- d_past %>%
  filter(age == "tfr" & year < forecast_start & ihme_loc_id %in% locs$ihme_loc_id) %>%
  mutate(year_id = floor(year),
         scenario = 0,
         Scenario = "Reference with 95% UI") %>%
  select(-c("age", "year")) %>%
  left_join(locs, by = c("ihme_loc_id"))


d <- rbind(data.table(d_future), data.table(d_past), fill = TRUE) %>%
  filter(year_id %in% 1990:2100) %>%
  arrange(location_id, Scenario, year_id)
d$lower <- ifelse(d$year_id < forecast_start, d$mean, d$lower)
d$upper <- ifelse(d$year_id < forecast_start, d$mean, d$upper)

# Plot

cols <- c("Reference with 95% UI" = "steelblue",
          "Slower Met Need and Education Pace" = "firebrick",
          "Faster Met Need and Education Pace" = "forestgreen",
          "Fastest Met Need and Education Pace" = "#984ea3",
          "SDG Met Need and Education Pace" = "#ff7f00")

theme_set(theme_bw(base_size = 14))
pdf(sprintf("%s/ccf/%s/tfr_time_series_all_scenarios_agg_finalpop_refUI.pdf", PLT_DIR, save_version),
    width = 15, height = 8)
for (l in unique(d$location_name)) {
  dl <- d %>% filter(location_name == l) %>% droplevels()
  tf <- dl %>% filter(year_id == 2100 & Scenario == "Reference with 95% UI")
  plt <- ggplot() +
    geom_line(data = dl,
              mapping = aes(x=year_id, y=mean, color=Scenario), size = 1) +
    geom_line(data = filter(dl, Scenario == "Reference with 95% UI"),
              mapping = aes(x=year_id, y=mean, color=Scenario), size = 1) +
    geom_line(size=1) +
    geom_vline(xintercept = 2017, linetype="dashed", color = "black") +
    scale_color_manual(values = cols) +
    ggtitle(sprintf("%s: reference TFR in 2100: %s", l, round(tf$mean,3))) +
    ylab("Total Fertility Rate") +
    xlab("Year") +
    expand_limits(y = 0) +
    scale_x_continuous(breaks = seq(1990, 2100, 10)) +
    labs(color="Scenario") +
    theme(axis.text=element_text(size=12),
          axis.text.x = element_text(angle = 45, hjust = 1),
          axis.title=element_text(size=12)) +
    geom_ribbon(data = filter(dl, Scenario == "Reference with 95% UI"),
                mapping = aes(x=year_id, ymin=lower, ymax=upper, fill=Scenario),
                alpha = 0.3, fill = "steelblue")
  print(plt)
}
dev.off()

cairo_pdf(
  sprintf("%s/ccf/%s/tfr_time_series_all_scenarios_agg_finalpop_GLOBALonly-final_refUI.pdf", PLT_DIR, save_version),
  width = 15, height = 8)
  dl <- d %>% filter(location_name == "Global") %>% droplevels()
  ggplot() +
    geom_line(data = dl,
              mapping = aes(x=year_id, y=mean, color=Scenario), size = 1) +
    geom_line(data = filter(dl, Scenario == "Reference with 95% UI"),
              mapping = aes(x=year_id, y=mean, color=Scenario), size = 1) +
    geom_line(size=1) +
    geom_ribbon(data = filter(dl, Scenario == "Reference with 95% UI"),
                mapping = aes(x=year_id, ymin=lower, ymax=upper, fill=Scenario),
                alpha = 0.3, fill = "steelblue") +
    geom_vline(xintercept = 2017, linetype="dashed", color = "black") +
    scale_color_manual(values = cols) +
    ggtitle("Global TFR time series") +
    ylab("Total Fertility Rate") +
    xlab("Year") +
    expand_limits(y = 0) +
    scale_x_continuous(breaks = seq(1990, 2100, 10)) +
    labs(color="Scenario") +
    theme(axis.text=element_text(size=12),
          axis.text.x = element_text(angle = 45, hjust = 1),
          axis.title=element_text(size=12))
dev.off()

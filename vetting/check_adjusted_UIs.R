###############################################################################
#### Purpose: check output from draw-ordered adjusted UIS for pops paper resub
#### EMG 18 Nov 2019
###############################################################################

setwd("~/repos/fhs_2019_population_paper/vetting")

library(reticulate, lib.loc = "/ihme/forecasting/shared/r")
library(tidyverse, lib.loc = "/ihme/forecasting/shared/r")
library(foreach)
theme_set(theme_bw(base_size = 14))

TFR_VERSIONS <- c(
    "20190806_141418_fix_draw_bound_ccfx_to2110",
    "20190807_163915_fix_draw_bound_ccfx_99th_to2110",
    "20190807_164000_fix_draw_bound_ccfx_sdg_to2110_scen_swapped",
    "20190806_141418_fix_draw_bound_ccfx_to2110_uncertainty_adjusted",
    "20190807_164000_fix_draw_bound_ccfx_sdg_to2110_scen_swapped_uncertainty_adjusted",
    "20190807_163915_fix_draw_bound_ccfx_99th_to2110_uncertainty_adjusted",
    "20190806_141418_fix_draw_bound_ccfx_to2110_uncertainty_adjusted_combined",
    "20190807_163915_fix_draw_bound_ccfx_99th_to2110_combined",
    "20190807_164000_fix_draw_bound_ccfx_sdg_to2110_scen_swapped_combined",
    "20190806_141418_fix_draw_bound_ccfx_to2110_addsex_ordered_20191118",
    "20190807_163915_fix_draw_bound_ccfx_99th_to2110_addsex_ordered_20191118",
    "20190807_164000_fix_draw_bound_ccfx_sdg_to2110_scen_swapped_addsex_ordered_20191118"
)

POP_VERSIONS <- c(
    "20191030_15_ref_85_resub_agg_arima_squeeze_shocks_only_decay_wt_15_hiv_all",
    "20191030_15_ref_85_resub_agg_arima_squeeze_shocks_only_decay_wt_15_hiv_all_ordered",
    "20191030_15_ref_85_resub_agg_arima_squeeze_shocks_only_decay_wt_15_hiv_all_combined",
    "20191030_15_ref_85_resub_agg_arima_squeeze_shocks_only_decay_wt_15_hiv_all_uncertainty_adjusted",
    "20191106_15_ref_99_resub_fix_sdi_agg_arima_squeeze_shocks_hiv_all",
    "20191106_15_ref_99_resub_fix_sdi_agg_arima_squeeze_shocks_hiv_all_ordered",
    "20191106_15_ref_99_resub_fix_sdi_agg_arima_squeeze_shocks_hiv_all_combined",
    "20191106_15_ref_99_resub_fix_sdi_agg_arima_squeeze_shocks_hiv_all_uncertainty_adjusted",
    "20191106_sdg_ref_15_resub_fix_sdi_agg_arima_squeeze_shocks_hiv_all",
    "20191106_sdg_ref_15_resub_fix_sdi_agg_arima_squeeze_shocks_hiv_all_ordered",
    "20191106_sdg_ref_15_resub_fix_sdi_agg_arima_squeeze_shocks_hiv_all_combined",
    "20191106_sdg_ref_15_resub_fix_sdi_agg_arima_squeeze_shocks_hiv_all_uncertainty_adjusted"
)

last_word <- function(x, sep = "_"){
    x <- sapply(strsplit(x, sep),"[")
    trimws(x[[length(x)]])
}

reticulate::use_condaenv(
    condaenv = "emg36",
    conda = "/ihme/forecasting/share/miniconda3/bin/conda",
    required = TRUE)

reticulate::source_python("/ihme/homes/egoren/repos/fhs_miscellaneous/egoren/utils/read_xr.py")


# Location maps
source('/ihme/cc_resources/libraries/current/r/get_location_metadata.R')

location_map <- get_location_metadata(location_set_id = 39, gbd_round_id = 5)


###############################################################################
# TFR
tfr_scenario_labels <- function(version_name) {
    if (grepl("99", version_name)) {
        out <- c("dupe", "dupe", "Fastest")
    } else if (grepl("sdg", version_name)) {
        out <- c("SDG", "dupe", "dupe")
    } else {
        out <- c("Slower", "Reference", "Faster")
    }
    return(out)
}

tfrs <- foreach(v = TFR_VERSIONS, .combine = "bind_rows") %do% {
    type <- last_word(v)
    if (type == "combined") {
        d <- xarray_to_R(stage = "tfr", past_or_future = "future",
                         gbd_round_id = 5L, version = v,
                         drop_scenario = FALSE, filename = "tfr_combined.nc")
        d$type <- "Combined"
    } else if (type == "adjusted") {
        d <- xarray_to_R(stage = "tfr", past_or_future = "future",
                         gbd_round_id = 5L, version = v,
                         drop_scenario = FALSE, filename = "tfr_agg_adjusted.nc")
        d$type <- "Adjusted"
    } else {
        d_mean <- xarray_to_R(stage = "tfr", past_or_future = "future",
                              gbd_round_id = 5L, version = v,
                              drop_scenario = FALSE, filename = "tfr_agg_mean.nc")
        d_mean$quantile <- "mean"
        d_quantile <- xarray_to_R(stage = "tfr", past_or_future = "future",
                                  gbd_round_id = 5L, version = v,
                                  drop_scenario = FALSE, filename = "tfr_agg_quantile.nc")
        d_quantile$quantile <- ifelse(d_quantile$quantile == 0.025, "lower", "upper")
        d <- bind_rows(d_mean, d_quantile)
        d$type <- ifelse(type == "20191118", "Ordered", "Unordered")
    }
    d$version <- v
    d$Scenario <- factor(d$scenario, levels = -1:1, labels = tfr_scenario_labels(v))
    return(d)
}
tfrs <- tfrs %>% left_join(location_map)
tfrs$Scenario <- ordered(tfrs$Scenario,
                         levels = c("Slower", "Reference", "Faster",
                                    "Fastest", "SDG", "dupe"),
                         labels = c("Slower", "Reference", "Faster",
                                    "Fastest", "SDG", "dupe"))

pdf("./TFR_UI_vetting_plots.pdf", width = 12, height = 6)
for (lid in sort(unique(tfrs$location_id))) {
    df <- subset(tfrs, location_id == lid & year_id > 2017 & Scenario != "dupe")
    loc <- unique(df$location_name)
    gg <- ggplot() +
        geom_line(data = subset(df, quantile == "mean"),
                  mapping = aes(x = year_id, y = value, color = type, linetype = type)) +
        geom_line(data = subset(df, quantile == "lower"),
                  mapping = aes(x = year_id, y = value, color = type, linetype = type)) +
        geom_line(data = subset(df, quantile == "upper"),
                  mapping = aes(x = year_id, y = value, color = type, linetype = type)) +
        facet_wrap(~Scenario, ncol = 5) +
        theme(legend.position = "bottom") +
        labs(y = "TFR", x = "Year", color = "Version Type", linetype = "Version Type") +
        scale_color_manual(values = c("green", "purple", "orange", "black")) +
        scale_linetype_manual(values = c("dashed", "longdash", "solid", "dotted")) +
        ggtitle(loc)
    print(gg)
}
dev.off()



###############################################################################
# pop
pop_scenario_labels <- function(version_name) {
    if (grepl("15_ref_85", version_name)) {
        out <- c("Slower", "Reference", "Faster")
    } else if (grepl("15_ref_99", version_name)) {
        out <- c("dupe", "dupe", "Fastest")
    } else if (grepl("sdg_ref_15", version_name)) {
        out <- c("SDG", "dupe", "dupe")
    }
    return(out)
}

pops_list <- foreach(v = POP_VERSIONS, .combine = "c") %do% {
    type <- last_word(v)
    if (type == "combined") {
        d <- xarray_to_R(stage = "population", past_or_future = "future",
                         gbd_round_id = 5L, version = v, drop_scenario = FALSE,
                         filename = "population_combined.nc",
                         select_sid = 3, select_age_group_ids = 23)
        d$type <- "Combined"
    } else if (type == "adjusted") {
        d <- xarray_to_R(stage = "population", past_or_future = "future",
                         gbd_round_id = 5L, version = v, drop_scenario = FALSE,
                         filename = "population_agg_adjusted.nc",
                         select_sid = 3, select_age_group_ids = 23)
        d$type <- "Adjusted"
    } else {
        d <- xarray_to_R_summaries(stage = "population",
                                   past_or_future = "future",
                                   gbd_round_id = 5L, version = v,
                                   filename = "population_agg.nc", test = FALSE,
                                   select_sid = 3, select_age_group_ids = 23)
        d$type <- ifelse(type == "ordered", "Ordered", "Unordered")
        if ("population" %in% names(d))
            names(d)[names(d) == "population"] <- "value"
    }
    d$version <- v
    d$Scenario <- factor(d$scenario, levels = -1:1, labels = pop_scenario_labels(v))
    return(list(d))
}

pops <- rbind_list(pops_list) %>% left_join(location_map)
pops$Scenario <- ordered(pops$Scenario,
                         levels = c("Slower", "Reference", "Faster",
                                    "Fastest", "SDG", "dupe"),
                         labels = c("Slower", "Reference", "Faster",
                                    "Fastest", "SDG", "dupe"))

pdf("./population_UI_vetting_plots.pdf", width = 12, height = 6)
for (lid in sort(unique(pops$location_id))) {
    df <- subset(pops, location_id == lid & year_id > 2017 & Scenario != "dupe")
    loc <- unique(df$location_name)
    gg <- ggplot() +
        geom_line(data = subset(df, quantile == "mean"),
                  mapping = aes(x = year_id, y = value, color = type, linetype = type)) +
        geom_line(data = subset(df, quantile == "lower"),
                  mapping = aes(x = year_id, y = value, color = type, linetype = type)) +
        geom_line(data = subset(df, quantile == "upper"),
                  mapping = aes(x = year_id, y = value, color = type, linetype = type)) +
        facet_grid(age_group_id + sex_id ~ Scenario) +
        theme(legend.position = "bottom") +
        labs(y = "Population (all ages, both sexes)", x = "Year",
             color = "Version Type", linetype = "Version Type") +
        scale_color_manual(values = c("green", "purple", "orange", "black")) +
        scale_linetype_manual(values = c("dashed", "longdash", "solid", "dotted")) +
        ggtitle(loc) +
        scale_y_continuous(labels = scales::scientific)
    print(gg)
}
dev.off()

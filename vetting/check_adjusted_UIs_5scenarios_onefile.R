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
    "20190806_141418_fix_draw_bound_ccfx_to2110_5_scen",
    "20190806_141418_fix_draw_bound_ccfx_to2110_5_scen_uncertainty_adjusted",
    "20190806_141418_fix_draw_bound_ccfx_to2110_5_scen_combined",
    "20190806_141418_fix_draw_bound_ccfx_to2110_5_scen_ordered"
)

POP_VERSIONS <- c(
    "20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all",
    "20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all_combined",
    "20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all_ordered",
    "20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all_uncertainty_adjusted"
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

scenario_labels <- c("Slower", "Reference", "Faster", "Fastest", "SDG")


###############################################################################
# TFR

tfrs <- foreach(v = TFR_VERSIONS, .combine = "bind_rows") %do% {
    type <- last_word(v)
    if (type == "combined") {
        d <- xarray_to_R(stage = "tfr", past_or_future = "future",
                         gbd_round_id = 5L, version = v,
                         drop_scenario = FALSE, filename = "test_tfr_agg_combined.nc")
        d$type <- "Combined"
    } else if (type == "adjusted") {
        d <- xarray_to_R(stage = "tfr", past_or_future = "future",
                         gbd_round_id = 5L, version = v,
                         drop_scenario = FALSE, filename = "test_tfr_agg_adjusted.nc")
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
        d$type <- ifelse(type == "ordered", "Ordered", "Unordered")
    }
    d$version <- v
    d$Scenario <- ordered(d$scenario, levels = -1:3, labels = scenario_labels)
    return(d)
}
tfrs <- tfrs %>% left_join(location_map)

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

pops_list <- foreach(v = POP_VERSIONS, .combine = "c") %do% {
    type <- last_word(v)
    if (type == "combined") {
        d <- xarray_to_R(stage = "population", past_or_future = "future",
                         gbd_round_id = 5L, version = v, drop_scenario = FALSE,
                         filename = "population_combined.nc",
                         select_sid = 3, select_age_group_ids = 22)
        d$type <- "Combined"
    } else if (type == "adjusted") {
        d <- xarray_to_R(stage = "population", past_or_future = "future",
                         gbd_round_id = 5L, version = v, drop_scenario = FALSE,
                         filename = "population_agg_adjusted.nc",
                         select_sid = 3, select_age_group_ids = 22)
        d$type <- "Adjusted"
    } else {
        d <- xarray_to_R_summaries(stage = "population",
                                   past_or_future = "future",
                                   gbd_round_id = 5L, version = v,
                                   filename = "population_agg.nc", test = FALSE,
                                   select_sid = 3, select_age_group_ids = 22)
        d$type <- ifelse(type == "ordered", "Ordered", "Unordered")
        if ("population" %in% names(d))
            names(d)[names(d) == "population"] <- "value"
    }
    d$version <- v
    d$Scenario <- ordered(d$scenario, levels = -1:3, labels = scenario_labels)
    return(list(d))
}

pops <- bind_rows(pops_list) %>% left_join(location_map)

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

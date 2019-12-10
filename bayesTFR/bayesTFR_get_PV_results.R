# PURPOSE
# Predictive Validity of Fertility
# using the WPP2019 TFR model
# fit to past GBD2017 data
# years 1980-2007 (use 2/7 midpoints of intervals)
# for population paper results appendix
#
# This code specifically extracts forecasts from
# the WPP2019 bayesTFR software run and computes
# predictive validity metrics
#
# AUTHOR
# Emily Goren
#

library(data.table)
library(tidyverse, lib.loc = "/ihme/forecasting/shared/r")
library(mortdb, lib.loc = "/ihme/mortality/shared/r")
library(reticulate, lib.loc = "/ihme/forecasting/shared/r")
library(mortcore, lib.loc = "/ihme/mortality/shared/r")


gbd_round <- 5
gbd_year <- 2017
first_forecast_year <- 2012
last_past_year <- 2017
condaenv <- "twh36"
tfr_yrs <- c(first_forecast_year, last_past_year)

# File paths
FBDpast <- sprintf("/ihme/forecasting/data/%i/past", gbd_round)
dir <- "/ihme/homes/egoren/repos/fertility_working/fbd_research/fbd_research/fertility/ccf/bayesTFR"
version <- "20190826_WPP2019_fit_to_GBD2017_PV_10yr"
codepath <- "/ihme/homes/egoren/repos/fertility_working/fbd_research/fbd_research/fertility/ccf"
setwd(codepath)
source("xarray_to_r.R")


# Read in age and location information
locs <- mortdb::get_locations(gbd_year = gbd_year, level = "country")
age_map  <- mortdb::get_age_map()[,.(age_group_id, age_group_years_start, age_group_name)]


# Functions to compute summary measures
# root mean squared error
compute_RMSE <- function(y, yhat) {
    sqrt(mean((y - yhat)^2, na.rm = TRUE))
}
#  mean absolute percentage error
compute_MAPE <- function(y, yhat) {
    100 * mean(abs((y - yhat) / y), na.rm = TRUE)
}
#  median absolute deviation
compute_MAD <- function(y, yhat) {
    median(abs(y - yhat), na.rm = TRUE)
}
# mean absolute deviation
compute_MAE <- function(y, yhat) {
    mean(abs(y - yhat), na.rm = TRUE)
}


##### TFR

# Extract past data for TFR -- pull from MortDB
tfr_past <- xarray_nc_to_R(sprintf("%s/tfr/%s/tfr.nc", FBDpast, "20190109_va84"))
tfr_past_mean <- tfr_past %>%
    group_by(year_id, location_id) %>%
    summarise(value = mean(value)) %>%
    left_join(locs, by = "location_id") %>%
    data.table()
# Read in forecasted TFR
tfr_pred_mean <- fread(sprintf("%s/%s/predictions_using_gbd2017.csv", dir, version)) %>%
    left_join(locs, by = "ihme_loc_id") %>%
    data.table()

# Subset on holdout years
tfr_past_mean <- tfr_past_mean[between(year_id, first_forecast_year, last_past_year), .(location_id, super_region_name, year_id, value)]
tfr_pred_mean <- tfr_pred_mean[between(year_id, first_forecast_year, last_past_year), .(location_id, super_region_name, year_id, value)]

setnames(tfr_past_mean, old = "value", new = "full")
setnames(tfr_pred_mean, old = "value", new = "holdout")
tfr_all <- merge(tfr_past_mean, tfr_pred_mean, by = c("location_id", "super_region_name", "year_id"))

tfr_sum <- tfr_all %>%
    summarise(RMSE = compute_RMSE(full, holdout),
              MAPE = compute_MAPE(full, holdout),
              MAD = compute_MAD(full, holdout),
              MAE = compute_MAE(full, holdout))

fwrite(tfr_sum, sprintf("%s/%s/predictive_validity_TFR.csv", dir, version))

### Find countries below replacement in 2017
below_replacement <- tfr_past_mean %>%
    filter(year_id == last_past_year & full < 2.1)
tfr_all$below_rpl <- ifelse(tfr_all$location_id %in% below_replacement$location_id, "Below2.1", "Atleast2.1")

tfr_sum_stratified <- tfr_all %>%
    group_by(below_rpl) %>%
    summarise(RMSE = compute_RMSE(full, holdout),
              MAPE = compute_MAPE(full, holdout),
              MAD = compute_MAD(full, holdout),
              MAE = compute_MAE(full, holdout))
fwrite(tfr_sum_stratified, sprintf("%s/%s/predictive_validity_TFR_stratified_by_replacement.csv", dir, version))

### Plot of TFR
pdf(sprintf("%s/%s/predictive_validity_TFR.pdf", dir, version), width = 12, height = 15)
plt <- ggplot(tfr_all) +
    geom_point(aes(x = full, y = holdout, color = as.factor(super_region_name))) +
    labs(color = "Super\nRegion",
         x = sprintf("Observed TFR"),
         y = sprintf("Forecasted TFR")) +
    theme_bw() +
    geom_abline(slope = 1, intercept = 0, color = "black") +
    theme(legend.position = "bottom") +
    scale_color_brewer(palette = 'Dark2') +
    theme(legend.title = element_text(size=10), legend.text=element_text(size=10))
print(plt)
dev.off()

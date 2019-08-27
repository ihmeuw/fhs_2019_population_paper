## Purpose: determine change in global population if TFR below replacement was increased by 0.1
## EMG, 20190819

rm(list = ls())


library(data.table)
library(tidyverse, lib.loc = "/ihme/forecasting/shared/r")
library(mortdb, lib.loc = "/ihme/mortality/shared/r")
library(reticulate, lib.loc = "/ihme/forecasting/shared/r")
library(mortcore, lib.loc = "/ihme/mortality/shared/r")


version <- "20190806_141418_fix_draw_bound_ccfx_to2110"
version_out <- paste0(version, "_adjust_tfr_by_point1")
gbd_round <- 5L
gbd_year <- 2017
last_past_year <- 2017
condaenv <- "twh36"

# File paths
FBDfuture <- sprintf("/ihme/forecasting/data/%i/future", gbd_round)
FBDplot <- sprintf("/ihme/forecasting/plot/%i/future", gbd_round)

codepath <- "/ihme/homes/egoren/repos/fertility_working/fbd_research/fbd_research/fertility/ccf"
setwd(codepath)
source("xarray_to_r.R")


# Read in age and location information
locs <- fread(sprintf("%s/ccf/%s/covariates/locs.csv", FBDfuture, version))
age_map  <- mortdb::get_age_map()[,.(age_group_id, age_group_years_start, age_group_name)]

# Read in ASFR, TFR draws
asfr <- xarray_nc_to_R(sprintf("%s/asfr/%s/asfr.nc", FBDfuture, version))
tfr <- xarray_nc_to_R(sprintf("%s/tfr/%s/tfr.nc", FBDfuture, version))

# subset on reference, >=2017
asfr <- asfr %>%
    filter(year_id >= last_past_year) %>%
    data.table()
tfr <- tfr %>%
    filter(year_id >= last_past_year) %>%
    mutate(tfr = value) %>%
    select(-value) %>%
    data.table()

# Find places with TFR < 2
tfr$below_replacement_by_point1 <- tfr$tfr < 2.0
below_replacement <- tfr %>%
    group_by(scenario, location_id, draw) %>%
    summarise(prop_below = mean(below_replacement_by_point1), first_year_below = min(year_id[below_replacement_by_point1]))
table(below_replacement$first_year_below)
summary(tfr$tfr < 2)

# merge tfr and asfr
d <- left_join(asfr, tfr, by = c("scenario", "year_id", "location_id", "draw"))

# add 0.1/(5*7) to age groups 8:14 (ages 15-49) if TFR < 2
reproductive_ages <- 8:14
delta <- 0.1/(5*length(reproductive_ages))
delta
d$below_replacement_by_point1[is.na(d$below_replacement_by_point1)] <- FALSE
d$adjust <- (d$below_replacement_by_point1 & (d$age_group_id %in% reproductive_ages))
d$asfr_adjusted <- ifelse(d$adjust, d$value + delta, d$value)

# check
cairo_pdf(sprintf("%s/ccf/%s/asfr_adjusted_st_TFR=TFR+0.1_if_below_replacement.pdf", FBDplot, version), width=20,height=10)
ggplot(subset(d, draw == 0)) +
    geom_point(aes(x = value, y = asfr_adjusted, color = year_id)) +
    facet_grid(~age_group_id) +
    geom_abline(slope = 1, intercept = 0) +
    labs(x = "Forecasted ASFR",
         y = "ASFR adjusted to TFR = TFR + 0.1 if TFR < 2",
         color = "Year") +
    ggtitle("Adjusting ASFR to raise TFR by 0.1 if less than 2 (draw 0 only)")
dev.off()

# recompute TFR and compare
tfr_check <- d %>%
    group_by(scenario, draw, year_id, location_id) %>%
    summarise(tfr = mean(tfr),
              tfr_adjusted = 5*sum(asfr_adjusted),
              tfr_recomputed = 5*sum(value),
              is_adjusted = mean(adjust))
summary(subset(tfr_check, is_adjusted == 0)$tfr_adjusted == subset(tfr_check, is_adjusted == 0)$tfr_recomputed)
summary(subset(tfr_check, is_adjusted == 0)$tfr_adjusted - subset(tfr_check, is_adjusted == 0)$tfr)
summary(subset(tfr_check, is_adjusted != 0)$tfr_adjusted - (subset(tfr_check, is_adjusted != 0)$tfr + 0.1))

# output
d$sex_id <- 2
out <- d %>%
    select(location_id, year_id, scenario, age_group_id, sex_id, draw, asfr_adjusted) %>%
    rename(value = asfr_adjusted)
fwrite(data.table(out), file = sprintf("%s/asfr/%s/asfr.csv", FBDfuture, version_out))

#### work in python from R
reticulate::use_condaenv(condaenv = condaenv, conda = "/ihme/forecasting/share/miniconda3/bin/conda", required = TRUE)
reticulate::source_python("diagnostic_code/adjust_tfr_by_point1_save_to_xarray.py")

output_to_xarray(gbd_round, out, version_out)

# check work
asfr_fixed <- xarray_nc_to_R(sprintf("%s/asfr/%s/asfr.nc", FBDfuture, version_out))
dim(out) == dim(asfr_fixed)
table(out$scenario)
table(asfr_fixed$scenario)
out %>% filter(scenario == 0 & year_id == 2100 & draw == 0 & location_id == 8)
asfr_fixed %>% filter(scenario == 0 & year_id == 2100 & draw == 0 & location_id == 8)
asfr %>% filter(scenario == 0 & year_id == 2100 & draw == 0 & location_id == 8)

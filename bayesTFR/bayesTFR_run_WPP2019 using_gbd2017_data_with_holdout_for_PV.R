# PURPOSE
# Predictive Validity of Fertility
# using the WPP2019 TFR model
# fit to past GBD2017 data
# years 1980-2007 (use 2/7 midpoints of intervals)
# for population paper results appendix
#
# This code runs the model.
# Use "bayesTFR_get_PV_results.R"
# to extracts forecasts from
# the WPP2019 bayesTFR software run and computes
# predictive validity metrics
#
# AUTHOR
# Emily Goren
#


library(data.table)
library(tidyverse)
library(mortdb, lib.loc = "/ihme/mortality/shared/r")
#install.packages("bayesTFR", lib = "/ihme/forecasting/shared/r", dependencies = TRUE)
library(bayesTFR, lib.loc = "/ihme/forecasting/shared/r")
#install.packages("wpp2019", lib = "/ihme/forecasting/shared/r", dependencies = TRUE, repos='http://cran.rstudio.com/')
library(wpp2019, lib.loc = "/ihme/forecasting/shared/r")

dir <- "/ihme/homes/egoren/repos/fertility_working/fbd_research/fbd_research/fertility/ccf/bayesTFR"
version <- "20190826_WPP2019_fit_to_GBD2017_PV_10yr"


# Run on GBD 2017 data
# input GBD asfr data and convert to tfr
gbd_year <- 2017
locs <- mortdb::get_locations(gbd_year = gbd_year, level = "country")
loc_mapping <- read_table(
    sprintf("%s/%s/iso3166_codes.txt", dir, version),
    col_names = TRUE) %>%
    mutate(ISO3166 = as.numeric(ISO3166))

# ID mapping for ihme data
locs_all <- locs %>%
    mutate(A3 = ihme_loc_id) %>%
    left_join(loc_mapping, by = "A3") %>%
    select(ihme_loc_id, location_name, ISO3166)
# UN Locations
data(UNlocations)
un.locs <- data.table(UNlocations)


asfr <- mortdb::get_mort_outputs(model_name = "asfr", model_type = "estimate", location_ids = locs$location_id, gbd_year = gbd_year)
tfr <- asfr %>%
    group_by(run_id, location_id, ihme_loc_id, year_id) %>%
    filter(age_group_id %in% 7:15) %>%
    summarise(mean = 5*sum(mean)) %>%
    mutate(year_grp_start = 5*floor(year_id/5)) %>%
    mutate(year_grp = paste0(year_grp_start, "-", year_grp_start + 5)) %>%
    left_join(locs_all, by = "ihme_loc_id")

# To convert to 5 year, only use midpoints
# need to structure s.t. the names are
#[1] "country_code"  "name"          "1950-1955"     "1955-1960"     "1960-1965"     "1965-1970"     "1970-1975"     "1975-1980"
#[9] "1980-1985"     "1985-1990"     "1990-1995"     "1995-2000"     "2000-2005"     "2005-2010"     "2010-2015"     "2015-2020"
#[17] "last.observed"
yrs <- c(seq(1982, 2002, by = 5))
tfr_bin <- tfr %>%
    filter(year_id %in% yrs) %>%
    mutate(country_code = ISO3166,
           name = location_name) %>%
    data.frame()
tfr_gbd_past <- tfr_bin %>%
    select(country_code, name, mean, year_grp) %>%
    spread(year_grp, mean) %>%
    mutate(last.observed = 2007)
loc_ids <-  un.locs %>% select(country_code)
tfr_gbd_past <- merge(tfr_gbd_past, loc_ids, by = "country_code", allow.cartesian = TRUE)
names(tfr_gbd_past)[names(tfr_gbd_past) == "name"] <- "country"
sim.dir <- sprintf("%s/%s/output", dir, version)
my.tfr.file <- sprintf("%s/%s/tfr_gbd_past.txt", dir, version)
write.table(tfr_gbd_past, my.tfr.file, row.names = FALSE, sep = "\t")
head(read.table(my.tfr.file, header = TRUE))

locs_table <-  un.locs
locs_table$include_code <- ifelse(locs_table$country_code %in% tfr_gbd_past$country_code, 2, 0)
my.locations.file <- sprintf("%s/%s/location_table.txt", dir, version)
write.table(locs_table, my.locations.file, row.names = FALSE, sep = "\t")
head(read.table(my.locations.file, header = TRUE))

# Phase II MCMCs
g <- run.tfr.mcmc(
    output.dir = sim.dir,
    my.tfr.file = my.tfr.file,
    my.locations.file = my.locations.file,
    seed = 1,
    replace.output = TRUE,
    verbose = TRUE,
    start.year = 1980,
    present.year = 2007,
    wpp.year = 2019,
    iter = 50000)
# Phase III MCMCs (not included in the package)
g3 <- run.tfr3.mcmc(
    sim.dir = sim.dir,
    replace.output = TRUE,
    seed = 1,
    verbose = TRUE,
    iter = 700000)
summary(g3)


# check convergence, phase III merged to our location IDs -- NOTE DONE AFTER QSUB

diag <- tfr.diagnose(sim.dir)
diag3 <- tfr3.diagnose(sim.dir)
has.mcmc.converged(diag)
has.mcmc.converged(diag3)

# continue MCMC
g.cont <- continue.tfr.mcmc(
    output.dir = sim.dir,
    my.tfr.file = my.tfr.file,
    iter = 2000)
g3.cont <- continue.tfr3.mcmc(
    sim.dir = sim.dir,
    iter = 5000)

diag <- tfr.diagnose(sim.dir)
diag3 <- tfr3.diagnose(sim.dir)
has.mcmc.converged(diag)
has.mcmc.converged(diag3)



g <- get.tfr.mcmc(sim.dir)
g3 <- get.tfr3.mcmc(sim.dir)

included <- g3$meta$id_phase3
phaseIII_locs <- data.frame(country_name = g3$meta$regions$country_name[included], country_code = g3$meta$regions$country_code[included])
phaseIII_locs
fwrite(phaseIII_locs, sprintf("%s/%s/bayesTFR_gbd2017_phase3.csv", dir, version))

# Prediction
pred <- tfr.predict(g, replace.output = TRUE)

# Read in predictions and use median prediction
# convert to long in time (rather than wide)
# extract year_id and add in IHME location info
preds <- fread(sprintf("%s/%s/output/predictions/projection_summary_user_friendly.csv", dir, version))
preds_med <- preds %>%
    filter(variant == "median") %>%
    select(-c("variant")) %>%
    gather("year", "value", -c(country_name, country_code)) %>%
    rename(ISO3166 = country_code) %>%
    mutate(year_id = as.numeric(substr(year, 1, 4)) + 2) %>%
    right_join(locs_all, by = "ISO3166")
preds_med$phaseIII <- ifelse(preds_med$country_name %in% phaseIII_locs$country_name, TRUE, FALSE)
fwrite(preds_med, sprintf("%s/%s/predictions_using_gbd2017.csv", dir, version))

ph3 <- fread(sprintf("%s/%s/bayesTFR_gbd2017_phase3.csv", dir, version)) %>%
    rename(ISO3166 = country_code) %>%
    right_join(locs_all, by = "ISO3166")
ph3

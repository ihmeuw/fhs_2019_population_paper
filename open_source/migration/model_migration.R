#!/usr/bin/env Rscript

require(plyr)
require(data.table)
require(grid)
require(gridExtra)
require(ggplot2)
require(RColorBrewer)
require(wpp2017)
library(matrixStats)
library(splines)
library(openxlsx)


GBD_ROUND_ID = 5


FORECAST_START = 2018

if (file.exists(sprintf("%scovariates_single_years.csv", DATA_DIR))) {
  mig_covs_pred = fread(sprintf("%scovariates_single_years.csv", DATA_DIR))
} else {
  # UN locs to IHME location_id
  loc_mapping <- fread(LOC_MAPPING_DIR)
  location_hierarchy = get_location_metadata(location_set_id=21, gbd_round_id=GBD_ROUND_ID)
  loc_mapping <- merge(loc_mapping, location_hierarchy, by = "location_id")
  loc_mapping <- loc_mapping[, list(location_id, iso_num, ihme_loc_id, super_region_name, region_name, location_name.x)]
  names(loc_mapping)[names(loc_mapping) == "location_name.x"] = "location_name"
  
  # Function to tidy, melt, and merge on location info
  wrangle_wpp_data <- function(wpp_data, data_column_name, location_mapping) {
    # Convert country code to iso_num for location merge
    names(wpp_data)[names(wpp_data) == "Country.code"] = "iso_num"
    wpp_data$iso_num = as.character(wpp_data$iso_num)
    wpp_data$iso_num = as.integer(wpp_data$iso_num)
    # Convert migration rate to numeric (no checks right now)
    wpp_data$migration_rate = as.numeric(wpp_data$migration_rate)
    # Merge on location_id and location_name info
    wpp_data = merge(wpp_data, loc_mapping, by = "iso_num")
    # Change five year interval display to just the first year and set to integer
    wpp_data$year_id = substr(wpp_data$year_id, start = 1, stop = 4)
    wpp_data$year_id = as.integer(wpp_data$year_id)
    # Subset columns
    wpp_data = wpp_data[,c("location_id", "location_name", "year_id", sprintf("%s", data_column_name)), with=FALSE]
    # if no 2100, copy 2095 to 2100
    if (max(wpp_data$year_id) < 2100){
      data2100 = wpp_data[year_id==2095,]
      data2100$year_id = 2100
      wpp_data = rbind(wpp_data, data2100)
    }
    # Remove duplicates if past and forecast overlap
    wpp_data = wpp_data[!duplicated(wpp_data),]
    return(wpp_data)
  }
  
  # Load and combine past/forecast migration rate
  past_mig_rate = as.data.table(read.xlsx(NET_MIGRATION_FILE, sheet = "ESTIMATES", startRow = 17, colNames = TRUE, na.strings = "…"))
  past_mig_rate = past_mig_rate[,-c("Index", "Variant", "Region,.subregion,.country.or.area.*", "Notes", "Type", "Parent.code")]
  past_mig_rate = melt(past_mig_rate, id.vars = c("Country.code"), variable.name = "year_id", value.name = "migration_rate")
  forecast_mig_rate = as.data.table(read.xlsx(NET_MIGRATION_FILE, sheet = "MEDIUM VARIANT", startRow = 17, colNames = TRUE, na.strings = "…"))
  forecast_mig_rate = forecast_mig_rate[,-c("Index", "Variant", "Region,.subregion,.country.or.area.*", "Notes", "Type", "Parent.code")]
  forecast_mig_rate = melt(forecast_mig_rate, id.vars = c("Country.code"), variable.name = "year_id", value.name = "migration_rate")
  mig_rate = rbind(past_mig_rate, forecast_mig_rate)
  rm(past_mig_rate); rm(forecast_mig_rate)
  mig_rate = wrangle_wpp_data(mig_rate, "migration_rate", loc_mapping)
  
  # Load and combine past/forecast natural population increase
  past_nat_pop = as.data.table(read.xlsx(NAT_POP_INC_FILE, sheet = "ESTIMATES", startRow = 17, colNames = TRUE, na.strings = "…"))
  past_nat_pop = past_nat_pop[,-c("Index", "Variant", "Region,.subregion,.country.or.area.*", "Notes", "Type", "Parent.code")]
  past_nat_pop = melt(past_nat_pop, id.vars = c("Country.code"), variable.name = "year_id", value.name = "natural_pop_increase")
  forecast_nat_pop = as.data.table(read.xlsx(NAT_POP_INC_FILE, sheet = "MEDIUM VARIANT", startRow = 17, colNames = TRUE, na.strings = "…"))
  forecast_nat_pop = forecast_nat_pop[,-c("Index", "Variant", "Region,.subregion,.country.or.area.*", "Notes", "Type", "Parent.code")]
  forecast_nat_pop = melt(forecast_nat_pop, id.vars = c("Country.code"), variable.name = "year_id", value.name = "natural_pop_increase")
  nat_pop = rbind(past_nat_pop, forecast_nat_pop)
  rm(past_nat_pop); rm(forecast_nat_pop)
  nat_pop = wrangle_wpp_data(nat_pop, "natural_pop_increase", loc_mapping)
  
  # Load and combine past/forecast median age
  past_med_age = as.data.table(read.xlsx(MEDIAN_AGE_FILE, sheet = "ESTIMATES", startRow = 17, colNames = TRUE, na.strings = "…"))
  past_med_age = past_med_age[,-c("Index", "Variant", "Region,.subregion,.country.or.area.*", "Notes", "Type", "Parent.code")]
  past_med_age = melt(past_med_age, id.vars = c("Country.code"), variable.name = "year_id", value.name = "median_age")
  forecast_med_age = as.data.table(read.xlsx(MEDIAN_AGE_FILE, sheet = "MEDIUM VARIANT", startRow = 17, colNames = TRUE, na.strings = "…"))
  forecast_med_age = forecast_med_age[,-c("Index", "Variant", "Region,.subregion,.country.or.area.*", "Notes", "Type", "Parent.code")]
  forecast_med_age = melt(forecast_med_age, id.vars = c("Country.code"), variable.name = "year_id", value.name = "median_age")
  med_age = rbind(past_med_age, forecast_med_age)
  rm(past_med_age); rm(forecast_med_age)
  med_age = wrangle_wpp_data(med_age, "median_age", loc_mapping)
  
  # Function to read and combine past/forecast shocks
  read_shock <- function(past_path, forecast_path, data_column_name){
    past_shock = fread(past_path)
    past_shock = past_shock[,c("location_id", "year_id", "mortality")]
    forecast_shock = fread(DISASTER_FORECAST_PATH)
    forecast_shock = forecast_shock[scenario==0,]
    forecast_shock = forecast_shock[,c("location_id", "year_id", "mortality")]
    forecast_shock = forecast_shock[year_id >= FORECAST_START,]
    past_shock = past_shock[year_id < FORECAST_START,]
    shock = rbind(past_shock, forecast_shock)
    names(shock)[names(shock) == "mortality"] = data_column_name
    return(shock)
  }
  
  # Read shocks
  disaster_dt = read_shock(DISASTER_PAST_PATH, DISASTER_FORECAST_PATH, "disaster")
  execution_dt = read_shock(EXECUTION_PAST_PATH, EXECUTION_FORECAST_PATH, "execution")
  terror_dt = read_shock(TERROR_PAST_PATH, TERROR_FORECAST_PATH, "terror")
  
  # Read SDI
  sdi = fread(SDI_PATH)
  sdi = sdi[scenario==0,]
  sdi = sdi[,c("location_id", "year_id", "sdi")]
  
  # hold med_age and birth_death constant over single years
  five_year_interval <- function(wpp_data){
    wpp_data = wpp_data[rep(seq_len(nrow(wpp_data)), each=5),]
    wpp_data = wpp_data[order(location_id, year_id)]
    wpp_data$year2 = rep(1950:2104, times=length(unique(wpp_data$location_id)))
    wpp_data = wpp_data[,-c("year_id")]
    names(wpp_data)[names(wpp_data) == "year2"] = "year_id"
    return(wpp_data)
  }
  
  mig_rate = five_year_interval(mig_rate)
  med_age = five_year_interval(med_age)
  nat_pop = five_year_interval(nat_pop)
  
  # merge
  mig_covs_pred = merge(sdi, disaster_dt, by=c("location_id", "year_id"))
  mig_covs_pred = merge(mig_covs_pred, execution_dt, by=c("location_id", "year_id"))
  mig_covs_pred = merge(mig_covs_pred, terror_dt, by=c("location_id", "year_id"))
  mig_covs_pred = merge(mig_covs_pred, med_age, by=c("location_id", "year_id"))
  mig_covs_pred = merge(mig_covs_pred, nat_pop, by=c("location_id", "location_name", "year_id"))
  mig_covs_pred = merge(mig_covs_pred, mig_rate, by=c("location_id", "location_name", "year_id"))
  
  # Add total shocks column
  mig_covs_pred$shocks = mig_covs_pred$disaster + mig_covs_pred$execution + mig_covs_pred$terror
  
  # Create a binary shocks variable if greater than certain arbitrary value
  # Not currently used for modeling
  mig_covs_pred$shocks_bin = mig_covs_pred$shocks > 5*10^-5
  
  fwrite(mig_covs_pred, sprintf("%scovariates_single_years.csv", DATA_DIR))
}

#----Model 6---------------------------
summary(mig_model_6 <- glm(migration_rate ~ sdi + natural_pop_increase + shocks, data = mig_covs_pred[year_id<FORECAST_START,]))
mig_covs_pred$predictions <- predict(mig_model_6, data.frame(mig_covs_pred))
fwrite(mig_covs_pred, sprintf("%smodel_6_single_years.csv", DATA_DIR))
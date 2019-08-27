####################################################################################################
## Author:      Thomas Hsiao
##
## Description: Prep UN Urbanicity Estimates (World Urbanization Prospects 2018)
####################################################################################################

rm(list=ls())
root <- ifelse(Sys.info()[1]=="Windows", "J:/", "/home/j/")

library(data.table)
library(readxl, lib.loc = paste0(root, "/WORK/02_fertilitypop/r_packages"))

shared_functions_dir <- paste0(root, "temp/central_comp/libraries/current/r/")
source(paste0(shared_functions_dir, "get_location_metadata.R"))
location_hierarchy = get_location_metadata(location_set_id=21, gbd_round_id=5)

data_dir <- "/homes/twh42/wup2018_prop_urban.xls"
loc_mapping_dir <- paste0(root, "DATA/IHME_COUNTRY_CODES/IHME_COUNTRY_CODES_Y2013M07D26.CSV")
un_years <- as.character(seq(1950, 2050, 5))
un_value_name <- "value"
output_dir <- "/ihme/forecasting/data/5/past/prop_urban/wup2018.csv"
output_dir_interpolate <- "/ihme/forecasting/data/5/past/prop_urban/wup2018_interpolated.csv"

# read in data
wup_data <- data.table(read_excel(path = data_dir, skip = 16))
names(wup_data) <- c("index", "un_location_name", "notes", "iso_num", un_years)

# merge on ihme_loc_id
loc_mapping <- fread(loc_mapping_dir)
loc_mapping <- merge(loc_mapping, location_hierarchy, by = "location_id")
loc_mapping <- loc_mapping[, list(ihme_loc_id, iso_num)]
wup_data <- merge(wup_data, loc_mapping, by = "iso_num", all.x = T)
wup_data <- wup_data[!is.na(ihme_loc_id)] # get rid of regions and locations we don't estimate
#wup_data[ihme_loc_id == "CHN", ihme_loc_id := "CHN_44533"]

wup_data <- melt(wup_data, id.vars = c("ihme_loc_id"), measure.vars = un_years,
                 variable.name = "year_id", value.name = un_value_name, variable.factor = F)
#wpp_data[, year_id := as.numeric(year_id) + 2.5]

wup_data <- wup_data[, list(ihme_loc_id, year_id, value)]
setkeyv(wup_data, c("ihme_loc_id", "year_id"))
readr::write_csv(wup_data, output_dir)

# Interpolate linearly between 5-yr period intervals
setnames(wup_data, "year_id", "year")
wup_data[, year := as.numeric(year)]
wup_data <- wup_data[rep(1:.N, times = 5)]
wup_data[, year_id := year + 0:(.N-1), by = .(ihme_loc_id, year)]
wup_data <- wup_data[year_id <= 2050]
setkeyv(wup_data, c("ihme_loc_id", "year_id"))
wup_data[year_id %% 5 != 0 , value := NA]
wup_data[, value := approx(year_id, value, xout = 1950:2050)$y, by = .(ihme_loc_id)]
wup_data[, year := NULL]
setnames(wup_data, "value", "prop_urban")
wup_data <- merge(wup_data, get_locations(gbd_year = 2017, level = "country")[,.(location_id, ihme_loc_id)],by = "ihme_loc_id")
readr::write_csv(wup_data, output_dir_interpolate)

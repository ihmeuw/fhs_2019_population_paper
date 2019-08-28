library(data.table)
library(tidyverse)

source("/ihme/homes/egoren/repos/fertility_working/fbd_research/fbd_research/fertility/ccf/xarray_to_r.R")

WPP_PATH <- "/ihme/homes/egoren/repos/fertility_working/fbd_research/fbd_research/fertility/ccf/bayesTFR"
WPP_RERUN_VERSION <- "20190826_WPP2019"
WPP_NEWP3_VERSION <- "20190826_WPP2019_custom_phaseIII"

# Read in data and set init to 2010-2015/2018 and final to 2095-2100

# new phase 3 predictions
res_newIII <- fread(sprintf("%s/%s/output/predictions/projection_summary_user_friendly.csv", WPP_PATH, WPP_NEWP3_VERSION))
res_newIII_final <- res_newIII$`2095-2100`
res_newIII_init <- res_newIII$`2015-2020`

# WPP2019 forecasts (not reestimated)
res_wpp_long <- xarray_nc_to_R("/share/forecasting/data/wpp/future/tfr/2019/tfr.nc")
res_wpp_final <- subset(res_wpp_long, year_id == 2095)$value
res_wpp_init <- subset(res_wpp_long, year_id == 2015)$value
length(res_wpp_init)
unique(res_wpp_long$location_id)

wpp <- c(
    mean(res_wpp_final),
    mean(res_wpp_final[res_wpp_final < 2.1]),
    mean(res_wpp_final[res_wpp_init < 2.1]),
    mean(res_wpp_final[res_wpp_final < 2.1 & res_wpp_final > res_wpp_init]),
    mean(res_wpp_final[res_wpp_init < 2.1 & res_wpp_final > res_wpp_init])
)
wpp_new_phaseIII <- c(
    mean(res_newIII_final),
    mean(res_newIII_final[res_newIII_final < 2.1]),
    mean(res_newIII_final[res_newIII_init < 2.1]),
    mean(res_newIII_final[res_newIII_final < 2.1 & res_newIII_final > res_newIII_init]),
    mean(res_newIII_final[res_newIII_init < 2.1 & res_newIII_final > res_newIII_init])
)
labs <- c(
    "location-average in 2100",
    "Below replacement in 2095-2100: location-average in 2100",
    "Below replacement in 2015-2020: location-average in 2100",
    "Below replacement in 2095-2100: location-average in 2100 where TFR increased 2015-2100",
    "Below replacement in 2015-2020: location-average in 2100 where TFR increased 2015-2100"
)
all <- data.frame(
    TFR = labs,
    WPP2019 = round(wpp,2),
    WPP2019_new_phaseIII = round(wpp_new_phaseIII,2)
    )
knitr::kable(all)
fwrite(all, sprintf("%s/WPP2019_terminal_TFR.csv", WPP_PATH))

# locations in phase III
phase3_wpp2019 <- fread(sprintf("%s/%s/bayesTFR_phase3_locations.csv", WPP_PATH, WPP_RERUN_VERSION))
phase3_new_wpp2019 <- fread(sprintf("%s/%s/bayesTFR_phase3_locations.csv", WPP_PATH, WPP_NEWP3_VERSION))

# added to phase III by modification
added <- phase3_new_wpp2019$country_name[!(phase3_new_wpp2019$country_name %in% phase3_wpp2019$country_name)]

# removed from phase III by modification
removed <- phase3_wpp2019$country_name[!(phase3_wpp2019$country_name %in% phase3_new_wpp2019$country_name)]

paste0("WPP2019 Phase III Countries: #", nrow(phase3_wpp2019))
cat(sort(phase3_wpp2019$country_name), sep = ", ")

paste0("WPP2019 Phase III Countries Added with Modification: #", nrow(phase3_new_wpp2019))
cat(sort(added), sep = ", ")

# check that no countries were removed
removed

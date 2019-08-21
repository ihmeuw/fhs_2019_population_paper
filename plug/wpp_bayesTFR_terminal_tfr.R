library(data.table)
library(tidyverse)

# Guessing at file paths from Thomas's code
# wide and long new phase 3
path_newIII <- "/ihme/fertilitypop/fertility/gbd_2019/forecasting/20181012_tfr_new_phaseIII/predictions/projection_summary_user_friendly.csv"
path_newIII_long <- "/ihme/fertilitypop/fertility/gbd_2019/forecasting/data/tfr_raftery_newphaseIII.csv"
# wide and long wpp
path_wpp <- "/share/forecasting/data/wpp/future/tfr/2019/tfr.nc"
source("/ihme/homes/egoren/repos/fertility_working/fbd_research/fbd_research/fertility/ccf/xarray_to_r.R")

# Read in data and set init to 2010-2015/2018 and final to 2095-2100
res_newIII <- fread(path_newIII)
res_newIII$final <- res_newIII$`2095-2100`
res_newIII$init <- res_newIII$`2015-2020`

res_newIII_long <- fread(path_newIII_long)
res_newIII_final <- subset(res_newIII_long, year_id == 2098)$median
res_newIII_init <- subset(res_newIII_long, year_id == 2018)$median
length(res_newIII_init)
unique(res_newIII_long$ihme_loc_id)

res_wpp_long <- xarray_nc_to_R(path_wpp)
res_wpp_final <- subset(res_wpp_long, year_id == 2095)$value
res_wpp_init <- subset(res_wpp_long, year_id == 2015)$value
length(res_wpp_init)
unique(res_wpp_long$location_id)

WPP <- c(
    mean(res_wpp_final),
    mean(res_wpp_final[res_wpp_final < 2.1]),
    mean(res_wpp_final[res_wpp_init < 2.1]),
    mean(res_wpp_final[res_wpp_final < 2.1 & res_wpp_final > res_wpp_init]),
    mean(res_wpp_final[res_wpp_init < 2.1 & res_wpp_final > res_wpp_init])
)
WPP_new_phaseIII <- c(
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
    WPP2019 = round(WPP,2),
    WPP2017_new_phaseIII = round(WPP_new_phaseIII,2)
    )
knitr::kable(all)

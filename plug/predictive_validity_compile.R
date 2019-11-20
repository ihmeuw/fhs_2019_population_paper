# Compile predictive validity results
# The predictive validity results came from two runs (10 year hold out with our model,
# and another with WPP). This script just puts these results in a .csv that can be opened in
# Excel then copied and pasted into the appendix.

library(data.table)
library(tidyverse, lib.loc = "/ihme/forecasting/shared/r")



FBD_PATH <- "/ihme/forecasting/data/5/future/ccf"
FBD_VERSION <- "20190815_150243_fix_draw_bound_ccfx_pv_10yr"
WPP_PATH <- "/ihme/homes/egoren/repos/fertility_working/fbd_research/fbd_research/fertility/ccf/bayesTFR"
WPP_VERSION <- "20190826_WPP2019_fit_to_GBD2017_PV_10yr"

wpp_res <- fread(sprintf("%s/%s/predictive_validity_TFR.csv", WPP_PATH, WPP_VERSION))
wpp_res_stratified <- fread(sprintf("%s/%s/predictive_validity_TFR_stratified_by_replacement.csv", WPP_PATH, WPP_VERSION))

fbd_res <- fread(sprintf("%s/%s/predictive_validity_TFR_2008-2017.csv", FBD_PATH, FBD_VERSION))
fbd_res_stratified <- fread(sprintf("%s/%s/predictive_validity_TFR_stratified_by_replacement_2008-2017.csv", FBD_PATH, FBD_VERSION))

overall <- rbind(fbd_res, wpp_res)
overall$below_rpl <- "All"
stratified <- rbind(fbd_res_stratified, wpp_res_stratified)
out <- rbind(overall[,c(5,1:4)], stratified)
fwrite(out, "./predictive_validity.csv")

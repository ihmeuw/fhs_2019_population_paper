###############################################################################
# Author: Julian Chalek / Emily Goren
# Date: 7/30/2019
# edited by EMG 8/19/2019 to count divergent trends
# this code now is an EDA to compares location-year increases between TFR and CCF
# used for plugging in the fertility methods section
###############################################################################

rm(list=ls())
pacman::p_load(data.table, tidyverse, scales)
source("/ihme/cc_resources/libraries/current/r/get_covariate_estimates.R")
source("/ihme/cc_resources/libraries/current/r/get_location_metadata.R")
source("/share/forecasting/working/jchalek/repos/utilities/read_ncdf.R")

# Paths
FBD_DIR  <- "/ihme/forecasting/data/5/future"
FERT_VERS <- "20190806_141418_fix_draw_bound_ccfx_to2110"

# Pull and wrangle data
locs <- get_location_metadata(location_set_id = 35, gbd_round_id = 5)[level==3]

tfr_past <- get_covariate_estimates(covariate_id=149, gbd_round_id=5,
                                    location_id = locs$location_id,
                                    status = "best") %>%
                                    .[, .(year_id, location_id, mean_value)]

setnames(tfr_past, "mean_value", "value")

tfr_fut <- setDT(paste0(FBD_DIR, "/tfr/", FERT_VERS, "/tfr_agg_mean.nc") %>%
                 xarray_nc_to_R(dimname="value")) %>%
                 .[scenario==0 & year_id >= 2018 & year_id <= 2100 &
                   location_id %in% locs$location_id] %>%
                 .[, scenario := NULL]

tfr <- rbindlist(list(tfr_past, tfr_fut))
tfr <- tfr[, scenario := "Reference"]
tfr[, Indicator := "Total Fertility Rate"]

ccf <- fread(paste0(FBD_DIR, "/ccf/", FERT_VERS, "/complete/ccf_final_time_series.csv"))
ccf[, cohort := cohort + 30]
ccf <- ccf[cohort <= 2100 & scenario == 0] %>%
       .[, scenario := as.character(scenario)] %>%
       .[, scenario :="Reference"] %>%
       .[, c("lower", "upper") := NULL]
setnames(ccf, c("cohort", "ccf"), c("year_id", "value"))
ccf[, Indicator := "Completed Cohort Fertility"]

both <- rbindlist(list(tfr, ccf), fill = T, use.names = T)
both <- both[year_id %in% c(1950:2100)]
both <- merge(both, locs[, .(location_id, location_name)], by="location_id")

both[, Indicator := factor(Indicator, levels = c("Total Fertility Rate",
                                                     "Completed Cohort Fertility"))]

# In the period 1950 to 2017, the TFR increased XX country-years whereas the CCF50 increased only XX country-years
# only have back to 1965 for both
d <- both %>%
  subset(year_id %in% 1965:2017) %>%
  spread(Indicator, value)
d$TFR <- d$`Total Fertility Rate`
d$CCF <- d$`Completed Cohort Fertility`

d_change <- d %>%
  group_by(location_id, scenario, location_name) %>%
  summarise(delta_TFR = TFR[year_id == 2017] - TFR[year_id == 1965],
            delta_CCF = CCF[year_id == 2017] - CCF[year_id == 1965])

ggplot(d_change) +
  geom_point(aes(x = delta_CCF, y = delta_TFR)) +
  geom_abline(intercept = 0, slope = 1) +
  labs(x = "Change in CCF (1965 to 2017)",
       y = "Change in TFR (1965 to 2017")

# Proportion where TFR had greater decrease than CCF
mean(abs(d_change$delta_TFR) > abs(d_change$delta_CCF))
# average change in TFR
mean(d_change$delta_TFR)
# average change in CCF
mean(d_change$delta_CCF)

# Look at annual changes in both
d_annual <- d %>%
  group_by(location_id, scenario, location_name) %>%
  arrange(year_id, .by_group = TRUE) %>%
  mutate(lag_CCF = lag(CCF),
         lag_TFR = lag(TFR),
         diff_CCF = CCF - lag_CCF,
         diff_TFR = TFR - lag(TFR),
         TFRbelow2.1 = TFR < 2.1,
         CCFbelow2.1 = CCF < 2.1
         )

diff_props <- d_annual %>%
  filter(year_id > 1965) %>%
  group_by(location_id, scenario, location_name) %>%
  summarise(prop_decrease_CCF = mean(diff_CCF < 0),
            prop_decrease_TFR = mean(diff_TFR < 0),
            ratio_decrease = prop_decrease_TFR/prop_decrease_CCF) %>%
  arrange(ratio_decrease)
diff_props


ggplot(diff_props) +
  geom_point(aes(x = prop_decrease_CCF, y = prop_decrease_TFR)) +
  geom_abline(intercept = 0, slope = 1) +
  labs(x = "Proportion of years with annual decrease in CCF (1966 to 2017)",
       y = "Proportion of years with annual decrease in TFR (1966 to 2017)")

# subset on CCF < 2.1
diff_props_ccfreplacement <- d_annual %>%
  filter(year_id > 1965 ) %>%
  group_by(location_id, scenario, location_name, CCFbelow2.1) %>%
  summarise(prop_decrease_CCF = mean(diff_CCF < 0),
            prop_decrease_TFR = mean(diff_TFR < 0),
            ratio_decrease = prop_decrease_TFR/prop_decrease_CCF) %>%
  arrange(ratio_decrease)
diff_props_ccfreplacement



# subset on TFR < 2.1
diff_props_tfrreplacement <- d_annual %>%
  filter(year_id > 1965 ) %>%
  group_by(location_id, scenario, location_name, TFRbelow2.1) %>%
  summarise(prop_decrease_CCF = mean(diff_CCF < 0),
            prop_decrease_TFR = mean(diff_TFR < 0),
            ratio_decrease = prop_decrease_TFR/prop_decrease_CCF) %>%
  arrange(ratio_decrease)
diff_props_tfrreplacement

diff_props %>%
  group_by(scenario) %>%
  summarise(average_prop_decrease_CCF = mean(prop_decrease_CCF),
            average_prop_decrease_TFR = mean(prop_decrease_TFR))
diff_props_ccfreplacement %>%
  group_by(scenario, CCFbelow2.1) %>%
  summarise(average_prop_decrease_CCF = mean(prop_decrease_CCF),
            average_prop_decrease_TFR = mean(prop_decrease_TFR))
diff_props_tfrreplacement %>%
  group_by(scenario, TFRbelow2.1) %>%
  summarise(average_prop_decrease_CCF = mean(prop_decrease_CCF),
            average_prop_decrease_TFR = mean(prop_decrease_TFR))

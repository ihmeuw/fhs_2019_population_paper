###############################################################################
# Author: Julian Chalek
# Date: 7/30/2019
# Goal: CCF/TFR time series panel plots
###############################################################################

rm(list=ls())
pacman::p_load(data.table, magrittr, ggplot2, scales)
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
                 .[scenario==0 & year_id >= 2018 & year_id <= 2017 &
                   location_id %in% locs$location_id] %>%
                 .[, scenario := NULL]

tfr <- rbindlist(list(tfr_past, tfr_fut))
tfr <- tfr[, scenario := "Reference"]
tfr[, Indicator := "Total Fertility Rate"]

ccf <- fread(paste0(FBD_DIR, "/ccf/", FERT_VERS, "/complete/ccf_final_time_series.csv"))
ccf[, cohort := cohort + 30]
ccf <- ccf[cohort <= 2017 & scenario == 0] %>%
       .[, scenario := as.character(scenario)] %>%
       .[, scenario :="Reference"] %>%
       .[, c("lower", "upper") := NULL]
setnames(ccf, c("cohort", "ccf"), c("year_id", "value"))
ccf[, Indicator := "Completed Cohort Fertility"]

both <- rbindlist(list(tfr, ccf), fill = T, use.names = T)
both <- both[year_id %in% c(1950:2017)]
both <- merge(both, locs[, .(location_id, location_name)], by="location_id")

# both_sub <- both[location_name %in% c("China", "Nigeria", "India", "Greece", "Denmark")]
both_sub <- both[location_name %in% c("United States", "Czech Republic", "Russian Federation", "Georgia")]
both_sub[, Indicator := factor(Indicator, levels = c("Total Fertility Rate",
                                                     "Completed Cohort Fertility"))]

# Plot

cols = c("Completed Cohort Fertility" = "steelblue", "Total Fertility Rate" = "black")

cairo_pdf("/ihme/forecasting/working/jchalek/plots/20180821_ccf_tfr_timeseries_country_test.pdf", width=14, height=8.5)
ggplot(both_sub, aes(x=year_id, y=value, color=Indicator)) + 
  geom_line(size=1) +
  scale_color_manual(values = cols) +
  theme_bw() + 
  ggtitle("CCF and TFR") + 
  ylab("Rate") + 
  xlab("Year") + 
  scale_x_continuous(breaks = pretty_breaks(8)) +
  expand_limits(y=0) + 
  facet_wrap(~location_name, nrow = 1) +
  guides(color = guide_legend(reverse = T)) +
  theme(axis.text=element_text(size=10),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title=element_text(size=12),
        strip.background = element_blank(),
        strip.text = element_text(size = 12),
        panel.border = element_rect(color = "black"),
        panel.spacing.x=unit(1, "lines"))
dev.off()

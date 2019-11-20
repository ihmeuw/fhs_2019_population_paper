###############################################################################
# Author: Julian Chalek
# Date: 7/30/2019
# Goal: CCF/TFR time series panel plots
###############################################################################

rm(list=ls())
pacman::p_load(data.table, magrittr, ggplot2, scales, foreach)
source("/ihme/cc_resources/libraries/current/r/get_covariate_estimates.R")
source("/ihme/cc_resources/libraries/current/r/get_location_metadata.R")
source("/share/forecasting/working/jchalek/repos/utilities/read_ncdf.R")

# Globals
FBD_DIR  <- "/ihme/forecasting/data/5/future"
BASE <- "20190806_141418_fix_draw_bound_ccfx_to2110"
IHME_NAME <- "tfr_agg_mean.nc"
WPP_DIR  <- "/ihme/forecasting/data/wpp/future"
WPP <- 2019
WPP_NAME <- "tfr.nc"
LOCS <- get_location_metadata(location_set_id = 35, gbd_round_id = 5)
OUTPATH <- "/ihme/forecasting/working/jchalek/plots/20180816_tfr_scatter_ihme_wpp.pdf"

# Functions
pull_tfr <- function(directory, version, filename, scen) {
  if (scen == "WPP2019") {
    tfr <- setDT(paste0(directory, "/tfr/", version, "/", filename) %>%
                 xarray_nc_to_R(dimname="value")) %>% 
                 .[year_id == 2095 & location_id %in% LOCS$location_id] %>%
                 .[, scenario := scen] %>% .[, year_id := 2100]
  } else {
    tfr <- setDT(paste0(directory, "/tfr/", version, "/", filename) %>%
                 xarray_nc_to_R(dimname="value")) %>% 
                 .[year_id == 2100 & scenario == 0 &
                   location_id %in% LOCS$location_id] %>%
                 .[, scenario := as.character(scenario)] %>%
                 .[, scenario := scen]
  }
  return(tfr)
}


# Pull and wrangle data

ihme_tfr <- pull_tfr(FBD_DIR, BASE, IHME_NAME, "Reference")
wpp_tfr <- pull_tfr(WPP_DIR, WPP, WPP_NAME, "WPP2019")
both2100 <- rbindlist(list(ihme_tfr, wpp_tfr), use.names = TRUE)
tfr_wide <- dcast(both2100, location_id + year_id  ~ scenario, value.var = "value")
tfr_wide <- merge(tfr_wide, LOCS[, .(location_id, location_ascii_name,
                                     ihme_loc_id, super_region_name)], by="location_id")
tfr_wide <- tfr_wide[location_ascii_name != "Global" & !is.na(WPP2019)]


# Plot

tfr_wpp_scatter <- function(df){
  cairo_pdf(OUTPATH, width=12.5, height=8.5)
  gg <- ggplot(df, aes(x = WPP2019, y = Reference, color = super_region_name, label = ihme_loc_id)) +
    geom_text(show.legend = F) + 
    geom_point(alpha = 0) +
    guides(color = guide_legend(override.aes = list(alpha = 1))) +
    geom_abline() + 
    coord_cartesian(xlim = c(1,2.75), ylim = c(1,2.75)) + 
    theme_bw() + 
    xlab("UNPD") + 
    ylab("IHME") + 
    labs(color="Super Region") +
    theme(axis.text=element_text(size=12),
          axis.title=element_text(size=14,face="bold")) + 
    ggtitle(sprintf("%s Comparison", 2100))
  print(gg)
  dev.off()
}

tfr_wpp_scatter(tfr_wide)

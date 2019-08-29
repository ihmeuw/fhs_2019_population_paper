###############################################################################
# Author: Julian Chalek
# Date: 7/30/2019
# Goal: TFR time series panel plots
###############################################################################

rm(list=ls())
pacman::p_load(data.table, magrittr, ggplot2, scales, foreach)
source("/ihme/cc_resources/libraries/current/r/get_covariate_estimates.R")
source("/ihme/cc_resources/libraries/current/r/get_location_metadata.R")
source("/share/forecasting/working/jchalek/repos/utilities/read_ncdf.R")

# Globals
FBD_DIR  <- "/ihme/forecasting/data/5/future"
BASE <- "20190806_141418_fix_draw_bound_ccfx_to2110_combined"
FASTEST <- "20190807_163915_fix_draw_bound_ccfx_99th_to2110_combined"
SDG <- "20190807_164000_fix_draw_bound_ccfx_sdg_to2110_scen_swapped_combined"
SCENARIOS <- c("Slower Met Need and Education Pace","Reference",
               "Faster Met Need and Education Pace",
               "Fastest Met Need and Education Pace",
               "SDG Met Need and Education Pace")
LOC_IDS <- c(6, 163, 102, 11, 165, 135, 214, 161, 171, 179)
# China, India, USA, Indonesia, Pakistan,
# Brazil, Nigeria, Bangladesh, DR Congo, Ethiopia
OUTPATH <- "/ihme/forecasting/working/jchalek/plots/20180827_tfr_panels.pdf"


# Functions
pull_tfr_future <- function() {
  tfr <- foreach(scen = c(BASE, FASTEST, SDG), .combine = "rbind") %do% {
    
    df <- setDT(paste0(FBD_DIR, "/tfr/", scen, "/tfr_combined.nc") %>%
                xarray_nc_to_R(dimname="value")) %>% 
                .[year_id >= 2018 & year_id <= 2100 &
                location_id %in% LOC_IDS] %>%
                .[, scenario := as.character(scenario)]
    
    if (scen == FASTEST) {
      df <- df[scenario == "1"] %>%
            .[, scenario := SCENARIOS[4]]
    } else if (scen == SDG) {
      df <- df[scenario == "-1"] %>%
        .[, scenario := SCENARIOS[5]]
    } else {
      df[scenario == "-1", scenario := SCENARIOS[1]]
      df[scenario == "0", scenario := SCENARIOS[2]]
      df[scenario == "1", scenario := SCENARIOS[3]]
    }
  }
  return(tfr)
}

pull_tfr_time_series <- function(){
  tfr_past <- get_covariate_estimates(covariate_id=149, gbd_round_id=5,
                                      location_id = LOC_IDS,
                                      status = "best") %>%
                                      .[, .(year_id, location_id, mean_value)] %>%
                                      .[, scenario := "Reference"]
  setnames(tfr_past, "mean_value", "value")
  tfr_fut <- pull_tfr_future()
  
  tfr_fut_ref <- tfr_fut[scenario == "Reference"]
  tfr_fut_ref <- dcast(tfr_fut_ref, year_id + location_id + scenario ~ quantile, value.var = "value")
  setnames(tfr_fut_ref, "mean", "value")
  tfr_fut_other <- tfr_fut[scenario != "Reference"]
  
  tfr <- rbindlist(list(tfr_past, tfr_fut_ref, tfr_fut_other),
                   use.names = T, fill = T)
  tfr <- tfr[quantile == "mean" | is.na(quantile)]
  tfr[, quantile := NULL]
  return(tfr)
}

make_2017_scenarios <- function(tfr){
  tfr_ref_17 <- tfr[year_id==2017]
  tfr_scens_17 <- foreach(scen = SCENARIOS[SCENARIOS != "Reference"],
                          .combine = "rbind") %do% {
    df <- copy(tfr_ref_17) %>% .[, scenario := scen]
  }
  return(rbindlist(list(tfr, tfr_scens_17)))
}

make_panel_plots <- function(tfr) {
  cols <- c("Reference" = "steelblue",
            "Slower Met Need and Education Pace" = "firebrick",
            "Faster Met Need and Education Pace" = "forestgreen",
            "Fastest Met Need and Education Pace" = "#984ea3",
            "SDG Met Need and Education Pace" = "#ff7f00")
  
  cairo_pdf(OUTPATH, width=17, height=11)
  plot <- ggplot(tfr, aes(x=year_id, y=value, color=scenario, fill=scenario)) + 
    geom_ribbon(aes(ymin=lower, ymax=upper), alpha=.4, color = NA) +
    geom_line(size=1) +
    geom_vline(xintercept = 2017, linetype="dashed", color = "black") +
    scale_fill_manual(values = cols) +
    scale_color_manual(values = cols) +
    theme_bw() + 
    ggtitle("Title here") + 
    ylab("Total Fertility Rate") + 
    xlab("Year") + 
    scale_x_continuous(breaks = seq(1950, 2100, 25)) +
    facet_wrap(~location_ascii_name, nrow = 2, scales = "free_x") +
    labs(color="Scenario") +
    guides(fill=FALSE) +
    theme(axis.text=element_text(size=11),
          axis.text.x = element_text(angle = 45, hjust = 1),
          axis.title=element_text(size=12),
          strip.background = element_blank(),
          strip.text = element_text(size = 12),
          panel.border = element_rect(color = "black"),
          panel.spacing.x=unit(1, "lines"),
          panel.spacing.y=unit(1, "lines"))
  print(plot)
  dev.off()
}

main <- function() {
  # Pull and wrangle data
  locs <- get_location_metadata(location_set_id = 35, gbd_round_id = 5)
  tfr <- pull_tfr_time_series()
  tfr <- make_2017_scenarios(tfr)
  tfr[year_id==2017 & scenario=="Reference", c("lower", "upper") := value]
  tfr[, scenario := factor(scenario, levels = SCENARIOS)]
  tfr <- merge(tfr, locs[, .(location_id, location_ascii_name)], by="location_id")
  tfr[, location_id := factor(location_id, levels = LOC_IDS)]
  tfr[, sort_order := as.numeric(location_id)]
  setorder(tfr, sort_order)
  tfr[, location_ascii_name := factor(location_ascii_name,
                                      levels = unique(location_ascii_name))]
  # Plot
  make_panel_plots(tfr)
}

# Run
main()
rm(list=ls())
pacman::p_load(data.table, magrittr, ggplot2, foreach)
source("/share/forecasting/working/jchalek/repos/utilities/read_ncdf.R")
library(mortdb, lib = "/home/j/WORK/02_mortality/shared/r")
source("/ihme/cc_resources/libraries/current/r/get_covariate_estimates.R")
source("/ihme/cc_resources/libraries/current/r/get_location_metadata.R")
source("/home/j/DATA/SHAPE_FILES/GBD_geographies/master/GBD_2016/inset_maps/noSubs/GBD_WITH_INSETS_MAPPING_FUNCTION.R")

# Paths
FBD_DIR  <- "/ihme/forecasting/data/5/future"
FERT_VERS <- "20190806_141418_fix_draw_bound_ccfx_to2110"
LTABLE_VERS <- "20190808_15_ref_85_agg"

FERT_AGE_GROUPS <- c(7:15)

get_first_of_series <- function(vec){
  y = numeric()
  for (i in 1:length(vec)) {
    if (i==1) {
      if (vec[i] == 1) {y[i] <- 1}
      else {y[i] <- 0}
    }
    else if (i>1) {
      if (vec[i] == 1 & vec[i-1] == 0) {y[i] <- 1}
      else {y[i] <- 0}
    }
  }
  return(y)
}

makeLimits <- function(start, end) {
  lims <- seq(start, end, 10)
  for (i in 1:length(lims)) {
    lims[i] <- ifelse((i %% 2 == 0) & (i != length(lims) - 1), lims[i] - 0.0000001, lims[i])
    if (i == length(lims) - 1) {
      lims[i] <- lims[i] + 0.9999999
    }
  }
  lims[1] <- -100
  lims[length(lims)] <- 3000
  return(lims)
}

prettyRound <- function(x, y) sprintf(paste0("%.", y, "f"), round(x, y)) ##prettyRound

makeLabs <- function (limits, roundto, percent = FALSE) { ##makeLabs
  labs <- paste0("Before ", prettyRound(limits[2], roundto))
  for (i in seq(2, length(limits) - 2)) {
    labs <- c(labs, paste0(prettyRound(limits[i], roundto),
                           " to ",
                           prettyRound(limits[i + 1] - 1, roundto)))
  }
  labs <- c(labs, paste0(prettyRound(limits[length(limits) - 1], roundto), " onward"))
  if (percent) labs <- paste0(labs, "%")
  return(labs)
}

# Pull and wrangle data
locs <- get_location_metadata(location_set_id = 35, gbd_round_id = 5)[level==3]

# ASFR
asfr_past <- get_covariate_estimates(covariate_id=13, gbd_round_id=5,
                                     age_group_id = FERT_AGE_GROUPS,
                                     location_id = locs$location_id,
                                     sex_id = 2, status = "best") %>%
                                     .[, .(age_group_id, year_id, location_id, mean_value)]

setnames(asfr_past, "mean_value", "value")

asfr_fut <- setDT(paste0(FBD_DIR, "/asfr/", FERT_VERS, "/asfr_agg_mean.nc") %>%
                  xarray_nc_to_R(dimname="value")) %>% 
                  .[scenario==0 & year_id >= 2018 & location_id %in% locs$location_id] %>%
                  .[, scenario := NULL]

asfr <- rbindlist(list(asfr_past, asfr_fut))

setnames(asfr, "value", "asfr")

# nLx
nlx_fut <- setDT(
  paste0(
    FBD_DIR, "/life_expectancy/",
    LTABLE_VERS, "/lifetable_ds_agg_nLx_only_means.nc") %>%
    xarray_nc_to_R(dimname="nLx")) %>% 
    .[sex_id==2 & scenario==0 & age_group_id %in% c(7:15)] %>%
    .[, scenario := NULL]

setnames(nlx_fut, "value", "mean")

nlx_past <- get_mort_outputs("with shock life table", "estimate", run_id = "best", life_table_parameter_ids = 7,
                             age_group_ids = c(7:15), sex_ids = 2, location_ids = locs$location_id, gbd_year = 2017,
                             year_ids = c(1950:2017))[, .(age_group_id, year_id, location_id, sex_id, mean)]

nlx <- rbindlist(list(nlx_past, nlx_fut)) %>% .[, sex_id := NULL]

setnames(nlx, "mean", "nlx")

# prop female at birth
srb_past <- get_mort_outputs("birth_sex_ratio", "estimate",  gbd_year = 2017,
                             location_ids = locs$location_id)[, .(year_id, location_id, mean)]
srb_fut <- foreach(i = c(2018:2100), .combine = "rbind") %do% {
  sub <- srb_past[year_id==2017] %>% .[, year_id := i]
}
srb <- rbindlist(list(srb_past, srb_fut))
srb[, prop_fem := (1/mean)/(1+(1/mean))] %>% .[, mean := NULL]

# calculate net reproductive rate
nrr <- merge(asfr, nlx, by=c("location_id", "year_id", "age_group_id")) %>%
       merge(srb, by=c("location_id", "year_id"))
nrr[, nrr := sum(asfr * nlx * prop_fem), by =.(location_id, year_id)] ############## 1e5?
nrr_only <- copy(nrr)[, c("age_group_id", "nlx", "prop_fem", "asfr") := NULL] %>% unique()

# find first year each location drops below replacement
frst_yr_below_repl <- copy(nrr_only)[, below1 := ifelse(nrr<1, 1, 0)]
frst_yr_below_repl[, first_of_series := get_first_of_series(below1), by="location_id"]
frst_yr_below_repl[, yr_of_drop := ifelse(first_of_series == 1, year_id, 0)]
frst_yr_below_repl[, mapvar := max(yr_of_drop), by="location_id"]
frst_yr_below_repl <- frst_yr_below_repl[, .(location_id, mapvar)] %>% unique()
frst_yr_below_repl[, mapvar := ifelse(mapvar==0, 2101, mapvar)]

# mapping
mapdf <- merge(frst_yr_below_repl, locs[, .(location_id, ihme_loc_id)])

# pre-mapping steps

# deciles <- makeLimits(mapdf[mapvar <= 2100]$mapvar, 10)

lims <- makeLimits(1990, 2110)
labs <- makeLabs(lims, 0)

pdf("/ihme/forecasting/working/jchalek/plots/20180820_replacement_map.pdf", width=14, height=8.5)
gbd_map(data = mapdf, 
        limits = lims, 
        labels = labs,
        col="Spectral", 
        col.reverse=TRUE, 
        na.color = "white", 
        title=paste0("Year NRR drops below replacement"),
        legend.columns = 2, 
        legend.cex=1, legend.shift=c(0,0), legend.title = "Years")
dev.off()


################
## diagnostics

tfr_past <- get_covariate_estimates(covariate_id=149, gbd_round_id=5,
                                     location_id = locs$location_id,
                                     status = "best") %>%
                                     .[, .(year_id, location_id, mean_value)]

setnames(tfr_past, "mean_value", "value")

tfr_fut <- setDT(paste0(FBD_DIR, "/tfr/", FERT_VERS, "/tfr_agg_mean.nc") %>%
                 xarray_nc_to_R(dimname="value")) %>% 
                 .[scenario==0 & year_id >= 2018 & location_id %in% locs$location_id] %>%
                 .[, scenario := NULL]

tfr <- rbindlist(list(tfr_past, tfr_fut))

tfr[, tfr_ovr_2.1 := value/2.1]

tfr2nrr <- merge(tfr, nrr_only, by=c("location_id", "year_id"))
setnames(tfr2nrr, "nrr", "NRR")
tfr2nrr <- merge(tfr2nrr, locs[, .(location_id, ihme_loc_id, super_region_name, location_ascii_name)],
                 by="location_id")

timeseries <- function(df){
  
  pdf("/ihme/forecasting/working/jchalek/plots/20180813_nrr_tfr_ovr_2.1_timeseries.pdf", width=15, height=5)
  
  for (loc in unique(df$location_ascii_name)) {
    
    plot <- ggplot(df[location_ascii_name==loc], aes(x=year_id)) +
      geom_line(aes(y = NRR, color="NRR")) +
      geom_line(aes(y = tfr_ovr_2.1, color="TFR/2.1")) +
      geom_hline(yintercept = 1, linetype="dashed", color = "grey") +
      ggtitle(paste0("NRR and TFR/2.1 in ", loc)) +
      theme_bw() +
      ylab("Mean") +
      xlab("Year") +
      scale_x_continuous(breaks = seq(1950, 2100, 2)) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(color="Scenario")
    print(plot)
    print(loc)
  }
  dev.off()
}

timeseries(tfr2nrr)

scatter <- function(df){
  
  pdf("/ihme/forecasting/working/jchalek/plots/20180813_nrr_tfr_ovr_2.1_scatters.pdf", width=9, height=5)
  
  for(year in c(1990, 2010, 2030, 2100)){
    gg <- ggplot(df[year_id==year], aes(x = tfr_ovr_2.1, y = NRR, color = super_region_name)) + 
      geom_point() +
      geom_abline() + 
      coord_cartesian(xlim = c(0,4), ylim = c(0,4)) + 
      theme_bw() + 
      xlab("TFR/2.1") + 
      ylab("NRR") + 
      labs(color="Super Region") +
      theme(axis.text=element_text(size=12),
            axis.title=element_text(size=14,face="bold")) + 
      ggtitle(paste0("National NRR vs TFR/2.1 in ", year))
    print(gg)
  }
  dev.off()
}

scatter(tfr2nrr)

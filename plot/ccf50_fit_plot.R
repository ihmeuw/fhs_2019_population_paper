#' Evaluate model fit for mean CCF50 forecasting
#'#'
#' This file reads in a completed cohort fertility file of \code{VERSION}
#'and  plots a fit of the following linear regression model:
#' \eqn{E(\text{CCF}_{cl}) = \beta_0 + ns(\text{EDU}_{cl}) + \beta_{mn}\text{MN}_{cl}},
#' where
#' \eqn{\text{CCF}_{cl}} is CCF50 for location \eqn{l} and birth cohort \eqn{c},
#' \eqn{\text{EDU}_{cl}} is the mean years of education at age group 10 for location \eqn{l} and birth cohort \eqn{c},
#' \eqn{\text{MN}_{cl}} is the proportion of met need for contraception at age group 10 for location \eqn{l} and birth cohort \eqn{c},
#' and  \eqn{ns(\text{EDU}_{cl})} is a natural spline with knots at \eqn{(3.1, 8.14, 12.8)} (from MR-BRT).
#'
#' @param version_dir The directory of the version of the fertility run.
#' @param version_plt_dir The directory of the version of the fertility run for saving plots.
#' @param gbd_year Year of the GBD round. Only used to pull super region location names. defaults to \code{2017}.
#' @param plot_past_data. Logical. Should the past data be added to the fit plot? Defaults to \code{TRUE}.
#' @author Emily Goren, \email{egoren@uw.edu}
#' @returns \code{NULL}
#'


CCF.fit.plot <- function(version_dir, version_plot_dir, gbd_year = 2017, plot_past_data = TRUE) {

  library(mortdb, lib.loc = "/ihme/mortality/shared/r/")
  library(tidyverse, lib.loc = "/ihme/forecasting/shared/r")
  library(ggpubr, lib.loc = "/ihme/forecasting/shared/r")
  library(splines)
  #   install.packages("ggnewscale", lib = "/ihme/forecasting/shared/r")
  library(ggnewscale, lib.loc = "/ihme/forecasting/shared/r")
  theme_set(theme_bw(base_size = 16))

  # Read in CCF50 data with covariates
  dat_ccf50 <- read.csv(paste0(version_dir, "/complete/ccf_input_data_w_covariates.csv"))
  # Read in age and location info from mortDB
  locs    <- mortdb::get_locations(gbd_year = gbd_year, level = "country")
  # Merge in location info
  dat_ccf50 <- left_join(dat_ccf50, locs, by = 'location_id')

  # Remove earlier cohorts with no met need data
  d <- filter(dat_ccf50, !is.na(mn))

  # Fit normal-space spline
  #fit <- lm(ccf ~ ns(edu, knots = c(3.1, 8.14, 12.8)) + mn, data = d)
  model <- readRDS(sprintf("%s/complete/complete_model.rds", version_dir))
  fit <- model

  # Points at which to plot model fit
  plot_dat <- expand.grid(mn = seq(0, 1, by = 0.1), edu = seq(0.01, 17.99, by = 0.01))
  plot_dat$ccf <- predict(fit, newdata = plot_dat)


  # Fit plot with data
  if (plot_past_data) {
  gg <- ggplot() +
    geom_point(data = d,
               mapping = aes(x = edu, y = ccf, color = super_region_name), size = 2, alpha = I(0.6)) +
    scale_color_brewer(palette = 'Dark2') +
    labs(color = 'Super\nRegion') +
    theme(legend.position = "bottom") +
    new_scale_color() +
    geom_line(data = plot_dat, mapping = aes(x = edu, y = ccf, group = mn, color = mn), size = 2) +
    scale_color_gradient(low = "gray90", high = "black") +
    labs(color = 'Contraceptive\nMet Need') +
    labs(x = 'Education (years)', y = 'Completed Cohort Fertility (CCF50)') +
    scale_x_continuous(breaks = 0:18, limits = c(0,18)) +
    scale_y_continuous(breaks = 1:8, limits = c(0.7,8.3)) +
    theme(legend.title = element_text(size=10), legend.text=element_text(size=10))
  } else {
    gg <- ggplot() +
      geom_line(data = plot_dat, mapping = aes(x = edu, y = ccf, group = mn, color = mn)) +
      labs(color = 'Contraceptive\nMet Need') +
      theme(legend.position = "bottom") +
      labs(x = 'Education (years)', y = 'Completed Cohort Fertility (CCF50)') +
      scale_x_continuous(breaks = 0:18, limits = c(0,18)) +
      scale_y_continuous(breaks = 1:8, limits = c(0.7,8.3)) +
      theme(legend.title = element_text(size=10), legend.text=element_text(size=10))
  }

  print(gg)
  ggsave(paste0(version_plot_dir, "/ccf50-fit.pdf"), height = 10, width = 14)
  return(NULL)
}


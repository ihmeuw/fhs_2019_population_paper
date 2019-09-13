#' Helper script - Parallelizes over location. Computes the incremental model results using

library(data.table)
library(magrittr)
library(argparse)
library(mortcore)
library(splines)

######## ARGUMENT BLOCK ##########################################################
parser <- ArgumentParser()
parser$add_argument(
  "--task_map_file",
  help = "name of task map",
  required = TRUE
)
args <- parser$parse_args()
list2env(args, environment()); rm(args)

task_id      <- as.integer(Sys.getenv("SGE_TASK_ID"))
task_map     <- fread(task_map_file)
version_dir  <- task_map[task_id, version_dir]
version      <- task_map[task_id, version]
code_path    <- task_map[task_id, code_path]
lid          <- task_map[task_id, location_id]
cohort_past_start     <- task_map[task_id, cohort_past_start]
cohort_forecast_start <- task_map[task_id, cohort_forecast_start]
cohort_forecast_end   <- task_map[task_id, cohort_forecast_end]
draws <- task_map[task_id, draws]


setwd(code_path)

######## DIRECTORIES ########################################################
version_cov_dir        <- sprintf("%s/covariates", version_dir)
version_past_dir       <- sprintf("%s/past_diff_data", version_dir)
version_ccfx_dir       <- sprintf("%s/ccfx", version_dir)
version_ccfx_asfr_dir  <- sprintf("%s/asfr", version_ccfx_dir)
version_ccfx_last_dir  <- sprintf("%s/last_past", version_ccfx_dir)
version_ccfx_ccf_dir   <- sprintf("%s/ccf50", version_ccfx_dir)

######## MODULES AND FUNCTIONS ####################################################
source("forecast_arima.R")
source("extract_coefs_knots_spline_matrix.R")
source("sort_draws.R")

ccfx_pred <- function(lid, cohort_past_start, cohort_forecast_start, cohort_forecast_end, draws) {

  # Pull in data, lm models, simulated betas
  df        <- fread(sprintf("%s/ccfx_data.csv", version_ccfx_dir))
  df        <- df[location_id == lid,.(location_id, year_id, age_group_id, scenario, cohort, diff_age_group, diff_age_first, ccfx, ccf50, ccfx_diff, ln_ccfx_diff)]
  models    <- readRDS(sprintf("%s/ccfx_models.rds", version_ccfx_dir))
  sim_betas <- readRDS(sprintf("%s/sim_betas.rds", version_ccfx_dir))
  setkeyv(df, c("year_id", "age_group_id", "scenario"))

  # Filter data down to the max CCFX attained by each cohort
  df <- df[!is.na(ccfx)]
  df <- df[, max_ccfx_age := max(diff_age_first), by = cohort]
  df <- df[diff_age_first <= max_ccfx_age]

  # Predict for each age group
  for(age in rev(seq(20, 45, 5))) {
    message(age)
    int_cohort <- cohort_forecast_start - 35

    # Subset data
    fit     <- models[[as.character(age)]]
    sim     <- sim_betas[[as.character(age)]]
    temp    <- copy(df[diff_age_first == age])
    temp[, `:=`(h = NULL, eps = NULL)] # since merging these two variables repeatedly, delete in beginning
    n_draws <- nrow(sim)
    if (draws != n_draws) {
      message("Number of specified draws does not match what is in covariate draw file. Using that instead.")
      draws <- n_draws
    }

    # Make predictions incorporating both COVARIATE and BETA uncertainty. Do by loc, age, cohort, scenario
    temp_list <- split(temp, by = c("cohort", "scenario"))
    message("Begin Secular Trend Prediction")
    temp <- rbindlist(lapply(temp_list, function(b){
      X    <- create_design_matrix(model = fit, dt = b)
      pred <- X %*% t(sim)
      # Since no covariate uncertainty, need to expand out data table to N number of draws in the betas
      b <- b[rep(1:.N, times = n_draws)][, draw := 0:(.N-1)]
      b[, est_ln_diff := pred]
      return(b)
    }))
    # My estimate will be the CCFX plus the actual model
    temp[, est := log(ccfx + exp(est_ln_diff))]
    # Once have secular trend, need to shift predictions to the int_cohort (either data or predictions)
    temp[, eps_past := log(ccf50) - est]
    eps_ts <- temp[!is.na(eps_past) & scenario == 0, .(eps_past = mean(eps_past)), by = .(cohort)]

    message("Begin ARIMA prediction")
    # Function to turn time series of past epsilon into X number of draws for forecasts up to last year.
    set.seed(20140719)
    eps <- forecast_arima(y = eps_ts$eps_past,
                          order = c(0, 1, 0),
                          forecast_start = int_cohort + 1,
                          num_draws = draws,
                          forecast_end = ifelse(age != 20, int_cohort + (49 - age + 1), int_cohort + (49 - age + 5)),
                          include.drift = F)$draw_forecast

    eps[, diff_age_first := age]
    eps[, cohort := int_cohort + h]
    temp <- merge(temp, eps, all.x = TRUE)

    # Swap out the age group from the original data frame with this new one with forecasts of secular trend and eps
    # Re oorder so that the age groups by cohort are in order (so that shift works)
    temp <- temp[order(scenario, cohort, draw, diff_age_first)]

    # Compute log final estimate as sum of secular trend and epsilon
    df <- df[diff_age_first != age]
    df <- rbind(df, temp, fill = TRUE)
  }

  setnames(df, "diff_age_first", "byage")

  # For each age group, this is the cohort that all draws should intercept shift to
  df[, int_cohort := cohort_forecast_start - byage + 10]

  # First intercept shift the CCF45 group
  df[byage == 45, ystar := est + eps]
  df[byage != 45, est := est + eps]

  # Now do a series of intercept shifts starting from CCF40, down to CCF20
  scenarios <- unique(temp$scenario)
  for(age in rev(seq(20, 40, 5))){
    message(age)
    temp <- df[byage == age]
    int_cohort_new <- temp[,unique(int_cohort)]
    int  <- df[byage == age + 5 & cohort == int_cohort_new][,.(scenario, draw, ystar)]
    setnames(int, "ystar", "ystar_int")

    temp <- rbindlist(lapply(scenarios, sort_draws, draw_data = temp, past_draw = int,
                             cohort_forecast_start = int_cohort_new + 1,
                             cohort_forecast_end = int_cohort_new + 4))

    temp[, shift := NULL]
    df <- df[byage != age]
    df <- rbind(df, temp, fill = TRUE)
  }

  # Backtransform draws from log to normal space with bias adjustment
  df[, variance := var(ystar), by = .(cohort, scenario, byage)]
  df[, ystar := exp(ystar - variance / 2)]
  df <- df[byage == max_ccfx_age]
  df <- df[!is.na(ystar)]

  ## FORCING DRAWS ABOVE 1 #############
  df[ystar < 1, ystar := 1.01]
  ######################################

  # Calculate ccf at each age and save the ccf mean lower and upper as data for secular trend model
  ccf <- df[, .(ccf = mean(ystar),
                lower = quantile(ystar, 0.025),
                upper = quantile(ystar, 0.975)), by = .(location_id, scenario, cohort)]

  # Last past draw
  last_past_draws <- df[cohort == cohort_forecast_start - 1, .(location_id, scenario, draw, ystar)]
  setnames(last_past_draws, "ystar", "ccf")
  # CCF Draws
  ccf_draws <- df[,.(location_id, cohort, scenario, draw, ystar)]
  setnames(ccf_draws, "ystar", "ccf")
  # CCF50 input data. If ages get dropped due to covariate availability, need to pull in proper ages from past data
  single_year <- fread(sprintf("%s/single_year_asfr.csv", version_past_dir))[location_id == lid,.(location_id, cohort, age, mean)]
  single_year[, num_ages := .N, by = cohort]
  past <- single_year[num_ages == 35, .(location_id = unique(location_id), ccf = sum(mean), lower = sum(mean), upper = sum(mean)), by = cohort]
  past <- past[rep(1:.N, length(scenarios))][, scenario := scenarios, by = cohort]
  ccf  <- rbind(past, ccf[cohort > past[, max(cohort)]])

  #fwrite(all, sprintf("%s/all_data_%s.csv", version_incomplete_all_dir, lid))
  fwrite(last_past_draws,  sprintf("%s/last_past_%s.csv", version_ccfx_last_dir, lid))
  fwrite(ccf, sprintf("%s/ccf50_%s.csv", version_ccfx_ccf_dir, lid))
  fwrite(ccf_draws, sprintf("%s/%s.csv", version_ccfx_asfr_dir, lid))
}

ccfx_pred(
  lid = lid,
  cohort_past_start = cohort_past_start,
  cohort_forecast_start = cohort_forecast_start,
  cohort_forecast_end = cohort_forecast_end,
  draws = draws)

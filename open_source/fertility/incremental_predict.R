
library(data.table)
library(magrittr)
library(argparse)
library(mortcore)
library(reticulate)
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

task_id                <- as.integer(Sys.getenv("SGE_TASK_ID"))
task_map               <- fread(task_map_file)
version                <- task_map[task_id, version]
lid                    <- task_map[task_id, location_id]
fedu_version           <- task_map[task_id, fedu_version]
mn_version             <- task_map[task_id, mn_version]
env                    <- task_map[task_id, env]
cohort_past_start      <- task_map[task_id, cohort_past_start]
cohort_forecast_start  <- task_map[task_id, cohort_forecast_start]
cohort_forecast_end    <- task_map[task_id, cohort_forecast_end]
draws                  <- task_map[task_id, draws]
future_start_year_id   <- task_map[task_id, future_start_year_id]
scenario_start_year_id <- task_map[task_id, scenario_start_year_id]
version_dir            <- task_map[task_id, version_dir]
code_path              <- task_map[task_id, code_path]
gbd_round              <- task_map[task_id, gbd_round]


setwd(code_path)

######## DIRECTORIES ########################################################
version_cov_dir              <- sprintf("%s/covariates", version_dir)
version_incomplete_dir       <- sprintf("%s/incomplete", version_dir)
version_incomplete_pasfr_dir <- sprintf("%s/pasfr", version_incomplete_dir)
version_incomplete_all_dir   <- sprintf("%s/all_data", version_incomplete_dir)
version_incomplete_asfr_dir  <- sprintf("%s/asfr", version_incomplete_dir)
version_incomplete_last_dir  <- sprintf("%s/last_past", version_incomplete_dir)
version_incomplete_ccf_dir   <- sprintf("%s/ccf50", version_incomplete_dir)

######## MODULES AND FUNCTIONS ####################################################
source("forecast_arima.R")
source("extract_coefs_knots_spline_matrix.R")

incremental_pred <- function(lid, fedu_version, mn_version, env,
                             cohort_past_start, cohort_forecast_start, cohort_forecast_end,
                             draws, future_start_year_id, scenario_start_year_id, gbd_round) {

  message(sprintf("Sourcing Conda Environment %s", env))
  reticulate::use_condaenv(env, conda = FILEPATH, required = T)
  reticulate::source_python(sprintf("%s/get_rid_of_draws_scenarios_and_sex_id.py", code_path))

  edu_dir <- as.character(FBDPath(sprintf("/%i/future/education/%s/education.nc", gbd_round, fedu_version)))
  mn_dir <- as.character(FBDPath(sprintf("/%i/future/met_need/%s/met_need.nc", gbd_round, mn_version)))

  # Covariate draws age-specific draws - mn and edu
  message("Reading Edu draws")
  edu_draws <- data.table(get_rid_of_draws_scenarios_and_sex_id(file = edu_dir,
                                                                select_lid = lid,
                                                                drop_draw=FALSE)) %>% setnames("value", "edu")
  message("Reading MN draws")
  mn_draws <- data.table(get_rid_of_draws_scenarios_and_sex_id(file = mn_dir,
                                                               select_lid = lid,
                                                               drop_draw=FALSE)) %>% setnames("value", "mn")

  gbd_year <- ifelse(gbd_round == 5, 2017, stop("gbd_year not currently supported"))
  if (gbd_year > future_start_year_id) {
    # First year of past met need data
    mn_draws_t0 <- min(mn_draws$year_id)
    met_need_start <- future_start_year_id - 49 + 15
    if (mn_draws_t0 > met_need_start) {
      add_mn_years <- met_need_start:(mn_draws_t0-1)
      add_mn_draws <- rbindlist(lapply(add_mn_years, function(yr) {
        add_yr <- subset(mn_draws, year_id == mn_draws_t0)
        add_yr$year_id <- yr
        return(add_yr)
      })
      )
      mn_draws <- rbind(add_mn_draws, mn_draws)
    }
  }

  # Pull in data, lm models, simulated betas
  data      <- fread(sprintf("%s/ccf_diff_data_w_covariates.csv", version_incomplete_dir))
  data      <- data[location_id == lid,.(location_id, year_id, age_group_id, scenario, cohort, diff_age_group, diff_age_first, lag_diff_age_group, ln_diff)]
  models    <- readRDS(sprintf("%s/incremental_models.rds", version_incomplete_dir))
  sim_betas <- readRDS(sprintf("%s/sim_betas.rds", version_incomplete_dir))

  setkeyv(data, c("year_id", "age_group_id", "scenario"))
  data[, ln_lag := shift(ln_diff, n = 1L, type = "lag"), by = .(scenario, cohort)]

  if (nrow(sim_betas[[1]]) != draws) {
    message("Using all draws in sim_betas though it doesn't match supplied draws.")
    draws <- nrow(sim_betas[[1]])
  }

  edu_draws <- edu_draws[draw %in% 0:(draws-1)]
  mn_draws <- mn_draws[draw %in% 0:(draws-1)]
  cov_draws <- merge(edu_draws, mn_draws)[, `:=`(sex_id = NULL, location_id = NULL)]
  setkeyv(cov_draws, c("year_id", "age_group_id", "scenario", "draw"))

  # Create draw version of the incremental model fit - ln_diff, draw, edu, mn, ln_lag.
  df <- merge(data, cov_draws, allow.cartesian = T)[order(cohort, scenario, draw)]

  # Predict for each age group
  for(age in seq(20,50,5)) {
    message(age)
    # Establish the last cohort with past data, given the age group.
    # If age 15-19 in 2017, just assume same fertility to finish out the 5-yr age group
    int_cohort <- ifelse(age != 20, cohort_forecast_start - age + 15, cohort_forecast_start - age + 19)

    # Subset data (only with education and met need covariates)
    fit     <- models[[age / 5 - 3]]
    sim     <- sim_betas[[age / 5 - 3]]
    temp    <- copy(df[diff_age_first == age])
    if(age != 20){temp[, `:=`(h = NULL, eps = NULL)]}

    # Make predictions incorporating both COVARIATE and BETA uncertainty. Do by loc, age, cohort, scenario
    temp_list <- split(temp, by = c("cohort", "scenario"))
    message("Begin Secular Trend Prediction")
    temp <- rbindlist(lapply(temp_list, function(b){
      b <- subset(b, draw < draws)
      X    <- create_design_matrix(model = fit, dt = b)
      pred <- apply(X * sim, 1, sum)
      b[, est := pred]
      return(b)
    }))
    temp[, eps_past := ln_diff - est]
    eps_ts <- temp[!is.na(eps_past) & scenario == 0, .(eps_past = mean(eps_past)), by = .(cohort)]

    message("Begin ARIMA prediction")
    # Function to turn time series of past epsilon into X number of draws for forecasts up to last year.
    set.seed(lid)
    eps <- forecast_arima(y = eps_ts$eps_past,
                          order = c(0, 1, 0),
                          forecast_start = int_cohort + 1,
                          forecast_end = cohort_forecast_end,
                          num_draws = draws,
                          include.drift = F)$draw_forecast

    eps[, diff_age_first := age]
    eps[, cohort := int_cohort + h]
    temp <- merge(temp, eps, all.x = T)

    # Swap out the age group from the original data frame with this new one with forecasts of secular trend and eps
    # Re oorder so that the age groups by cohort are in order (so that shift works)
    df <- fsetdiff(df, df[diff_age_first == age])
    df <- rbind(df, temp, fill = T)
    df <- df[order(scenario, cohort, draw, diff_age_first)]

    # Compute log final estimate as sum of secular trend and epsilon
    df[, ystar := est + eps]
    # To propagate the predictions up - need to fill in the ln_lag column with the new column
    df[, newlndiff := ln_diff]
    df[is.na(newlndiff), newlndiff := ystar]
    message("Shifting the forecasts to ln_lag of older age group")
    df[, est_ln_lag := shift(newlndiff, type = "lag", n = 1), by = .(cohort, scenario, draw)]
    df[is.na(ln_lag), ln_lag := est_ln_lag]
  }

  setnames(df, "diff_age_first", "byage")

  # Backtransform draws from log to normal space with bias adjustment
  df[, variance := var(ystar), by = .(cohort, scenario, diff_age_group)]
  df[, ystar := exp(ystar - variance / 2)]
  df[is.na(ystar), ystar := exp(ln_diff)]
  df[, num_age_groups := .N, .(scenario, cohort, draw)]

  # CCF diagnostic data. Mean upper and lowers
  all <- df[,.(location_id, scenario, cohort, byage, draw, est, eps_past, eps, ystar, newlndiff)]
  all <- all[,.(est = mean(est),                          # Secular trend
                eps = mean(eps),                          # Forecasted epsilon
                eps_past = mean(eps_past),                # Past epsilons
                ystar = mean(ystar),                      # Exponentiated ICF's
                newlndiff = mean(newlndiff)),             # Log ICF's (past and future)
             by = .(location_id, scenario, cohort, byage)]

  df <- df[num_age_groups == 7]
  df[, ccf := cumsum(ystar), by = .(scenario, cohort, draw)]
  # Calculate ccf at each age and save the ccf mean lower and upper as data for secular trend model
  summary <- df[, .(ccf = mean(ccf),
                    lower = quantile(ccf, 0.025),
                    upper = quantile(ccf, 0.975)), by = .(location_id, scenario, cohort, byage)]

  # PASFR - Compute the age-specific contribuion to CCF and arrive at PASFR
  df[, final_ccf := max(ccf), by = .(scenario, cohort, draw)]
  df[, pasfr := ystar / final_ccf]

  # PASFR
  pasfr <- df[, .(location_id, scenario, cohort, byage, draw, pasfr)]
  # Last past draw
  last_past_draws <- df[cohort == cohort_forecast_start - 1 & byage == 50, .(location_id, scenario, draw, ccf)]
  # Location specific asfr draws
  asfr_draws <- df[,.(location_id, scenario, cohort, byage, draw, ccf)]
  scenarios <- unique(asfr_draws$scenario)
  # CCF50 input data. If ages get dropped due to covariate availability, need to pull in proper ages from past data
  ccf <- summary[byage == 50][, byage := NULL]
  single_year <- fread(sprintf("%s/past_diff_data/single_year_asfr.csv", version_dir))[location_id == lid,.(location_id, cohort, age, mean)]
  single_year[, num_ages := .N, by = cohort]
  past <- single_year[num_ages == 35, .(location_id = unique(location_id), ccf = sum(mean), lower = sum(mean), upper = sum(mean)), by = cohort]
  past <- past[rep(1:.N, length(scenarios))][, scenario := scenarios, by = cohort]
  ccf  <- rbind(past, ccf[cohort > past[, max(cohort)]])

  fwrite(all, sprintf("%s/all_data_%s.csv", version_incomplete_all_dir, lid))
  fwrite(pasfr, sprintf("%s/pasfr_%s.csv", version_incomplete_pasfr_dir, lid))
  fwrite(last_past_draws,  sprintf("%s/last_past_%s.csv", version_incomplete_last_dir, lid))
  fwrite(ccf, sprintf("%s/ccf50_%s.csv", version_incomplete_ccf_dir, lid))
  fwrite(asfr_draws, sprintf("%s/%s.csv", version_incomplete_asfr_dir, lid))
}


incremental_pred(lid = lid,
                 fedu_version = fedu_version,
                 mn_version = mn_version,
                 env = env,
                 cohort_past_start = cohort_past_start,
                 cohort_forecast_start = cohort_forecast_start,
                 cohort_forecast_end = cohort_forecast_end,
                 draws = draws,
                 future_start_year_id = future_start_year_id,
                 scenario_start_year_id = scenario_start_year_id,
                 gbd_round = gbd_round)

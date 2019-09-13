#' Helper script - Parallelizes over location.
#' Computes the CCF50 draws results using
#' draws of the covariates
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
merge_aid              <- task_map[task_id, merge_aid]
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

######## GLOBAL VARIABLES ########################################################
# Define directories dependent on version
version_cov_dir              <- sprintf("%s/covariates", version_dir)
version_complete_dir         <- sprintf("%s/complete", version_dir)
version_complete_asfr_dir    <- sprintf("%s/asfr", version_complete_dir)
version_complete_ccf_dir     <- sprintf("%s/ccf50", version_complete_dir)
version_complete_all_dir     <- sprintf("%s/all_data", version_complete_dir)
version_incomplete_dir       <- sprintf("%s/incomplete", version_dir)
version_ccfx_dir             <- sprintf("%s/ccfx", version_dir)

######## MODULES AND FUNCTIONS ####################################################
source("forecast_arima.R")
source("extract_coefs_knots_spline_matrix.R")
source("transformations.R")
source("sort_draws.R")

#' 1. Make secular trend predictions with draws of betas and covariates
#' 2. Fit the ARIMA component and compute CCF50 forecasts
#' 3. Sort draws to last past cohort

complete_pred <- function(lid, fedu_version, mn_version, env, merge_aid,
                          cohort_past_start, cohort_forecast_start, cohort_forecast_end, draws,
                          future_start_year_id, scenario_start_year_id, gbd_round) {

  merge_age <- fread(sprintf("%s/age_map.csv", version_cov_dir))[age_group_id == merge_aid, age_group_years_start]

  message(sprintf("Sourcing Conda Environment %s", env))
  reticulate::use_condaenv(env, conda = FILEPATH, required = T)
  reticulate::source_python(sprintf("%s/get_rid_of_draws_scenarios_and_sex_id.py", code_path))

  edu_dir <- as.character(FBDPath(sprintf("/%i/future/education/%s/education.nc", gbd_round, fedu_version)))
  mn_dir <- as.character(FBDPath(sprintf("/%i/future/met_need/%s/met_need.nc", gbd_round, mn_version)))

  # Covariate draws age-specific draws - mn and edu
  message("Reading Edu draws")
  edu_draws <- data.table(get_rid_of_draws_scenarios_and_sex_id(file = edu_dir,
                                                                select_lid = lid,
                                                                select_age_group_ids = merge_aid,
                                                                drop_draw=FALSE)) %>% setnames("value", "edu")
  message("Reading MN draws")
  mn_draws <- data.table(get_rid_of_draws_scenarios_and_sex_id(file = mn_dir,
                                                               select_lid = lid,
                                                               select_age_group_ids = merge_aid,
                                                               drop_draw=FALSE)) %>% setnames("value", "mn")

  # Set draws for future years prior to scenario start year to reference
  edu_draws[year_id %in% future_start_year_id:(scenario_start_year_id-1), edu := edu[scenario == 0], by = .(location_id, year_id, age_group_id, sex_id, draw)]
  mn_draws[year_id %in% future_start_year_id:(scenario_start_year_id-1), mn := mn[scenario == 0], by = .(location_id, year_id, age_group_id, sex_id, draw)]

  cov_draws <- merge(edu_draws, mn_draws)[, `:=`(sex_id = NULL, location_id = NULL)]
  cov_draws[, cohort := year_id - merge_age]
  setkeyv(cov_draws, c("cohort", "scenario", "draw"))

  # Pull in data, lm models, simulated betas
  data      <- fread(sprintf("%s/ccf_input_data_ccfx.csv", version_complete_dir))[location_id == lid]
  model     <- readRDS(sprintf("%s/complete_model.rds", version_complete_dir))
  sim_betas <- readRDS(sprintf("%s/sim_betas.rds", version_complete_dir))
  setkeyv(data, c("cohort", "scenario"))

  if (nrow(sim_betas) != draws) {
    message("Using all draws in sim_betas though it doesn't match supplied draws.")
    draws <- nrow(sim_betas)
  }

  cov_draws <- cov_draws[draw %in% 0:(draws-1)]

  # Create draw version of the incremental model fit - ln_diff, draw, edu, mn, ln_lag.
  df <- merge(data, cov_draws, allow.cartesian = T)
  scenarios <- unique(df$scenario)
  # Make predictions incorporating both COVARIATE and BETA uncertainty
  df_list <- split(df, by = c("cohort", "scenario"))
  message("Begin Secular Trend Prediction")
  draw_data <- rbindlist(lapply(df_list, function(b){
    X    <- create_design_matrix(model, b)
    pred <- apply(X * sim_betas, 1, sum)
    b[, est := pred]
    return(b)
  }))


  #' Turn normal CCF secular trend into scaled logit. Caluclate epsilons in logit
  draw_data[, est := scaled_logit(est, 1, 10)]
  draw_data[, logit_ccf := scaled_logit(ccf, 1, 10)]
  draw_data[, eps_past := logit_ccf - est]

  message("Begin ARIMA prediction")
  eps_ts <- draw_data[!is.na(eps_past) & scenario == 0, .(eps_past = mean(eps_past)), by = cohort]
  # Function to turn time series of past epsilon into X number of draws for forecasts up to last year.
  set.seed(lid)
  eps <- forecast_arima(y = eps_ts$eps_past,
                        order = c(0, 1, 0),
                        method = "ML",
                        forecast_start = cohort_forecast_start,
                        forecast_end = cohort_forecast_end,
                        num_draws = draws)$draw_forecast
  eps[, cohort := cohort_forecast_start - 1 + h]
  draw_data <- rbindlist(lapply(scenarios, function(s){
    message(s)
    temp <- merge(draw_data[scenario == s], eps, all.x = T)
    return(temp)
  }))


  # Forecast period, sum of trend and forecasted residuals.
  draw_data[, ystar := est + eps]
  # For past, just use the past data.
  draw_data[is.na(ystar), ystar := logit_ccf]
  draw_data[, ystar := anti_scaled_logit(ystar, lower = 1, upper = 10, bias_adj = T), by = .(cohort, scenario)]

  # Intercept shift at the draw level, between the last forecasted cohort and last past year
  past_draw <- fread(sprintf("%s/last_past/last_past_%s.csv", version_ccfx_dir, lid))
  past_draw[, y := scaled_logit(ccf, lower = 1, upper = 10)]
  pred_draws   <- rbindlist(lapply(scenarios, sort_draws, draw_data = draw_data, past_draw = past_draw,
                                   cohort_forecast_start = cohort_forecast_start, cohort_forecast_end = cohort_forecast_end))

  pred_draws[ystar <= 1, ystar := 1.01]

  # Calculate ccf at each age and save the ccf mean lower and upper as data for secular trend model
  summary <- pred_draws[cohort >= cohort_forecast_start, .(ccf = mean(ystar),
                                                           lower = quantile(ystar, 0.025),
                                                           upper = quantile(ystar, 0.975)), by = .(location_id, scenario, cohort)]
  summary <- rbind(data[cohort < cohort_forecast_start], summary)
  ccf_draws <- pred_draws[cohort >= cohort_forecast_start, .(location_id, cohort, scenario, draw, ystar)]
  setnames(ccf_draws, "ystar", "ccf")

  # All data
  all <- pred_draws[,.(location_id, cohort, scenario, draw, est, shift, eps_past, eps, ystar)]
  all <- all[, .(est = mean(est),
                 shift = mean(shift),
                 eps_past = mean(eps_past),
                 eps = mean(eps),
                 ystar = mean(ystar)),
             by = .(location_id, cohort, scenario)]
  fwrite(all, sprintf("%s/%s.csv", version_complete_all_dir, lid))
  fwrite(summary, sprintf("%s/%s.csv", version_complete_ccf_dir, lid))
  fwrite(ccf_draws, sprintf("%s/%s.csv", version_complete_asfr_dir, lid))
}


complete_pred(
  lid = lid,
  fedu_version = fedu_version,
  mn_version = mn_version,
  env = env,
  merge_aid = merge_aid,
  cohort_past_start = cohort_past_start,
  cohort_forecast_start = cohort_forecast_start,
  cohort_forecast_end = cohort_forecast_end,
  draws = draws,
  future_start_year_id = future_start_year_id,
  scenario_start_year_id = scenario_start_year_id,
  gbd_round = gbd_round)

# Prep data by merging CCF difference data with covariate (education and metneed) data.
prep_covariates <- function(metneed_version,
                            fedu_version,
                            condaenv,
                            gbd_round,
                            merge_aid,
                            version_cov_dir,
                            codepath) {
  #' @description Preps covariates for model fitting by collapsing to the mean and necessary age groups prior to use in the incremental and
  #' complete cohort fertility models. Also saves draw version of those covariates.
  #' @param metneed_version : Version of met need to use. Assumes past and future are present in the output file
  #' @param fedu_version : Version of education to use. Assumes past and future are present in the output file.
  #' @param env : Name of desired conda environment.
  #' @param version_cov_dir : Full file path to the CCF version covariate subfolder of the forecasting model
  #' @param merge_aid : The age group ID for the covariate that should represent the cohort in the secular trend model
  #' @param future_start_year_id First year of the future
  #' @param scenario_start_year_id first year that the covariates should be applied
  #' @param directory_list list of directories to avoid hard coding conflicts -- EMG addition
  #' @return None. Writes edu and mn covariates from .nc to .csv for easier use in R

  message(sprintf("Sourcing Conda Environment %s and python script", condaenv))
  reticulate::use_condaenv(condaenv, conda = FILEPATH, required = TRUE)
  reticulate::source_python(sprintf("%s/get_rid_of_draws_scenarios_and_sex_id.py", codepath))

  edu_dir <- as.character(FBDPath(sprintf("/%i/future/education/%s/education.nc", gbd_round, fedu_version)))
  mn_dir <- as.character(FBDPath(sprintf("/%i/future/met_need/%s/met_need.nc", gbd_round, metneed_version)))

  # Runs the python function to get a data table in R.
  message("Collapsing Education to means")
  edu <- data.table(get_rid_of_draws_scenarios_and_sex_id(edu_dir)) %>% setnames("value", "edu")
  message("Collapsing Met Need to means")
  mn  <- data.table(get_rid_of_draws_scenarios_and_sex_id(mn_dir)) %>% setnames("value", "mn")

  fwrite(edu, sprintf("%s/edu_forecast.csv", version_cov_dir))
  fwrite(mn, sprintf("%s/mn_forecast.csv", version_cov_dir))
}

#' @description Detect and check scenarios in the covariate files.
#' @returns A vector of the scenarios.
get_scenarios_from_prepped_covariates <- function(version_cov_dir) {
  edu      <- fread(sprintf("%s/edu_forecast.csv", version_cov_dir))
  mn       <- fread(sprintf("%s/mn_forecast.csv", version_cov_dir))
  scenarios_edu <- unique(edu$scenario)
  scenarios_mn <- unique(mn$scenario)
  if (!(all(scenarios_edu == scenarios_mn)))
    stop("Scenarios in met need and education differ.")
  out <- scenarios_edu
  if (!(0 %in% out))
    stop("Could not detect a reference scenario in the covariates. Reference must be labeled as 0.")
  return(out)
}

merge_covariates <- function(version_cov_dir,
                             version_incomplete_dir,
                             version_ccfx_dir,
                             version_past_dir,
                             past_start,
                             forecast_start,
                             forecast_end) {
  #' @description Attaches the necessary covariates - met need and education to the CCF diff past data.
  #' @param version_cov_dir - Covariate subdirectory for the version
  #' @param version_incomplete_dir - Incomplete cohort subdirectory for the version
  #' @param version_ccfx_dir - Directory for CCFX
  #' @return None. Just saves flat file of covariates with CCF difference data folder

  age_map  <- data.table(age_group_id = 8:14, age_group_years_start = seq(15, 45, 5))

  # Youngest cohort is 15 at start of forecast
  ccf_diff <- fread(sprintf("%s/ccf_diff_data.csv", version_past_dir))[between(cohort, past_start - 25, forecast_start - 16)]
  edu      <- fread(sprintf("%s/edu_forecast.csv", version_cov_dir))
  mn       <- fread(sprintf("%s/mn_forecast.csv", version_cov_dir))

  # Detect and check scenarios
  scenarios_edu <- unique(edu$scenario)
  scenarios_mn <- unique(mn$scenario)
  if (!(all(scenarios_edu == scenarios_mn)))
    stop("Scenarios in met need and education differ.")
  scenarios <- scenarios_edu

  forecasts <- CJ(ihme_loc_id = ccf_diff[,unique(ihme_loc_id)],
                  cohort = (forecast_start - 15):(forecast_end - 15),
                  diff_age_group = ccf_diff[,unique(diff_age_group)])
  map <- unique(ccf_diff[,.(ihme_loc_id, location_id, diff_age_group, diff_age_first, diff_age_second, lag_diff_age_group)])
  setkeyv(map, c("ihme_loc_id", "diff_age_group"))
  forecasts <- merge(forecasts, map, all.x = TRUE)
  ccf_diff  <- rbind(ccf_diff, forecasts, fill = TRUE)

  # Merging in CCF difference on the year that the cohort reaches the age start of the difference.
  ccf_diff <- merge(ccf_diff, age_map[,.(age_group_years_start, age_group_id)],
                    by.x = "diff_age_second",
                    by.y = "age_group_years_start", all.x = TRUE)
  ccf_diff[, year_id := cohort + diff_age_second]
  cov_merge <- merge(edu, mn,
                     by = c("location_id", "year_id", "age_group_id", "scenario", "sex_id"),
                     all.x = TRUE) # EDU has more past years than MN
  cov_merge[,sex_id := NULL]
  tmp_ccf_diff <- rbindlist(lapply(scenarios, function(s) {
    out <- ccf_diff
    out$scenario <- s
    out
  }))
  ccf_diff_merge <- merge(tmp_ccf_diff, cov_merge,
                          by = c("location_id", "year_id", "age_group_id", "scenario"),
                          all.x = TRUE)
  ccf_diff <- ccf_diff_merge
  # Saving CCFX data for alternative cohort completion model.
  ccfx_data <- ccf_diff[, ccfx := cumsum(diff), by = .(location_id, cohort, scenario)]
  ccf50     <- ccfx_data[diff_age_first == 50]
  setnames(ccf50, "ccfx", "ccf50")
  ccfx_data <- merge(ccfx_data, ccf50[,.(location_id, cohort, scenario, ccf50)], by = c("location_id", "cohort", "scenario"), allow.cartesian = TRUE)
  ccfx_data[, ccfx_diff := ccf50 - ccfx]
  ccfx_data[, ln_ccfx_diff := log(ccfx_diff)]

  fwrite(ccfx_data, sprintf("%s/ccfx_data.csv", version_ccfx_dir))
  fwrite(ccf_diff, sprintf("%s/ccf_diff_data_w_covariates.csv", version_incomplete_dir))
}

# Purpose: fit the CCF50 model with explanatory variables met need and education (females @ age 25)

complete_model_fit <- function(version_complete_dir, version_cov_dir, merge_aid, draws,
                               cohort_past_start, cohort_forecast_start,
                               model.formula = "ccf ~ ns(edu, knots = c(3.1, 8.14, 12.8)) + mn") {

  # Age and Locs
  age_map   <- fread(sprintf("%s/age_map.csv", version_cov_dir))
  merge_age <- age_map[age_group_id == merge_aid, age_group_years_start]
  locs      <- fread(sprintf("%s/locs.csv", version_cov_dir))

  # Read input data
  data <- fread(sprintf("%s/ccf_input_data_ccfx.csv", version_complete_dir))
  data <- data[between(cohort, cohort_past_start, cohort_forecast_start-1L)]
  data[, year_id := cohort + merge_age]

  # Mean Covariates
  edu        <- fread(sprintf("%s/edu_forecast.csv", version_cov_dir))
  mn         <- fread(sprintf("%s/mn_forecast.csv", version_cov_dir))
  merge_edu  <- edu[age_group_id == merge_aid]
  merge_mn   <- mn[age_group_id == merge_aid]
  covariates <- merge(merge_edu, merge_mn)
  covariates[, `:=`(sex_id = NULL, age_group_id = NULL)]

  # Create data frame for modeling, covs
  data <- merge(data, covariates, by = c("location_id", "year_id", "scenario"))

  # Fit model on scenario 0 data.
  model_data <- data[scenario == 0 & !is.na(ccf)]
  model      <- lm(model.formula, data = model_data)

  betas        <- coef(model)
  vc           <- vcov(model)
  set.seed(20140719)
  sim_betas    <- MASS::mvrnorm(n = draws, betas, vc)
  fit_stats    <- data.table(broom::glance(model))
  coef_summary <- data.table(broom::tidy(model))

  fwrite(data, file = sprintf("%s/ccf_input_data_w_covariates.csv", version_complete_dir))
  fwrite(coef_summary, file = sprintf("%s/model_coefs.csv", version_complete_dir))
  fwrite(fit_stats, file = sprintf("%s/fit_stats.csv", version_complete_dir))
  saveRDS(sim_betas, file = sprintf("%s/sim_betas.rds", version_complete_dir))
  saveRDS(model, file = sprintf("%s/complete_model.rds", version_complete_dir))

}


# Purpose: fitting  models to forecast remaining fertility in incomplete cohorts

ccfx_model_fit <- function(version_ccfx_dir, draws, cohort_past_start, cohort_forecast_start) {
  all_df   <- fread(sprintf("%s/ccfx_data.csv", version_ccfx_dir))[scenario == 0]
  model_df <- split(all_df[scenario == 0], by = "diff_age_first")
  model_df <- model_df[1:6]
  ages     <- names(model_df)

  # init lists
  formula      <- list() # Model formula template
  lag_df       <- list() # Knots for the lag coefficient
  edu_knots    <- list() # Knots for the education coefficient
  mn_knots     <- list() # Knots for the met need coefficients
  models       <- list() # S3 lm model objects for each age group
  sim_betas    <- list() # Matrix for draws of the regression coefficients
  fit_stats    <- list() # Glance of the model object - Rsquared, AIC, etc.
  coef_summary <- list() # data.table of regression coefficients and standard errors
  equations    <- list()

  # formulas
  SPLINE_CCFX <- "ln_ccfx_diff ~ ns(ccfx, df = %s)"

  for(a in ages) {
    message(a)
    data <- model_df[[a]]
    # Make sure ccfX is NA for out of sample holdouts
    # Note spline will be based on CCFX in incomplete cohort data,
    # which has more past than CCF50 in completed cohort data
    last_complete_cohort <- cohort_forecast_start - 35
    last_incomplete_cohort <- cohort_forecast_start - 35 + (50-as.numeric(a))
    data$ln_ccfx_diff[data$cohort > data$last_complete_cohort] <- NA
    data$ccfx[data$cohort > data$last_incomplete_cohort] <- NA

    # Subset on specified past years
    if (a == "20") {
      formula[[a]]   <- SPLINE_CCFX
      lag_df[[a]]    <- 3
      edu_knots[[a]] <- NA
      mn_knots[[a]]  <- NA
    } else if (a == "25") {
      formula[[a]]   <- SPLINE_CCFX
      lag_df[[a]]    <- 3
      edu_knots[[a]] <- NA
      mn_knots[[a]]  <- NA
    } else if (a == "30") {
      formula[[a]]   <- SPLINE_CCFX
      lag_df[[a]]    <- 3
      edu_knots[[a]] <- NA
      mn_knots[[a]]  <- NA
    } else if (a == "35") {
      formula[[a]]   <- SPLINE_CCFX
      lag_df[[a]]    <- 3
      edu_knots[[a]] <- NA
      mn_knots[[a]]  <- NA
    } else if (a == "40") {
      formula[[a]]   <- SPLINE_CCFX
      lag_df[[a]]    <- 3
      edu_knots[[a]] <- NA
      mn_knots[[a]]  <- NA
    } else if (a == "45") {
      formula[[a]]   <- SPLINE_CCFX
      lag_df[[a]]    <- 3
      edu_knots[[a]] <- NA
      mn_knots[[a]]  <- NA
    }

    equation       <- as.formula(sprintf(formula[[a]], lag_df[[a]], edu_knots[[a]], mn_knots[[a]]))
    model          <- lm(equation, data = data)
    models[[a]]    <- model
    equations[[a]] <- equation

    betas <- coef(model)
    vc    <- vcov(model)
    set.seed(20140719)
    sim_betas[[a]]    <- MASS::mvrnorm(n = draws, betas, vc)
    fit_stats[[a]]    <- data.table(broom::glance(model))[, age := a]
    coef_summary[[a]] <- data.table(broom::tidy(model))[, age := a]
  }

  fit_stats    <- rbindlist(fit_stats)
  coef_summary <- rbindlist(coef_summary)

  fwrite(coef_summary, file = sprintf("%s/model_coefs.csv", version_ccfx_dir))
  fwrite(fit_stats, file = sprintf("%s/fit_stats.csv", version_ccfx_dir))
  saveRDS(sim_betas, file = sprintf("%s/sim_betas.rds", version_ccfx_dir))
  saveRDS(models, file = sprintf("%s/ccfx_models.rds", version_ccfx_dir))
  saveRDS(equations, file = sprintf("%s/ccfx_formulas.rds", version_ccfx_dir))
}


# Different model formulas depending on age group
YOUNG                        <- "ln_diff ~ ns(edu, knots = 9) + mn"
SPLINE_LAG                   <- "ln_diff ~ ns(ln_lag, knots = c(%s)) + edu + mn"
SPLINE_LAG_BS_SPLINE_EDU     <- "ln_diff ~ ns(ln_lag, knots = c(%s)) + bs(edu, knots = c(%s)) + mn"

# Model fitting functions
select_knots <- function(df, num_knots, method = "quantile", xcov = "ln_lag"){
  df <- data.table(df)
  if (method == "quantile") {
    quantile(df[, get(xcov)],na.rm=T, prob = seq(0,1,length.out=num_knots+2)[2:(num_knots+1)]) %>% round(3) %>% paste0(collapse=", ")
  } else if (method == "equal") {
    quantile(df[, get(xcov)],na.rm=T, prob = seq(0,1,length.out=num_knots+2)[2:(num_knots+1)]) %>% round(3) %>% paste0(collapse=", ")
  }
}

incremental_model_fit <- function(version_incomplete_dir,
                                  draws,
                                  cohort_past_start,
                                  cohort_past_end){

  all_df   <- fread(sprintf("%s/ccf_diff_data_w_covariates.csv", version_incomplete_dir))
  all_df <- all_df[between(cohort, cohort_past_start, cohort_past_end)]
  model_df <- split(all_df[scenario == 0], by = "diff_age_group")
  ages     <- names(model_df)

  formula      <- list() # Model formula template
  lag_knots    <- list() # Knots for the lag coefficient
  edu_knots    <- list() # Knots for the education coefficient
  mn_knots     <- list() # Knots for the met need coefficients
  models       <- list() # S3 lm model objects for each age group
  sim_betas    <- list() # Matrix for draws of the regression coefficients
  fit_stats    <- list() # Glance of the model object - Rsquared, AIC, etc.
  coef_summary <- list() # data.table of regression coefficients and standard errors
  equations    <- list()

  for (a in ages){
    message(a)
    df <- model_df[[a]]
    if (a == "20_minus_15") {
      formula[[a]] <- YOUNG
      lag_knots[[a]] <- NA
      edu_knots[[a]] <- 9
      mn_knots[[a]] <- NA
    } else if (a == "25_minus_20") {
      formula[[a]] <- SPLINE_LAG_BS_SPLINE_EDU
      lag_knots[[a]] <- select_knots(df, 4, method = "quantile", xcov = "ln_lag")
      edu_knots[[a]] <- 12
      mn_knots[[a]] <- NA
    } else if (a == "30_minus_25") {
      formula[[a]] <- SPLINE_LAG
      lag_knots[[a]] <- select_knots(df, 4, method = "quantile", xcov = "ln_lag")
      edu_knots[[a]] <- NA
      mn_knots[[a]] <- NA
    } else if (a == "35_minus_30") {
      formula[[a]] <- SPLINE_LAG
      lag_knots[[a]] <- select_knots(df, 4, method = "quantile", xcov = "ln_lag")
      edu_knots[[a]] <- NA
      mn_knots[[a]] <- NA
    } else if (a == "40_minus_35") {
      formula[[a]] <- SPLINE_LAG
      lag_knots[[a]] <- select_knots(df, 4, method = "quantile", xcov = "ln_lag")
      edu_knots[[a]] <- NA
      mn_knots[[a]] <- NA
    } else if (a == "45_minus_40") {
      formula[[a]] <- SPLINE_LAG
      lag_knots[[a]] <- select_knots(df, 4, method = "quantile", xcov = "ln_lag")
      edu_knots[[a]] <- NA
      mn_knots[[a]] <- NA
    } else if (a == "50_minus_45") {
      formula[[a]] <- SPLINE_LAG
      lag_knots[[a]] <- select_knots(df, 4, method = "quantile", xcov = "ln_lag")
      edu_knots[[a]] <- NA
      mn_knots[[a]] <- NA
    }

    equation       <- as.formula(sprintf(formula[[a]], lag_knots[[a]], edu_knots[[a]], mn_knots[[a]]))
    model          <- lm(equation, data = df)
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

  fwrite(coef_summary, file = sprintf("%s/model_coefs.csv", version_incomplete_dir))
  fwrite(fit_stats, file = sprintf("%s/fit_stats.csv", version_incomplete_dir))
  saveRDS(sim_betas, file = sprintf("%s/sim_betas.rds", version_incomplete_dir))
  saveRDS(models, file = sprintf("%s/incremental_models.rds", version_incomplete_dir))
  saveRDS(equations, file = sprintf("%s/incremental_formulas.rds", version_incomplete_dir))
}


MODEL_WITH_INTERACTION <- "ccf ~ ns(edu, knots = c(3.1, 8.14, 12.8)) + mn + edu:mn"
MODEL <- "ccf ~ ns(edu, knots = c(3.1, 8.14, 12.8)) + mn"
MODEL_NO_MN <- "ccf ~ ns(edu, knots = c(3.1, 8.14, 12.8))"
version_complete_dir <- "/ihme/forecasting/data/5/future/ccf/20190806_141418_fix_draw_bound_ccfx_to2110/complete"
version_cov_dir <- "/ihme/forecasting/data/5/future/ccf/20190806_141418_fix_draw_bound_ccfx_to2110/covariates"
merge_aid <- 10
draws <- 1000
cohort_past_start <- 1955
cohort_forecast_start <- 2003

# Purpose: Modeling code for the incremental cohort fertility model. Switches between the tidyverse and data.table packages due to
# ease of use in some packages and not so much in others

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

model      <- lm(MODEL, data = model_data)
summary(model)
model_int      <- lm(MODEL_WITH_INTERACTION, data = model_data)
summary(model_int)
model_no_mn      <- lm(MODEL_NO_MN, data = model_data)
summary(model_no_mn)

anova(model, model_int)
anova(model, model_no_mn)

BIC(model) - BIC(model_int)
BIC(model) - BIC(model_no_mn)

model_data$pred <- fitted(model)
model_data$pred_int <- fitted(model_int)

ggplot(model_data) +
  geom_point(aes(x = ccf, y = pred, color = "No EDU:MN interaction"),
             alpha = 0.4) +
  geom_point(aes(x = ccf, y = pred_int, color = "With EDU:MN interaction"),
             alpha = 0.4) +
  labs(x = "Observed CCF", y = "Predicted CCF") +
  ggtitle("Predicted vs Observed CCF: include EDU:MN interaction?")

ggplot(subset(model_data, ccf < 2.1)) +
  geom_point(aes(x = ccf, y = pred, color = "No EDU:MN interaction"),
             alpha = 0.4) +
  geom_point(aes(x = ccf, y = pred_int, color = "With EDU:MN interaction"),
             alpha = 0.4) +
  labs(x = "Observed CCF", y = "Predicted CCF") +
  ggtitle("Predicted vs Observed CCF: include EDU:MN interaction?\nSubset CCF<2.1")

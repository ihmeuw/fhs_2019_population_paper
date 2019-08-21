## Get "terminal" CCF50 estimate for edu = 16 and mn = 0.95
## Used in the discussion 

library(splines)

fit <- readRDS("/share/forecasting/data/5/future/ccf/20190806_141418_fix_draw_bound_ccfx_to2110/complete/complete_model.rds")

newdat <- data.frame(edu = 16, mn = 0.95)

round(predict(fit, newdat, interval = "confidence"), 2)

## CCF model fit and years used for the methods section on CCF
summary(fit)
length(fit$model$ccf) / 195
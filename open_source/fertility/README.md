README for fertility code
=========================

Data preparation
-----------------
1. From GBD ASFR Data, prep CCF50 and incremental fertility data `prep_ccf_difference_data.R`
2. Merge in contraceptive met need and education covariates `prep_ccf_difference_covariates.R`

Forecasting incomplete cohorts: CCFX
-------------------------------------
1. Fit model `ccfx_model_fit.R`
2. Forecast `ccfx_predict.R`

Forecasting CCF50
------------------
1. Fit model `ccf50_model_fit.R`
2. Forecast `ccf50_predict.R`

Incremental fertility to obtain age-specific patterns
-----------------------------------------------------
1. Fit model `incremental_model_fit.R`
2. Forecast `incremental_predict.R`

Converting CCF50 to ASFR
-------------------------
1. Use CCF50 and PASFR to obtain ASFR `convert_ccf50_to_asfr.R`
2. Moving average smoothing `ccf_intercept_shift.py`

Compute TFR and aggregate over location hierarchies
---------------------------------------------------
1. `make_tfr_and_agg.py`


Utility functions used by above
-------------------------------
- Logit/expit `transformations.R`
- Draw sorting over time `sort_draws.R`
- ARIMA residual forecasting `forecast_arima.R`
- Utility functions for design matrices and regression coefficients with splines `extract_coefs_knots_spline_matrix.R`
- Convert ncdf files to R `xarray_to_r.R`
- Data extraction from ncdf source files `get_rid_of_draws_scenarios_and_sex_id.py`

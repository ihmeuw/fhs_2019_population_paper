README
=======
for `bayesTFR` code repo.

AUTHOR
=======
Emily Goren

DATE
=====
August 30, 2019


DESCRIPTION
============
This folder has code utilized in the populations paper for custom UNPD WPP2019 TFR forecasts.

These codes were used for:

1) replicate WPP2019 forecasts using their past data, to confirm ability to run code and check which locations were utilized in estimation of their Phase III fertility model. Associated code is in `bayesTFR_run_WPP2019.R` and the version is `/ihme/forecasting/data/wpp/future/tfr/2019_20190826_WPP2019`.


2) run a modified version of their model on their past data, where the Phase III fertility entrance criteria was modified to be only TFR < 2.0. Associated code is in `bayesTFR_run_WPP2019_custom_phaseIII.R` and the version is `/ihme/forecasting/data/wpp/future/tfr/2019_20190826_WPP2019_custom_phaseIII`.


3) compare predictive validity of their model to our fertility model, by comparing forecasts for 2008-2017 obtained using past data from 1980-2007. This required running their WPP2019 model using our GBD2017 data. The relevant code for forecast generatino is `bayesTFR_run_WPP2019 using_gbd2017_data_with_holdout_for_PV.R` and predictive validity summary metrics are computed by `bayesTFR_get_PV_results.R`. The run version is `/share/forecasting/data/5/future/tfr/20190826_WPP2019_fit_to_GBD2017_PV_10yr`.


4) checking MCMC convergence of the model estimation in 1)-3) using code in `bayesTFR_check_convergence.R`.

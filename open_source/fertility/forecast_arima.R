library(data.table)
library(forecast)

forecast_arima <- function(..., auto = F, forecast_start, forecast_end, num_draws = 1000, sigma_user = NULL){
  #' @description Wrapper for the forecast::Arima and auto.arima function with different formatted outputs.
  #' @param auto - Whether to automate selection of the ARIMA model parameters
  #' @param forecast_start - First year or cohort
  #' @param forecast_end - Last year or cohort
  #' @param num_draws - Number of draws to generate for each forecasted time period
  #' @param sigma_user - Input own starting sigma that will be used for the whole time series.
  #' @return data.table - columns for draw, time step, and draw value

  order <- list(...)[["order"]]

  if(!auto %in% c(T, F)){
    stop("auto argument must be boolean 'TRUE' or 'FALSE'")
  }
  if(forecast_end < forecast_start){
    stop("forecast_end must be greater than forecast_start")
  }
  if(num_draws <= 0){
    stop("num_draws must be greater than 0")
  }
  if(!is.null(sigma_user) & !all(order == c(0, 1, 0))){
    stop("Can only enforce sigma on AR(1) and ARIMA(0, 1, 0) random walk models")
  }

  if(auto) {
    arima_fit <- forecast::auto.arima(...)
  } else {
    arima_fit <- forecast::Arima(...)
  }

  # Do actual forecasts
  num_years <- forecast_end - forecast_start + 1
  forecasts <- forecast::forecast(arima_fit, h = num_years, level = 95)
  eps       <- data.table(h = 1:num_years,
                          mean_eps = forecasts$mean,
                          lower = as.numeric(forecasts$lower),
                          upper = as.numeric(forecasts$upper))
  # If there is an inputted sigma and fitting a RW model, execute this code. Can give it own sigma
  if(!is.null(sigma_user) & all(order == c(0, 1, 0))){
    eps[, sigma := sigma_user * sqrt(h)]
  } else{
    eps[, sigma := (upper - mean_eps) / 1.96]
  }
  eps <- eps[rep(1:.N, times = num_draws)][, draw := 0:(num_draws - 1), by = h]
  eps[, eps := mapply(rnorm, n = 1, mean = mean_eps, sd = sigma)]

  # Returns data.table of forecasts, the actual forecast model object
  return(list(mean_forecast = forecasts,
              draw_forecast = eps[,.(h, draw, eps)],
              model = arima_fit))
}

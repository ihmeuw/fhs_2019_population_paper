scaled_logit <- function(x, lower, upper, eps = 1e-6){
  x <- ifelse(x <= lower, lower + eps, x)
  x <- ifelse(x >= upper, upper - eps, x)
  log((x - lower) / (upper - x))
}


anti_scaled_logit <- function(x, lower, upper, bias_adj = F){
  if(bias_adj) {
    norm <- (upper - lower) * exp(x) / (1 + exp(x)) + lower
    norm <- norm - (mean(norm) - anti_scaled_logit(mean(x), lower, upper))
    norm
  } else {
    (upper - lower) * exp(x) / (1 + exp(x)) + lower
  }
}

library(data.table)
library(splines)

.extract_coefs_and_knots <- function(model){
  #' @description Returns a list of essential data for creating a design matrix for a lm object.
  #' @param model =- S3 lm model object. Does not work for random effects models
  has_intercept <- attr(model$terms, "intercept") == 1
  all_terms <- names(model$model)
  xcov      <- attr(model$terms, "term.labels") # Try to extract the actual variable from the ns call
  y         <- setdiff(all_terms, xcov)

  linear_vars <- xcov[!grepl("ns\\(|bs\\(", xcov)]
  spline_vars <- xcov[grepl("ns\\(|bs\\(", xcov)]

  # Extract spline type - natural or B spline usually
  spline_types <- lapply(spline_vars, function(s){attr(model$model[[s]], "class")[1]})
  # Extract internal and boundary knots
  internal_knots <- lapply(spline_vars, function(s){attr(model$model[[s]], "knots")})
  boundary_knots <- lapply(spline_vars, function(s){attr(model$model[[s]], "Boundary.knots")})

  # Reset the names of the spline variables to whatever is in between the ns( and the next comma
  spline_vars <- gsub(".*[nb]s\\(\\s*|,.*", "", spline_vars)
  names(spline_types)   <- spline_vars
  names(internal_knots) <- spline_vars
  names(boundary_knots) <- spline_vars

  # Return list
  list(linear_vars = linear_vars, spline_vars = spline_vars, spline_types = spline_types,
       internal_knots = internal_knots, boundary_knots = boundary_knots, has_intercept = has_intercept)
}

create_design_matrix <- function(model, dt){
  #' @description Creates a design matrix given a model object and data frame
  #' @param model - S3 lm model object
  #' @param dt - data frame
  #' @output design matrix

  model_data <- .extract_coefs_and_knots(model)
  b_matrix   <- matrix(ncol = 0, nrow = nrow(dt))
  x_matrix   <- matrix(ncol = 0, nrow = nrow(dt))

  # Check that all variables are present. Display warning
  all_vars <- c(model_data$linear_vars, model_data$spline_vars)
  lapply(all_vars, function(var) if(any(is.na(dt[, get(var)]))) warning(sprintf("%s is NA - may cause errors in spline basis bs() or ns()", var)))

  # Create spline part of the matrix
  for(term in model_data$spline_vars){
    if(model_data$spline_types[[term]] == "ns"){
      temp <- ns(dt[, get(term)], knots = model_data$internal_knots[[term]], Boundary.knots = model_data$boundary_knots[[term]])
    } else if(model_data$spline_types[[term]] == "bs"){
      temp <- bs(dt[, get(term)], knots = model_data$internal_knots[[term]], Boundary.knots = model_data$boundary_knots[[term]])
    }
    colnames(temp) <- paste0(term, 1:ncol(temp))
    b_matrix <- cbind(b_matrix, temp)
  }

  # Create linear part of the matrix
  for(term in model_data$linear_vars){
    x_matrix <- cbind(x_matrix, dt[, get(term)])
  }
  colnames(x_matrix) <- model_data$linear_vars

  design_matrix <- cbind(b_matrix, x_matrix)
  if(model_data$has_intercept) design_matrix <- cbind(1, design_matrix)
  return(design_matrix)
}

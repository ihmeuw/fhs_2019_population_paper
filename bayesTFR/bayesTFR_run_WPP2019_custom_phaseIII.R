######################################################
# Purpose: run the WPP2019 TFR model on all past WPP2019 data,
# with modified Phase III entrance criteria: TFR < 2.0
# author: Emily Goren
######################################################


library(data.table)
library(tidyverse)
library(mortdb, lib.loc = "/ihme/mortality/shared/r")
#install.packages("bayesTFR", lib = "/ihme/forecasting/shared/r", dependencies = TRUE)
library(bayesTFR, lib.loc = "/ihme/forecasting/shared/r")
#install.packages("wpp2019", lib = "/ihme/forecasting/shared/r", dependencies = TRUE, repos='http://cran.rstudio.com/')
library(wpp2019, lib.loc = "/ihme/forecasting/shared/r")

dir <- "/ihme/homes/egoren/repos/fertility_working/fbd_research/fbd_research/fertility/ccf/bayesTFR"
version <- "20190826_WPP2019_custom_phaseIII"


sim.dir <- sprintf("%s/%s/output", dir, version)

# FROM TH
#' Create own function for deciding where Phase III starts
my.phase3 <- function(tfr, T_end){
    #' @description My phase III function that changes definition of when Phase III begins depending on the tfr's decided upon from Phase II
    message("Using my Phase III function")
    lambda <- T_end
    is.below <- (tfr < 2)
    if (any(is.below))
        lambda <- which(is.below %in% TRUE)[1]
    return(lambda)
}
#' So that the package actually uses my new function, set new options..has print statements so
#' can verify it's actually being used.
options(TFRphase3findfct = my.phase3)

# Phase II MCMCs
g <- run.tfr.mcmc(
    output.dir = sim.dir,
    seed = 1,
    replace.output = TRUE,
    verbose = TRUE,
    start.year = 1950,
    present.year = 2018,
    wpp.year = 2019,
    iter = 50000)
# Phase III MCMCs (not included in the package)
g3 <- run.tfr3.mcmc(
    sim.dir = sim.dir,
    seed = 1,
    replace.output = TRUE,
    verbose = TRUE,
    iter = 75000)

# Continue MCMC if not converged
#g.cont <- continue.tfr.mcmc(
#    output.dir = sim.dir,
#    iter = 2000)



g <- get.tfr.mcmc(sim.dir)
g3 <- get.tfr3.mcmc(sim.dir)

included <- g3$meta$id_phase3
phaseIII_locs <- data.frame(country_name = g3$meta$regions$country_name[included], country_code = g3$meta$regions$country_code[included])
phaseIII_locs
fwrite(phaseIII_locs, sprintf("%s/%s/bayesTFR_phase3_locations.csv", dir, version))

# Prediction
pred <- tfr.predict(g, replace.output = TRUE)

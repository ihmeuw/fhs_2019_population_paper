######################################################
# Purpose: check MCMC convergence of a bayesTFR run
# which fits the WPP2019 fertility model
######################################################

library(data.table)
library(tidyverse)
library(mortdb, lib.loc = "/ihme/mortality/shared/r")
#install.packages("bayesTFR", lib = "/ihme/forecasting/shared/r", dependencies = TRUE)
library(bayesTFR, lib.loc = "/ihme/forecasting/shared/r")
#install.packages("wpp2019", lib = "/ihme/forecasting/shared/r", dependencies = TRUE, repos='http://cran.rstudio.com/')
library(wpp2019, lib.loc = "/ihme/forecasting/shared/r")
dir <- "/ihme/homes/egoren/repos/fertility_working/fbd_research/fbd_research/fertility/ccf/bayesTFR"
version <- "20190826_WPP2019"


sim.dir <- sprintf("%s/%s/output", dir, version)

# check convergence
m <- get.tfr.mcmc(sim.dir)
diag <- tfr.diagnose(sim.dir)
summary(diag)
diag3 <- tfr3.diagnose(sim.dir)
summary(diag3)
has.mcmc.converged(diag)
has.mcmc.converged(diag3)

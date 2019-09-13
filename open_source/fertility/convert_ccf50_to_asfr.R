# This code converts CCF to ASFR
# Gets called by a cluster qsub script

library(data.table)
library(magrittr)
library(argparse)
library(mortdb)
library(mortcore)
library(reticulate)


# Read in arguments from cluster qsub
parser <- ArgumentParser()
parser$add_argument(
  "--task_map_file",
  help = "name of task map",
  required = TRUE)
args <- parser$parse_args()
list2env(args, environment()); rm(args)

task_id             <- as.integer(Sys.getenv("SGE_TASK_ID"))
task_map            <- fread(task_map_file)
version             <- task_map[task_id, version]
lid                 <- task_map[task_id, location_id]
draws               <- task_map[task_id, draws]
forecast_start      <- task_map[task_id, forecast_start]
forecast_end        <- task_map[task_id, forecast_end]
version_dir         <- task_map[task_id, version_dir]
env                 <- task_map[task_id, env]
code_path           <- task_map[task_id, code_path]


setwd(code_path)
reticulate::use_condaenv(condaenv = env, conda = FILEPATH, required = T)
reticulate::source_python(sprintf("%s/get_rid_of_draws_scenarios_and_sex_id.py", code_path))
source(sprintf("%s/sort_draws.R", code_path))

message(sprintf("Location ID: %d", lid))
# Create own age map for single age
age_map    <- data.table(age_group_id = rep(8:14), age_start = seq(15, 49, 5))


########### MAIN #################################
main <- function(version, lid, draws, forecast_start, forecast_end, version_dir) {

  # Read in completely forecasted CCF 15-49 draws
  complete_draws      <- fread(sprintf("%s/complete/asfr/%s.csv", version_dir, lid))
  id_vars <- c("cohort", "scenario", "draw")
  setkeyv(complete_draws, id_vars)
  scenarios <- unique(complete_draws$scenario)

  # Check number of draws
  if (max(complete_draws$draw) < (draws-1)) {
    draws <- max(complete_draws$draw) + 1
    message(paste0("Not enough available draws in the proportional ASFR file. Setting number of draws to ", draws))
  } else {
    complete_draws <- complete_draws[draw < draws]
  }

  # Incomplete cohort draws
  # Read in GBD past cohort asfr data
  single_year_asfr <- fread(sprintf("%s/past_diff_data/single_year_asfr.csv", version_dir))[location_id == lid,.(location_id, year_id, cohort, age_start, age, mean)]
  setkeyv(single_year_asfr, c("location_id", "cohort", "age"))
  single_year_asfr[, ccf := cumsum(mean), by = .(location_id, cohort)]
  # Fertility by age
  single_year_asfr[,byage := age + 1]

  # Create incomplete cohort approximation data frame
  past_cohorts <- min(single_year_asfr$cohort):(forecast_start - 15L)
  approx_df <- CJ(location_id = lid, scenario = scenarios, cohort = past_cohorts, byage = 15:50, draw = 0:(draws - 1))
  approx_df[, age := byage - 1]
  approx_df <- merge(approx_df, single_year_asfr[,.(location_id, cohort, byage, age_start, mean, ccf)],
                     by = c("location_id", "cohort", "byage"), all.x = T, allow.cartesian = T)
  approx_df[byage==15,ccf := 0]

  # PASFR
  pasfr <- fread(sprintf("%s/incomplete/pasfr/pasfr_%s.csv", version_dir, lid))
  setnames(pasfr, "byage", "age")
  # Check number of draws
  if (max(pasfr$draw) < (draws-1)) {
    draws <- max(pasfr$draw) + 1
    message(paste0("Not enough available draws in the proportional ASFR file. Setting number of draws to ", draws))
  } else {
    pasfr <- pasfr[draw < draws]
  }

  # Pull in predicted CCF 5-yr step
  ccfx_complete_draws <- fread(sprintf("%s/ccfx/asfr/%s.csv", version_dir, lid))
  # Check number of draws
  if (max(ccfx_complete_draws$draw) < (draws-1)) {
    draws <- max(ccfx_complete_draws$draw) + 1
    message(paste0("Not enough available draws in the completed ccfx file. Setting number of draws to ", draws))
  } else {
    ccfx_complete_draws <- ccfx_complete_draws[draw < draws]
  }
  ccfx_complete_draws <- ccfx_complete_draws[rep(1:.N, times = 8)][, age := 15+5*0:(.N-1), by = .(cohort, scenario, draw)]
  # Looks like this sets ccf to 0 in 10-14 and ccf to that at age 50 in 45-49
  ccfx_complete_draws[age != 50, ccf := NA]
  ccfx_complete_draws[age == 15, ccf := 0]
  ccfx_complete_draws[, age_start := age - 5]
  # This just replicates out to single year but sets CCF45 to CCF49 to CCF50
  ccfx_df <- merge(ccfx_complete_draws, pasfr,  all.x = T)
  ccfx_df <- ccfx_df[rep(1:.N, each = 5)][, age := age + 0:(.N-1) - 5L, by = .(age,scenario, cohort, draw)]
  ccfx_df <- ccfx_df[age %in% 15:49]
  ccfx_df[, asfr := pasfr * max(ccf,na.rm=T) / 5, by = .(scenario, draw, cohort)]
  ccfx_df[, pasfr := NULL]
  ccfx_df[, year_id := cohort + age]
  age_map <- data.table(age_group_id = 8:14, age_start = seq(15, 49, 5))
  ccfx_df <- merge(ccfx_df, age_map)

  ccfx_df <- merge(ccfx_df, approx_df[, .(location_id, cohort, scenario, draw, age, mean)],
                   by = c("location_id", "cohort", "scenario", "draw", "age"), all.y = T)
  ccfx_df <- ccfx_df[age != 14]
  ccfx_df[is.na(mean), mean := asfr]
  ccfx_df[, asfr := mean]
  # Save single year for intercept shifting for last year of past
  ccfx_df_single_year <- ccfx_df[year_id >= (forecast_start - 1)]
  ccfx_df <- ccfx_df[,.(location_id, year_id, scenario,draw, age_group_id, asfr)]

  # Replicate out draws of CCF 15-49 for each 5-yr age group
  complete_draws <- complete_draws[rep(1:.N, times = 8)][, age := 15+5*0:(.N-1), by = .(cohort, scenario, draw)]
  complete_draws[age != 50, ccf := NA]
  complete_draws[age == 15, ccf := 0]
  complete_draws[, age_start := age - 5]

  df <- merge(complete_draws, pasfr,  all.x = T)
  df <- df[rep(1:.N, each = 5)][, age := age + 0:(.N-1) - 5L, by = .(age,scenario, cohort, draw)]
  df <- df[age %in% 15:49]
  df[, asfr := pasfr * max(ccf,na.rm=T) / 5, by = .(scenario, draw, cohort)]
  df[, pasfr := NULL]
  df[, year_id := cohort + age]
  age_map    <- data.table(age_group_id = 8:14, age_start = seq(15, 49, 5))
  df <- merge(df, age_map)
  final <- df[,.(location_id, year_id, scenario, draw, age_group_id, asfr)]

  # save single-year
  df_single_year <- df[year_id >= forecast_start & year_id <= forecast_end]
  out_single_year <- rbind(ccfx_df_single_year[,.(location_id, year_id, scenario, draw, age, asfr)],
                           df_single_year[,.(location_id, year_id, scenario, draw, age, asfr)])
  fwrite(out_single_year, sprintf("%s/asfr_single_year/%s.csv", version_dir, lid))

  # Bind incomplete cohorts to complete cohorts
  last <- rbind(ccfx_df, final)
  last <- last[, .(asfr = mean(asfr)), by = .(location_id, year_id, scenario, draw, age_group_id)]

  last <- last[year_id >= forecast_start & year_id <= forecast_end]
  last <- last[order(location_id, scenario, age_group_id, year_id, draw)]
  fwrite(last, sprintf("%s/asfr/%s.csv", version_dir, lid))
}

########## LAUNCH ################################
main(version = version,
     lid = lid,
     draws = draws,
     forecast_start = forecast_start,
     forecast_end = forecast_end,
     version_dir = version_dir)

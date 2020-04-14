# Purpose: From GBD ASFR Data, prep CCF50 and incremental fertility data

prep_ccf_diff <- function(forecast_start, cohort_age_start = 15, cohort_age_end = 49, gbd_year = 2017,
                          impute_remaining = TRUE, version_past_dir,
                          write_to_disk = TRUE, return_data = FALSE) {

  locs    <- mortdb::get_locations(gbd_year = gbd_year, level = "country")
  age_map <- mortdb::get_age_map() %>% setnames("age_group_years_start", "age_start")
  asfr    <- mortdb::get_mort_outputs(model_name = "asfr", model_type = "estimate", location_ids = locs$location_id, gbd_year = gbd_year)
  asfr <- asfr[,.(age_group_id, location_id, ihme_loc_id, year_id, mean, lower, upper)]
  asfr <- asfr[year_id < forecast_start]

  # Replicate out 5 year to 1 year
  DT <- asfr[rep(1:4), .(age_group_id, location_id, ihme_loc_id, year_id)][,add_yrs := 1:4]

  # Create single-year ASFR by replicating 5-yr ASFR for all ages - compute cohorts
  asfr <- merge(asfr, age_map[,.(age_group_id, age_start)], by = "age_group_id")
  asfr <- asfr[rep(1:.N,times = 5)][,age := age_start + 0:(.N-1), by = .(ihme_loc_id, year_id, age_group_id)]
  asfr[, cohort := year_id - age]

  asfr <- asfr[between(age, cohort_age_start, cohort_age_end)]
  asfr <- asfr[year_id < forecast_start]
  # Save data with cohorts that have single year ASFR's within completed five year age groups
  # Reorder ASFR and only keep cohorts that start from age 15
  setkeyv(asfr, c("ihme_loc_id", "cohort", "age"))
  asfr[,`:=`(minage = min(age),maxage = max(age)), by = .(ihme_loc_id, cohort)]
  asfr <- asfr[minage == cohort_age_start]
  asfr[, `:=`(minage = NULL)]

  # Fill in the rest of the ASFR data where we only have partial observation
  imputed_cohorts <- (forecast_start - cohort_age_start - 3):(forecast_start - cohort_age_start) - 1
  if (impute_remaining == T){
    young <- asfr[maxage < cohort_age_start+4]
    # Copy the remaining rows of cohort that need to be filled in
    template <- CJ(ihme_loc_id = unique(young$ihme_loc_id), cohort = imputed_cohorts, age = cohort_age_start:(cohort_age_start+4))
    final <- merge(template, young, by = c("cohort", "age", "ihme_loc_id"), all.x = T)
    final[, remain_asfr := mean(mean, na.rm = T), .(ihme_loc_id, cohort)]
    final[is.na(mean), mean := remain_asfr]
    final[, remain_asfr := NULL]
    final[, age_start := cohort_age_start]
    final[, maxage := cohort_age_end]
    final[, age_group_id := 8]
    final[, year_id := cohort + age]
    final[, location_id := as.integer(mean(location_id,na.rm=T)), .(ihme_loc_id)]
    asfr <- rbind(fsetdiff(asfr, asfr[cohort %in% imputed_cohorts]), final)
  }

  single_year_asfr <- asfr[,.(location_id, ihme_loc_id, age_group_id, age_start, age, year_id, cohort, mean, lower, upper)]
  if (write_to_disk) {
    message("Writing single_year_asfr to disk")
    fwrite(single_year_asfr, sprintf("%s/single_year_asfr.csv", version_past_dir))
  }
  asfr <- single_year_asfr[,total := .N,.(ihme_loc_id, cohort, age_group_id)]
  asfr <- asfr[total == 5 | cohort %in% imputed_cohorts]
  asfr[, ccf := cumsum(mean), .(ihme_loc_id, cohort)]
  asfr <- asfr[age %in% seq(cohort_age_start+4, cohort_age_end, 5)]
  asfr[, byage := paste0("ccf_", age + 1)]

  # Cast data wide to calculate differences easier
  wide_ccf <- dcast.data.table(asfr, ihme_loc_id + location_id + cohort ~ byage, value.var = c("ccf"))
  wide_ccf[, ccf_15 := 0]
  for (j in seq(20, 50, 5)){
    message(j)
    low <- j - 5
    wide_ccf[, paste0("diff_", j, "_", low) := get(paste0("ccf_",j)) - get(paste0("ccf_", low))]
    wide_ccf[, paste0("ln_", paste0("diff_", j, "_", low)) := log(get(paste0("diff_", j, "_", low)))]
  }

  # Melt back to long and create new variable for the age group differences
  ccf_diff <- melt.data.table(wide_ccf, id.vars = c("ihme_loc_id", "location_id", "cohort"),
                              measure = patterns("^ln_diff", "^diff"),
                              value.name = c("ln_diff", "diff"), variable.factor = F)
  ccf_diff[, variable := as.numeric(variable)]
  ccf_diff[, `:=`(diff_age_first = 5*variable+15,
                  diff_age_second = 5*variable+10)]
  ccf_diff[, diff_age_group := paste0(diff_age_first, "_minus_", diff_age_second)]
  ccf_diff[, lag_diff_age_group := paste0(diff_age_second, "_minus_", 5*variable+5)]

  # We are fitting the difference model with a lagged difference term - so need to add an additional column
  ccf_diff_lag <- copy(ccf_diff)[,variable := variable + 1] %>% setnames("ln_diff", "ln_lag")
  ccf_diff     <- merge(ccf_diff, ccf_diff_lag[,.(ihme_loc_id, cohort, variable, ln_lag)], by = c("ihme_loc_id", "cohort", "variable"), all.x = T)
  ccf_diff[, variable := NULL]
  ccf_diff     <- ccf_diff[, .(ihme_loc_id, location_id, cohort, diff_age_group, diff_age_first, diff_age_second, ln_diff, diff, lag_diff_age_group, ln_lag)]
  # Write to disk
  if (write_to_disk) {
    message("Writing ccf_diff to disk")
    fwrite(ccf_diff, sprintf("%s/ccf_diff_data.csv", version_past_dir))
  }
  if (return_data) {
    return(ccf_diff)
  }
}

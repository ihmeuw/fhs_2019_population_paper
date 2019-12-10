locs <- get_locations(gbd_year = 2017, level = "country")$location_id
source("/home/j/temp/central_comp/libraries/current/r/get_covariate_estimates.R")
prop_urban <- get_covariate_estimates(covariate_id = 854, 
                                      location_id = locs, gbd_round_id = 6, 
                                      decomp_step = "iterative")
prop_urban <- prop_urban[,.(location_id, location_name, year_id, mean_value)][year_id <= 2017]
setnames(prop_urban, "mean_value", "prop_urban")

#data[, prop_urban := prop_urban / 100]

# 
# ggplot(model_data[scenario == 0], aes(prop_urban, log(ccf))) + 
#   geom_point(alpha = 0.3) + 
#   geom_path(aes(group = location_id))  + 
#   ggtitle("UN Urbanicity against Log CCF")v

#fit_urban   <- lm(formula_urban, data = model_data)
#fit.random <- lmer(formula, data = model_data)

# tidy(fit)
# 
# coefs <- data.table(tidy(fit))[, type := "No Urbanicity"] %>% rbind(data.table(tidy(fit_urban))[, type := "UN Urbanicity"])
# fits <- data.table(glance(fit))[,type := "No Urbanicity"] %>% rbind(data.table(glance(fit_urban))[, type := "UN Urbanicity"])
# fits <- fits[, c("type", names(fits)), with = F]
# coefs <- coefs[, c("type", names(coefs)), with = F]
# 
# 
# old_coefs <- fread("/home/j/temp/twh42/urbanicity_coefs.csv")
# old_fits <- fread("/home/j/temp/twh42/urbanicity_fits.csv")
# old_coefs <- rbind(old_coefs, coefs[type == "UN Urbanicity"])
# old_fits <- rbind(old_fits, fits[type == "UN Urbanicity"])
# fwrite(old_fits, "/homes/twh42/fits_urban.csv")
# fwrite(old_coefs, "/homes//twh42/coefs_urban.csv")

# covariates <- merge(covariates, wup_data[,.(location_id, year_id, prop_urban)], by = c("location_id", "year_id"), all.x = T)[year_id <= 2017]
# 
# compare_urban <- merge(prop_urban, wup_data, by = c("location_id", "year_id"))
# setnames(compare_urban, c("prop_urban.x", "prop_urban.y"), c("ihme", "un"))
# compare_urban[, un := un / 100]
# ggplot(compare_urban, aes(ihme, un)) + 
#   geom_point(alpha = 0.3, aes(label = ihme_loc_id)) + 
#   geom_abline() + 
#   xlab("IHME Proportion Urban") + 
#   ylab("UN Proportion Urban") + 
#   ggtitle("Scatter of IHME and UN Urbanicity Estimates (1970-2017)")

#covariates <- merge(covariates, prop_urban[,.(location_id, year_id, prop_urban)], by = c("location_id", "year_id"), all.x = T)
# oast timeseries of sex ratio of birth
# for RTR

rm(list = ls())
library(mortdb, lib = "/home/forecasting/shared/r")
library(tidyverse, lib.loc = "/ihme/forecasting/shared/r")
theme_set(theme_bw(base_size = 18))

# Locations
source('/ihme/cc_resources/libraries/current/r/get_location_metadata.R')
location_map <- get_location_metadata(location_set_id = 39, gbd_round_id = 5)
location_order <- read.csv(
    "/ihme/homes/egoren/repos/fhs_miscellaneous/egoren/utils/tv_location_gbd2017_fshorder_224_20191127.csv")
location_map <- location_map %>% left_join(location_order)


# extract the data use get_mort_outputs
bsr <- get_mort_outputs(model_name = "birth sex ratio",
                        model_type = "estimate", gbd_year = 2017) %>%
    left_join(location_map) %>%
    filter(level == 3) %>%
    arrange(order)

# plot

locs <- subset(location_map, level == 3)$location_id
loc_chunks <- split(locs, cut_number(locs, 20))

pdf("/ihme/homes/egoren/scratch/sex_ratio_at_birth_timeseries.pdf",
    width = 30, height = 20)
for (j in 1:length(loc_chunks)) {
    dt <- subset(bsr, location_id %in% unlist(loc_chunks[j]))
    gg <- ggplot(dt) +
        geom_line(aes(x = year_id, y = mean)) +
        geom_ribbon(aes(x = year_id, ymax = upper, ymin = lower),
                    fill = "black", alpha = 0.3) +
        geom_hline(yintercept = 1.06, color = "gray") +
        labs(x = "Year", y = "Sex ratio at birth") +
        facet_wrap(~location_name, ncol = 5, scales = "fixed") +
        ylim(1, 1.25)
        #theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 10)) +
    print(gg)
}
dev.off()


bsr %>% filter(year_id %in% c(1990, 2017), location_name == "India")
bsr %>% filter(year_id %in% c(1990, 2017), location_name == "China")

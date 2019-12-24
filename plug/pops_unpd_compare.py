"""
Comparison to other models
Figure 9 shows a global comparison of the TFR, life expectancy at birth,
and population between our reference scenario, the UNPD median variant,
and the Wittgenstein SSP2 scenario. Differences between 2100 population
forecasts from the UNPD and our reference scenario are largely explained by
differences in population levels in sub-Saharan Africa; south Asia; southeast
Asia, east Asia, and Oceania (###·## million, ###·## million, and ###·##
million fewer people in our reference scenario, respectively)
"""

import pandas as pd

from db_queries import get_location_metadata
from fbd_core import YearRange, argparse
from fbd_core.file_interface import FBDPath, open_xr
from datetime import datetime


def return_pop_differences(fbd_pop_version, unpd_pop_version):
    GBD_LOC_DF = get_location_metadata(gbd_round_id=5, location_set_id=39)
    gbd_locs_df = GBD_LOC_DF.query("level < 4")

    fbdpoppath = FBDPath(f"/5/future/population/{fbd_pop_version}/"
                         "population_combined.nc")

    unpdpoppath = FBDPath(f"/wpp/future/population/{unpd_pop_version}/"
                          "population_all_age.nc")

    unpd_pop_xr = open_xr(unpdpoppath).data

    unpd_pop_100 = unpd_pop_xr.sel(year_id=2100,
                                   sex_id=3).drop(
        ["year_id", "sex_id", "age_group_id"]).squeeze().rename(
        "unpd_pop").to_dataframe().reset_index()

    ihme_pop_100 = open_xr(fbdpoppath).data.sel(
        age_group_id=22, sex_id=3, scenario=0, year_id=2100,
        quantile="mean").drop(
        ["age_group_id", "sex_id", "scenario", "year_id",
         "quantile"]).squeeze().rename("ihme_pop").to_dataframe().reset_index()

    ihme_pop_100["ihme_pop_int"] = ihme_pop_100["ihme_pop"].astype(int)
    unpd_pop_100["unpd_pop_int"] = unpd_pop_100["unpd_pop"].astype(int)

    pop_2100_df = ihme_pop_100.merge(unpd_pop_100, how="left")

    final_pop_df = pop_2100_df.merge(gbd_locs_df[["location_id",
                                                  "lancet_label"]])

    location_ids = [166, 158, 4]
    locs_of_interest = final_pop_df[
        final_pop_df.location_id.isin(location_ids)]
    new_pop_diff = {}
    for loc_id in location_ids:
        ihme_num = (locs_of_interest[locs_of_interest[
            "location_id"] == loc_id]["ihme_pop"].values[0])
        unpd_num = (locs_of_interest[locs_of_interest[
            "location_id"] == loc_id]["unpd_pop"].values[0])
        location_name = locs_of_interest[locs_of_interest[
            "location_id"] == loc_id].lancet_label.values[0]
        difference = ((unpd_num-ihme_num)/1e6).round(2)
        new_pop_diff[loc_id] = difference

    print(f"Figure 9 shows a global comparison of the TFR, life expectancy at"
          f" birth, and population between our reference scenario, the UNPD "
          f"median variant, and the Wittgenstein SSP2 scenario. Differences"
          f" between 2100 population forecasts from the UNPD and our reference"
          f" scenario are largely explained by differences in population"
          f" levels in sub-Saharan Africa; south Asia and ; southeast Asia,"
          f" east Asia, and Oceania ({new_pop_diff[location_ids[0]]} million, "
          f"{new_pop_diff[location_ids[1]]}  million, and"
          f" {new_pop_diff[location_ids[2]]} million fewer people in our"
          f" reference scenario, respectively)")


if __name__ == "__main__":
    fbd_pop_version = ("20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_"
                       "squeeze_shocks_hiv_all_combined")
    unpd_pop_version = "2019_fhs_agg"
    return_pop_differences(fbd_pop_version, unpd_pop_version)

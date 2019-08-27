"""
Comparing UNPD and our reference scenario in terms of cumulative births from
2017 to 2100, the largest differences by super-region are in sub-Saharan Africa (887·93
million); south Asia (628·97 million); and southeast Asia, east Asia, and Oceania (402·14
million). While the global difference in cumulative births is 2.16 billion, differences
in cumulative deaths are dramatically smaller, accounting for only 117·42 million deaths
globally from 2017 to 2100.
"""

import numpy as np
import pandas as pd
import xarray as xr

from db_queries import get_location_metadata
from fbd_core.etl import expand_dimensions, Aggregator
from fbd_core.file_interface import FBDPath, save_xr, open_xr


def return_ordered_cumulative_diff(measure, gbd_round_id, wpp_version, fbd_version):
    wpp_past_path = FBDPath(f"wpp/past/{measure}/{wpp_version}")
    wpp_future_path = FBDPath(f"wpp/future/{measure}/{wpp_version}")
    fbd_path = FBDPath(f"{gbd_round_id}/future/{measure}/{fbd_version}")

    wpp_past = open_xr(f"{wpp_past_path}/{measure}.nc").data
    wpp_future = open_xr(f"{wpp_future_path}/{measure}.nc").data

    wpp_future = wpp_future.sel(year_id=range(2020,2100))
    wpp_past = wpp_past.sel(year_id=range(2017,2020))
    combine_wpp_da = xr.concat([wpp_past, wpp_future], dim="year_id")
    wpp_agg = combine_wpp_da.sum("year_id")

    fbd_da = open_xr(f"{fbd_path}/{measure}.nc").data
    fbd_da = fbd_da.sel(year_id=range(2017,2100))
    fbd_agg = fbd_da.sum("age_group_id")
    fbd_agg = fbd_agg.sum("year_id")
    if measure == "death":
        fbd_agg = fbd_agg.sum("sex_id")
    fbd_agg = fbd_agg.mean("draw")

    if measure == "death":
        diff = fbd_agg - wpp_agg
    else: diff = wpp_agg - fbd_agg
    diff = diff.squeeze().drop("variable")
    output = diff.sortby(diff, ascending=False)

    return output[:5]

def get_location_name(location_id):
    loc_map = get_location_metadata(
        location_set_id=39, gbd_round_id=5).query(
            "level < 3")[["location_name","location_id"]]
    location_name_searched = loc_map[
        loc_map["location_id"]==int(location_id)].location_name.iloc[0]
    return location_name_searched

def print_sentences(birth_diffs, death_diffs):
    top_1_birth_value = (birth_diffs[1].values/1e6).round(2)
    top_2_birth_value = (birth_diffs[2].values/1e6).round(2)
    top_3_birth_value = (birth_diffs[4].values/1e6).round(2)
    top_1_birth_loc = get_location_name(birth_diffs[1].location_id.values)
    top_2_birth_loc = get_location_name(birth_diffs[2].location_id.values)
    top_3_birth_loc = get_location_name(birth_diffs[4].location_id.values)

    global_birth_value = (birth_diffs[0].values/1e9).round(2)
    global_death_value = (death_diffs[0].values/1e6).round(2)
    print(f"Comparing UNPD and our reference scenario in terms of cumulative "
          f"births from 2017 to 2100, the largest differences by super-region "
          f"are in {top_1_birth_loc} ({top_1_birth_value} million); "
          f"{top_2_birth_loc} ({top_2_birth_value} million); and "
          f"{top_3_birth_loc} ({top_3_birth_value} million). While the global "
          f"difference in cumulative births is {global_birth_value} billion, "
          f"differences in cumulative deaths are dramatically smaller, "
          f"accounting for only {global_death_value} million deaths globally "
          f"from 2017 to 2100.")


if __name__ == "__main__":
    birth_diffs = return_ordered_cumulative_diff(
        "live_births", 5, "20190822_annual_agg", "20190821_for_pop")
    death_diffs = return_ordered_cumulative_diff(
        "death", 5, "20190822_annual_agg", "20190821_for_pop_deaths")
    print_sentences(birth_diffs, death_diffs)


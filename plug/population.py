"""
Combining the scenarios for mortality, fertility, and migration, we expect
global population in the reference scenario to peak at 9·75 billion (95% UI
8·89–10·86) in the year 2063 and then decline to 8·72 billion (6·75–11·76) in
2100 (table 1, figure 6). Across alternative scenarios, the range in 2100 is
from 13·53 billion (10·68–17·62) in the slower met need and education scenario
to 6·20 billion (4·73–8·64) in the SDG-pace scenario for education and
contraceptive met need (table 1, figure 6). The faster and fastest scenarios
give 2100 global populations of 7·60 billion (5.81–10.38) and 6·79 billion
(5.17–9.41), respectively. Peak population in the SDG scenario occurs in 2046,
while the global population continues to grow through the century in the slower
scenario. The huge differences in TFR in 2100 across the scenarios translates
into differences of 7·33 billion (5.97–8.77) people in 2100. Figure 7 shows the
age-structure of the world population in 2017 and in the five scenarios for
2100. Huge shifts in the age structure are expected. Median age will increase
in the reference scenario from 32.48 in 2017 to 46.78 (42.47–47.47) in 2100.
The number of children under age 5 will decline from 680.65 million in 2017 to
only 402.08 million (251.33–706.11) in 2100, a drop of 40.93% (23.42–51.76). At
the same time, the number of individuals aged over 80 will increase from 141.09
million in 2017 to 781.13 million (531.93–1047.98) in 2100.
"""

import numpy as np
import pandas as pd
import xarray as xr

from db_queries import get_location_metadata
from fbd_core import db
from fbd_core.etl import Aggregator, expand_dimensions, weighted_quantile
from fbd_core.etl.scenarios import weighted_quantile_with_extra_dim
from fbd_core.file_interface import FBDPath, open_xr, save_xr

import settings


def find_peak_year(pop_da):
    pop_mean = pop_da.sel(quantile="mean")
    pop_ordered = pop_mean.sortby(pop_mean, ascending=False)
    peak_year = int(pop_ordered[0].year_id.values)
    return peak_year


def return_mean_and_quantiles(pop_da):
    pop_mean = pop_da.sel(quantile="mean").values.round(2)
    pop_lower = pop_da.sel(quantile="lower").values.round(2)
    pop_upper = pop_da.sel(quantile="upper").values.round(2)
    return pop_mean, pop_lower, pop_upper


def calculate_diff(highest_pop, lowest_pop):
    diff = highest_pop - lowest_pop
    diff_mean = diff.mean("draw").values.round(2)

    diff_quantile = diff.quantile([0.025, 0.975], dim="draw")
    diff_lower = diff_quantile.sel(quantile=0.025).values.round(2)
    diff_upper = diff_quantile.sel(quantile=0.975).values.round(2)
    return diff_mean, diff_lower, diff_upper


def get_median_age(pop_da, gbd_round_id, year_index):
    age_metadata = db.get_ages(gbd_round_id=gbd_round_id)
    age_group_ids = age_metadata["age_group_id"].tolist()

    days = age_metadata[["age_group_id", "age_group_days_start",
                         "age_group_days_end"]].set_index(
                                 "age_group_id").to_xarray()
    days["mean_age"] = (days["age_group_days_end"] - (
        days["age_group_days_end"] - days["age_group_days_start"]) / 2) / 365.25
    if year_index == 2100:
        pop_draw_base = pop_da.sel(scenario=0, sex_id=3, location_id=1,
                                   age_group_id=age_group_ids,
                                   year_id=year_index)
        median_age = weighted_quantile_with_extra_dim(days["mean_age"], 0.5,
                                                           ["age_group_id"],
                                                           pop_draw_base,
                                                           extra_dim="draw")
        median_age_mean = median_age.mean("draw").values.round(2)
        median_age_upper = median_age.quantile(0.975, dim="draw").values.round(2)
        median_age_lower = median_age.quantile(0.025, dim="draw").values.round(2)
        return median_age_mean, median_age_lower, median_age_upper
    else:
        pop_da_2017 = pop_da.sel(sex_id=3, location_id=1,
                                 age_group_id=age_group_ids,
                                 year_id=year_index)
        median_age = weighted_quantile(days["mean_age"], 0.5, ["age_group_id"],
                                       pop_da_2017)
        return median_age.values.round(2)


def print_pop_stats(pop_base_version, pop_sdg_version, pop_99_version,
                    pop_past_version, gbd_round_id, measure):

    pop_base_draw_path = FBDPath(
        f"{gbd_round_id}/future/{measure}/{pop_base_version}")
    pop_sdg_draw_path = FBDPath(
        f"{gbd_round_id}/future/{measure}/{pop_sdg_version}")
    pop_99_draw_path = FBDPath(
        f"{gbd_round_id}/future/{measure}/{pop_99_version}")
    pop_base_path = FBDPath(
        f"{gbd_round_id}/future/{measure}/{pop_base_version}_combined")
    pop_sdg_path = FBDPath(
        f"{gbd_round_id}/future/{measure}/{pop_sdg_version}_combined")
    pop_99_path = FBDPath(
        f"{gbd_round_id}/future/{measure}/{pop_99_version}_combined")
    pop_past_path = FBDPath(
        f"{gbd_round_id}/past/{measure}/{pop_past_version}")

    pop_past = open_xr(f"{pop_past_path}/{measure}_agg.nc").data
    pop_draw_base = open_xr(f"{pop_base_draw_path}/{measure}_agg.nc").data
    pop_draw_sdg = open_xr(f"{pop_sdg_draw_path}/{measure}_agg.nc").data
    pop_draw_99 = open_xr(f"{pop_99_draw_path}/{measure}_agg.nc").data
    pop_base = open_xr(f"{pop_base_path}/{measure}_combined.nc").data
    pop_sdg = open_xr(f"{pop_sdg_path}/{measure}_combined.nc").data
    pop_99 = open_xr(f"{pop_99_path}/{measure}_combined.nc").data

    pop_slower_2100 = pop_base.sel(
        scenario=-1, sex_id=3, location_id=1, age_group_id=22, year_id=2100)/1e9
    pop_sdg_2100 = pop_sdg.sel(
        scenario=-1, sex_id=3, location_id=1, age_group_id=22, year_id=2100)/1e9
    pop_faster_2100 = pop_base.sel(
        scenario=1, sex_id=3, location_id=1, age_group_id=22, year_id=2100)/1e9
    pop_fastest_2100 = pop_99.sel(
        scenario=1, sex_id=3, location_id=1, age_group_id=22, year_id=2100)/1e9
    pop_ref_2100 = pop_base.sel(
        scenario=0, sex_id=3, location_id=1, age_group_id=22, year_id=2100)/1e9
    pop_ref = pop_base.sel(
        scenario=0, sex_id=3, location_id=1, age_group_id=22)/1e9

    pop_sdg_draw_2100 = pop_draw_sdg.sel(scenario=-1, sex_id=3, location_id=1,
                                         age_group_id=22, year_id=2100)/1e9
    pop_slower_draw_2100 = pop_draw_base.sel(scenario=-1, sex_id=3,
                                             location_id=1, age_group_id=22,
                                             year_id=2100)/1e9
    pop_fastest_draw_2100 = pop_draw_99.sel(scenario=1, sex_id=3,
                                            location_id=1, age_group_id=22,
                                            year_id=2100)/1e9

    ref_peak_year = find_peak_year(pop_ref)
    ref_peak_da = pop_ref.sel(year_id=ref_peak_year)

    ref_peak_mean, ref_peak_lower, ref_peak_upper = \
    return_mean_and_quantiles(ref_peak_da)

    ref_2100_mean, ref_2100_lower, ref_2100_upper = \
    return_mean_and_quantiles(pop_ref_2100)

    slower_2100_mean, slower_2100_lower, slower_2100_upper = \
    return_mean_and_quantiles(pop_slower_2100)

    sdg_2100_mean, sdg_2100_lower, sdg_2100_upper = \
    return_mean_and_quantiles(pop_sdg_2100)

    faster_2100_mean, faster_2100_lower, faster_2100_upper = \
    return_mean_and_quantiles(pop_faster_2100)

    fastest_2100_mean, fastest_2100_lower, fastest_2100_upper = \
    return_mean_and_quantiles(pop_fastest_2100)

    if sdg_2100_mean < fastest_2100_mean:
        diff_mean, diff_lower, diff_upper = calculate_diff(pop_slower_draw_2100, pop_sdg_draw_2100)
    else:
        diff_mean, diff_lower, diff_upper = calculate_diff(pop_slower_draw_2100, pop_fastest_draw_2100)

    median_age_2017 = get_median_age(pop_past, gbd_round_id, 2017)
    median_age_2100_mean, median_age_2100_lower, median_age_2100_upper = \
    get_median_age(pop_draw_base, gbd_round_id, 2100)

    pop_under5_2017 = pop_past.sel(sex_id=3, location_id=1, age_group_id=1,
                                   year_id=2017)/1e6
    pop_under5_2100 = pop_base.sel(scenario=0, sex_id=3, location_id=1,
                                   age_group_id=1, year_id=2100)/1e6
    pop_over80_2017 = pop_past.sel(sex_id=3, location_id=1, age_group_id=21,
                                   year_id=2017)/1e6
    pop_over80_2100 = pop_base.sel(scenario=0, sex_id=3, location_id=1,
                                   age_group_id=21, year_id=2100)/1e6

    age_under5_2017 = pop_under5_2017.values.round(2)
    age_under5_2100_mean, age_under5_2100_lower, age_under5_2100_upper = \
    return_mean_and_quantiles(pop_under5_2100)
    age_over80_2017 = pop_over80_2017.values.round(2)
    age_over80_2100_mean, age_over80_2100_lower, age_over80_2100_upper = \
    return_mean_and_quantiles(pop_over80_2100)

    pop_under5_2017_draws = expand_dimensions(pop_under5_2017,
                                              draw=range(1000))
    pop_base_draw_under5_2100 = pop_draw_base.sel(scenario=0, sex_id=3,
                                                  location_id=1,
                                                  age_group_id=1,
                                                  year_id=2100).squeeze().drop("scenario")/1e6
    percent_drop = (pop_under5_2017_draws - pop_base_draw_under5_2100)/pop_under5_2017_draws
    drop_percent_mean = (percent_drop.mean("draw")*100).values.round(2)
    drop_percent_lower = (percent_drop.quantile(0.025,
                                                dim="draw")*100).values.round(2)
    drop_percent_upper = (percent_drop.quantile(0.975,
                                                dim="draw")*100).values.round(2)

    print(f"Combining the scenarios for mortality, fertility, and migration, "
          f"we expect global population in the reference scenario to peak at "
          f"{ref_peak_mean} (95% UI {ref_peak_lower}-{ref_peak_upper}) billion "
          f"in the year {ref_peak_year} and then decline to {ref_2100_mean} "
          f"({ref_2100_lower}-{ref_2100_upper}) billion in 2100. "
          f"Across alternative scenarios, the range in 2100 is from "
          f"{slower_2100_mean} billion "
          f"({slower_2100_lower}–{slower_2100_upper}) in the slower met need "
          f"and education scenario to {sdg_2100_mean} billion "
          f"({sdg_2100_lower}-{sdg_2100_upper}) in the SDG-pace scenario for "
          f"education and contraceptive met need (figure 6)."
          f"The faster and fastest scenarios give 2100 global populations of "
          f"{faster_2100_mean} ({faster_2100_lower}-{faster_2100_upper}) and "
          f"{fastest_2100_mean} ({fastest_2100_lower}-{fastest_2100_upper}) "
          f"billion, respectively. Peak population in the SDG scenario is in "
          f"2046, while the global population continues to grow through the "
          f"century in the slower scenario. The huge differences in TFR in "
          f"2100 across the scenarios translates into differences of {diff_mean} "
          f"({diff_lower}-{diff_upper}) billion people in 2100. "
          f"Median age will increase in the reference scenario from "
          f"{median_age_2017} in 2017 to {median_age_2100_mean} "
          f"({median_age_2100_lower}-{median_age_2100_upper}) in 2100 "
          f"The number of children under age 5 will decline from {age_under5_2017} "
          f"million in 2017 to only {age_under5_2100_mean} "
          f"({age_under5_2100_lower}-{age_under5_2100_upper}) in 2100, a drop "
          f"of {drop_percent_mean}% "
          f"({drop_percent_lower}-{drop_percent_upper}) million. At the same "
          f"time, the number of individuals aged over 80 will will increase "
          f"from {age_over80_2017} million in 2017 to {age_over80_2100_mean} "
          f"({age_over80_2100_lower}-{age_over80_2100_upper}) in 2100.")


if __name__ == "__main__":

    pop_past_version= settings.PAST_VERSIONS["population"].version
    pop_base_version = settings.BASELINE_VERSIONS["population"].version
    pop_99_version = settings.FASTEST_SCENARIO_VERSIONS["population"].version
    pop_sdg_version = settings.SDG_SCENARIO_VERSIONS["population"].version

    print_pop_stats(pop_base_version, pop_sdg_version, pop_99_version,
                    pop_past_version, 5, "population")

"""
    Results - Regional and country-level mortality:

    Both global and super-region life expectancy forecasts show future slowdown,
    particularly in the latter half of the century, compared to the recent past
    from 1990 to 2017. The slowdown is particularly notable in the XX; XX; XX;
    and XX super-regions. The slowdown is less in the XX; XX; and XX
    super-regions. Overall, the pattern is one of global convergence towards the
    end of the century. Among the ten countries with the largest populations in
    2017 or 2100, XX, XX, XX, XX, XX, and XX have the highest life expectancies
    in 2100 according to the reference forecast, ranging from XX·X
    (95% UI XX·X–XX·X) to XX·X (XX·X–XX·X) years. XX, XX, XX, and XX will have
    the lowest life expectancies among these ten large countries, from XX·X
    (XX·X–XX·X) to XX·X (XX·X–XX·X) years. See appendix 2 (section 3) for
    additional results. 
"""

import xarray as xr
import pandas as pd
import sys

from db_queries import get_location_metadata
from fbd_core.file_interface import FBDPath, open_xr, save_xr
from fbd_core.etl import compute_summaries, expand_dimensions

import settings as sett

LOCS = get_location_metadata(location_set_id=35, gbd_round_id=5)
SUPER_REGS = LOCS[LOCS.level==1]
NATS = LOCS[LOCS.level==3]

lex_past_vers = sett.PAST_VERSIONS["lex"].version
lex_past_dir = "/5/past/life_expectancy/"
lex_past_path = FBDPath(lex_past_dir + lex_past_vers)

lex_fut_vers = sett.BASELINE_VERSIONS["lex"].version
lex_fut_dir = "/5/future/life_expectancy/"
lex_fut_path = FBDPath(lex_fut_dir + lex_fut_vers)

pop_past_vers = sett.PAST_VERSIONS["population"].version
pop_past_dir = "/5/past/population/"
pop_past_path = FBDPath(pop_past_dir + pop_past_vers)

pop_fut_vers = sett.BASELINE_VERSIONS["population_mean_ui"].version
pop_fut_dir = "/5/future/population/"
pop_fut_path = FBDPath(pop_fut_dir + pop_fut_vers)

lex_past = open_xr(lex_past_path / "lifetable_ds.nc").data["ex"].sel(
    year_id=range(1990, 2018), sex_id=3, age_group_id=2)

lex_fut = open_xr(lex_fut_path / "lifetable_ds_agg.nc").data["ex"].sel(
    year_id=range(2018, 2101), sex_id=3, age_group_id=2, scenario=0)

lex = lex_past.combine_first(lex_fut).drop(
    ["sex_id", "age_group_id", "scenario"]).squeeze()

lex_mean_ui = lex.rename("value").to_dataset()
compute_summaries(lex_mean_ui)
lex_mean_ui = lex_mean_ui[["mean", "upper", "lower"]].drop("quantile").squeeze()
lex_mean = lex_mean_ui["mean"]

pop_past = open_xr(pop_past_path / "population_agg.nc").data.sel(
    year_id=range(1990, 2018), sex_id=3, age_group_id=22)
pop_past = expand_dimensions(pop_past, quantile = ["mean", "lower", "upper"])

pop_fut = open_xr(pop_fut_path / "population_combined.nc").data.sel(
    year_id=range(2018, 2101), sex_id=3, age_group_id=22, scenario=0)

pop = pop_past.combine_first(pop_fut).sel(location_id=NATS.location_id.tolist())


def get_rate_of_change(ex, year_start, year_end):
    # Get ARC

    lex_start = ex.sel(year_id=year_start).drop("year_id").squeeze()
    lex_end = ex.sel(year_id=year_end).drop("year_id").squeeze()
    rate_change = (lex_start - lex_end)/(year_end-year_start)
    return rate_change


def print_ratio_arc(ex_da, past_start, past_end, fut_start, fut_end):
    # Print ratio of future ARC to past ARC sorted by ratio

    rate_change_past = get_rate_of_change(ex_da, past_start, past_end)
    rate_change_fut = get_rate_of_change(ex_da, fut_start, fut_end)
    
    ratio_rate_change = (rate_change_fut / rate_change_past).rename(
        "ratio").to_dataframe().reset_index()
    rat_sup_reg = ratio_rate_change.merge(
        SUPER_REGS[["location_id", "location_ascii_name"]]
    ).sort_values(by="ratio")
    
    print("Super-regions ordered by ex slowdown" +
          f" (ARC ratio, {fut_start}-{fut_end}:{past_start}-{past_end})")
    print(rat_sup_reg)


def get_highest_pop_locs_past_fut(pops, past_year, fut_year, num_locs):
    # Get num_locs highest locs in past and future and combine

    highest_past = set(pops.sel(
        year_id=past_year,
        quantile="mean").to_dataframe().reset_index().nlargest(
        num_locs, "population").location_id)
    highest_fut = set(pops.sel(
        year_id=fut_year,
        quantile="mean").to_dataframe().reset_index().nlargest(
        num_locs, "population").location_id)
    highest = list(highest_past | highest_fut)
    return highest


def print_lex_rank(pop, lex, past_year, fut_year, num_locs):
    """
    Get ex in 2100. Locations chosen by combining n countries with highest
    population in past_year with n countries with highest pop in fut_year.
    """
    highest = get_highest_pop_locs_past_fut(pop, past_year, fut_year, num_locs)
    lex_high_pop = lex.sel(
        location_id=highest,
        year_id=[past_year, fut_year]).to_dataframe().reset_index()
    lex_rank = lex_high_pop.query(
        f"year_id=={fut_year}").sort_values(
        by="mean").merge(NATS[["location_id", "location_ascii_name"]])
    print("\nex in countries with highest pop in" +
        f" {past_year} or {fut_year} sorted")
    print(lex_rank)


if __name__ == "__main__":
    """
    "Both global and super-region life expectancy forecasts show future
    slowdown, particularly in the latter half of the century, compared to the
    recent past from 1990 to 2017."

    We're comparing ARC in the latter half of the 21st century (2051-2100) to
    the recent past (1990-2017). Thus, those years are reflected in the args
    in print_ratio_arc.
    """
    print_ratio_arc(lex_mean, 1990, 2017, 2051, 2100)

    """
    "Among the ten countries with the largest populations in
    2017 or 2100..."

    Though this may sound like we want the combinations of the 10 highest
    population countries in 2017 and the 10 highest population countries in
    2100, we actually want 10 countries total. To do this, we combine the 8
    largest countries in 2017 and the 8 largest countries in 2100.
    """
    print_lex_rank(pop, lex_mean_ui, 2017, 2100, 8)

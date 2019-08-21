'''
Figure 3 shows the evolution of global life expectancy in the reference,
slower, faster, fastest and SDG pace contraceptive met need and education
scenarios. While life expectancy is expected to increase,
the rate of progress is likely to slow. Large inequalities remain at the
global level in 2100 with the country life expectancies ranging from
69 to 94 years in the reference scenario. The standard deviation of life
expectancy across countries narrows from 6.9 years in 2017 to 3.9 years in
2100. Nine countries in 2100 still have life expectancies below 75 years,
seven of them in sub-Saharan Africa. The range of global life expectancy in
2100 across scenarios is moderate (from 79.5 years in the slower scenario to
81.1 in the fastest and SDG pace scenarios).

country ref life exp min and max
std dev of countries
countries with life exp below 75
range of global lex across scenarios

'''

import xarray as xr
import pandas as pd

from fbd_core import db
from fbd_core.file_interface import FBDPath, open_xr
from fbd_core.etl import expand_dimensions

import settings


gbd_round_id = settings.BASELINE_VERSIONS["lex"].gbd_round_id

# lex versions
PAST = settings.PAST_VERSIONS["lex"].version
REF = settings.BASELINE_VERSIONS["lex"].version
ALT_99 = settings.FASTEST_SCENARIO_VERSIONS["lex"].version
ALT_SDG = settings.SDG_SCENARIO_VERSIONS["lex"].version


def lex_min_max(lex_ref_version):
    min_val = lex_ref_version.sel(scenario=0,
                              year_id=2100,
                              sex_id=3,
                              age_group_id=2).min().values

    max_val = lex_ref_version.sel(scenario=0,
                              year_id=2100,
                              sex_id=3,
                              age_group_id=2).max().values

    print_statement = f"Country life expectancy in 2100 for ref scenario " \
        f"range from {min_val.round(1)} to {max_val.round(1)}"

    print(print_statement)

    return min_val, max_val


def std_dev_year(lex_ref_version, lex_past_version):

    std_dev_2017 = lex_past_version.sel(year_id=2017,
                                         sex_id=3,
                                         age_group_id=2).std()


    std_dev_2100 = lex_ref_version.sel(year_id=2100,
                                       scenario=0,
                                       sex_id=3,
                                       age_group_id=2)\
                                  .std()

    print_statement = f"Standard deviation in 2017: "\
        f"{std_dev_2017.values.round(1)}," \
        f"Standard deviation in 2100: {std_dev_2100.values.round(1)}"

    print(print_statement)

    return std_dev_2017, std_dev_2100


def low_lex_countries(lex_ref_version):

    lex_df = lex_ref_version.sel(sex_id=3, scenario=0, year_id=2100,
                                 age_group_id=2).rename(
        "lex").to_dataframe().reset_index()
    lex_under_75_df = lex_df.query("lex < 75")

    location_metadata = db.get_locations_by_max_level(3)[
        ["location_id", "location_name", "region_name"]]
    lex_location_verbose_df = lex_under_75_df.merge(location_metadata)
    assert len(lex_location_verbose_df) == len(lex_under_75_df)

    print_statement = f"Country regions with life expectancy less than 75:" \
    f"{lex_location_verbose_df.region_name.value_counts()}"

    print(print_statement)

    return lex_location_verbose_df


def range_of_lex(lex_ref, lex_99, alt_sdg):
    slower = lex_ref.mean('draw').values
    slower_lower = lex_ref.quantile(0.025, dim="draw")
    slower_upper = lex_ref.quantile(0.95, dim="draw")

    fastest = lex_99.mean('draw').values
    fastest_lower = lex_99.quantile(0.025, dim="draw")
    fastest_upper = lex_99.quantile(0.95, dim="draw")

    sdg = alt_sdg.mean('draw').values
    sdg_lower = alt_sdg.quantile(0.025, dim="draw")
    sdg_upper = alt_sdg.quantile(0.95, dim="draw")

    print_statement = f"2100 global life exp in slower scenario:"\
        f"{slower.round(1)} with UI: {slower_lower.values.round(1)},"\
        f"{slower_upper.values.round(1)}." \
        f" 2100 global life exp in fastest scenario:"\
        f" {fastest.round(1)} with UI:{fastest_lower.values.round(1)}," \
        f" {fastest_upper.values.round(1)}." \
        f" 2100 global life exp in sdg scenario: {sdg.round(1)} "\
        f" with UI: {sdg_lower.values.round(1)}, {sdg_upper.values.round(1)}."

    print(print_statement)

    return slower, fastest, sdg


if __name__ == "__main__":

    lex_ref_file = FBDPath(
        f"/{gbd_round_id}/future/life_expectancy/{REF}/lex_agg.nc")
    lex_ref = open_xr(lex_ref_file).data
    lex_ref_mean = open_xr(lex_ref_file).data.mean('draw')

    lex_past_file = FBDPath(
        f"/{gbd_round_id}/past/life_expectancy/{PAST}/lex.nc")
    lex_past = open_xr(lex_past_file).data

    alt_99 = open_xr(FBDPath(
        f"/{gbd_round_id}/future/life_expectancy/{ALT_99}/lex_agg.nc")).data

    alt_sdg = open_xr(FBDPath(
        f"/{gbd_round_id}/future/life_expectancy/{ALT_SDG}/lex_agg.nc")).data

    min_val, max_val = lex_min_max(lex_ref_mean)
    std_dev_2017, std_dev_2100 = std_dev_year(lex_ref_mean,
                                              lex_past)
    low_lex = low_lex_countries(lex_ref_mean)

    lex_ref_2100 = lex_ref.sel(year_id=2100,
                         location_id=1,
                         sex_id=3,
                         age_group_id=2,
                         scenario=-1)
    alt_99_2100 = alt_99.sel(year_id=2100,
                         location_id=1,
                         sex_id=3,
                         age_group_id=2,
                         scenario=-1)
    alt_sdg_2100 = alt_sdg.sel(year_id=2100,
                         location_id=1,
                         sex_id=3,
                         age_group_id=2,
                         scenario=-1)
    slower, fastest, sdg = range_of_lex(lex_ref_2100, alt_99_2100, alt_sdg_2100)
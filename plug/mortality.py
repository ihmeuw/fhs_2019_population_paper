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
        f"range from {min_val} to {max_val}"

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

    print_statement = f"Standard deviation in 2017: {std_dev_2017.values}," \
        f"Standard deviation in 2100: {std_dev_2100.values}"

    print(print_statement)

    return std_dev_2017, std_dev_2100


def low_lex_countries(lex_ref_version):
    t = lex_ref_version.sel(sex_id=3, scenario=0, year_id=2100, age_group_id=2)
    cond = lex_ref_version.sel(sex_id=3, scenario=0, year_id=2100,
                               age_group_id=2).values < 75
    locations_under_75 = t.where(
        cond).to_dataframe().reset_index().dropna().location_id

    location_metadata = db.get_locations_by_max_level(3)

    lm = location_metadata[["location_id", "location_name", "region_name"]]

    tt = pd.merge(locations_under_75, lm)

    print_statement = f"Country regions the life expectancy less than 75: {tt.region_name.value_counts()}"

    print(print_statement)

    return tt


def range_of_lex(lex_ref, lex_99, alt_sdg):
    slower = lex_ref.sel(year_id=2100,
                         location_id=1,
                         sex_id=3,
                         age_group_id=2,
                         scenario=-1).values
    fastest = lex_99.sel(year_id=2100,
                         location_id=1,
                         sex_id=3,
                         age_group_id=2,
                         scenario=1).values

    sdg = alt_sdg.sel(year_id=2100,
                         location_id=1,
                         sex_id=3,
                         age_group_id=2,
                         scenario=-1).values

    print_statement = f"2100 global life exp in slower scenario: {slower}." \
        f" 2100 global life exp in fastest scenario: {fastest}." \
        f" 2100 global life exp in sdg scenario: {sdg}."
    print(print_statement)

    return slower, fastest, sdg


if __name__ == "__main__":

    lex_ref_file = FBDPath(
        f"/{gbd_round_id}/future/life_expectancy/{REF}/lex_agg.nc")
    lex_ref = open_xr(lex_ref_file).data.mean("draw")

    lex_past_file = FBDPath(
        f"/{gbd_round_id}/past/life_expectancy/{PAST}/lex.nc")
    lex_past = open_xr(lex_past_file).data

    alt_99 = open_xr(FBDPath(
        f"/{gbd_round_id}/future/life_expectancy/{ALT_99}/lex_agg.nc")).data.mean(
        'draw')

    alt_sdg = open_xr(FBDPath(
        f"/{gbd_round_id}/future/life_expectancy/{ALT_SDG}/lex_agg.nc")).data.mean(
        'draw')

    min_val, max_val = lex_min_max(lex_ref)
    std_dev_2017, std_dev_2100 = std_dev_year(lex_ref,
                                              lex_past)
    low_lex = low_lex_countries(lex_ref)

    slower, fastest, sdg = range_of_lex(lex_ref, alt_99, alt_sdg)
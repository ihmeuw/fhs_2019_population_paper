"""
The global total fertility rate (TFR) in 2100 is forecasted to be 1.66
(95% UI xx–xx).  In the reference forecast, the global population is projected
to peak in 2063 at 9.7 (UI)  billion people, and decline to 8.7 (UI) in 2100.
The reference projections fore the 5 five largest countries in 2100 are India
(1067 million ([UI), ], Nigeria (791 million[(UI])), China (716 million([UI])),
USA USA (335 million [(UI)]), and DR Congo (246 million [UI]). Findings also
suggest a shifting age structure in many parts of the world, with XX individuals
above the age of 65,  and XX individuals below the age of 20, globally in 2100.
By 2050, XX countries will have a TFR below replacement and XX below
replacement by 2100.  XX countries including China, XX, XX and XX in the
reference scenario will have declines of greater than 50% from 2017 by 2100.
Alternative scenarios suggest meeting the SDG targets for education and
contraceptive met need will result in a global population of 6.2 billion in
2100, and 6.8 billion assuming 99th percentile rates of change in educational
attainment and met need for contraception. Forecasts from this study differ
from the UNPD and the Wittgenstein Centre, which project 10.9 billion and 9.3
billion people globally in 2100, respectively. The difference with UNPD in 2100
is due to XX million fewer people in the reference scenario in sub-Saharan
Africa and XX million in other regions, primarily due to the level of fertility
achieved in below replacement populations; Chine will decline by XX. China is
expected to become the largest economy by XX but in the reference scenario the
USA would once again become in the largest economy in XX.
"""
import numpy as np
import pandas as pd
import xarray as xr

from fbd_core import argparse, db, great_job
from fbd_core import great_job
from fbd_core.etl import resample
from fbd_core.etl.aggregator import Aggregator
from fbd_core.etl.transformation import expand_dimensions
from fbd_core.file_interface import FBDPath, open_xr, save_xr

import sys
sys.path.append(".")

import settings

ALL_AGE_ID = 22
BOTH_SEX_ID = 3

N_DECIMALS = 2

COUNTRIES = db.get_locations_by_level(3).location_id.values.tolist()


def _helper_round(val, divide_by_this):
    return round(val/divide_by_this, N_DECIMALS)

def _location_id_to_name(location_ids):
    locs = db.get_locations_by_max_level(3)[['location_id', 'location_name']]
    locs = locs[locs['location_id'].isin(location_ids)]
    location_names = locs['location_name'].tolist()

    return location_names


# The global total fertility rate (TFR) in 2100 is forecasted to be 1.66
# (95% UI xx–xx).
def tfr_2100():
    tfr_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['tfr_mean_ui'].gbd_round_id}/"
        f"future/tfr/"
        f"{settings.BASELINE_VERSIONS['tfr_mean_ui'].version}")
    tfr_path = tfr_dir / "tfr_combined.nc"
    tfr_da = open_xr(tfr_path).data.sel(scenario=0, location_id=1, year_id=2100)

    tfr_val = tfr_da.sel(quantile="mean").values.item(0)
    tfr_upper = tfr_da.sel(quantile="upper").values.item(0)
    tfr_lower = tfr_da.sel(quantile="lower").values.item(0)

    tfr_val = _helper_round(tfr_val, 1)
    tfr_upper = _helper_round(tfr_upper, 1)
    tfr_lower = _helper_round(tfr_lower, 1)

    print(f"The global total fertility rate (TFR) in 2100 is forecasted to be "
          f"{tfr_val} (95% UI {tfr_lower}–{tfr_upper}).\n")

# In the reference forecast, the global population is projected
# to peak in 2063 at 9.7 (UI)  billion people, and decline to 8.7 (UI) in 2100.
def pop_peak():
    pop_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['population_mean_ui'].gbd_round_id}/"
        f"future/population/"
        f"{settings.BASELINE_VERSIONS['population_mean_ui'].version}")
    pop_path = pop_dir / "population_combined.nc"
    pop_da = open_xr(pop_path).data.sel(
        location_id=1, age_group_id=ALL_AGE_ID, sex_id=BOTH_SEX_ID, scenario=0)

    # Max values
    max_da = pop_da.sel(quantile="mean")
    max_val = max_da.max().values.item(0)
    max_year = max_da.where(max_da==max_val, drop=True).year_id.values[0]
    max_upper = pop_da.sel(quantile="upper", year_id=max_year).values.item(0)
    max_lower = pop_da.sel(quantile="lower", year_id=max_year).values.item(0)

    # Values in 2100
    end_val = pop_da.sel(quantile="mean", year_id=2100).values.item(0)
    end_upper = pop_da.sel(quantile="upper", year_id=2100).values.item(0)
    end_lower = pop_da.sel(quantile="lower", year_id=2100).values.item(0)

    max_val = _helper_round(max_val, 1e9)
    max_upper = _helper_round(max_upper, 1e9)
    max_lower = _helper_round(max_lower, 1e9)
    end_val = _helper_round(end_val, 1e9)
    end_upper = _helper_round(end_upper, 1e9)
    end_lower = _helper_round(end_lower, 1e9)


    print(f"In the reference forecast, the global population is projected to "
          f"peak in {max_year} at {max_val} "
          f"({max_lower}-{max_upper}) billion people, and decline to "
          f"{end_val} ({end_lower}-{end_upper}) in 2100."
          f"\n")


# The reference projections for the 5 five largest countries in 2100 are India
# (1067 million ([UI), ], Nigeria (791 million[(UI])), China (716 million([UI]))
# USA USA (335 million [(UI)]), and DR Congo (246 million [UI])
def most_populated_2100():
    pop_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['population_mean_ui'].gbd_round_id}/"
        f"future/population/"
        f"{settings.BASELINE_VERSIONS['population_mean_ui'].version}")
    pop_path = pop_dir / "population_combined.nc"
    pop_da = open_xr(pop_path).data.sel(
        age_group_id=ALL_AGE_ID, sex_id=BOTH_SEX_ID, scenario=0, year_id=2100)

    top_5_locs = pop_da.sel(quantile="mean",location_id=COUNTRIES).\
        to_dataframe().reset_index().sort_values("population", ascending=False).\
        head(5)["location_id"].to_list()

    first_loc = _location_id_to_name([top_5_locs[0]])
    first_val = pop_da.sel(
        location_id=top_5_locs[0], quantile="mean").values.item(0)
    first_upper = pop_da.sel(
        location_id=top_5_locs[0], quantile="upper").values.item(0)
    first_lower = pop_da.sel(
        location_id=top_5_locs[0], quantile="lower").values.item(0)

    second_loc = _location_id_to_name([top_5_locs[1]])
    second_val = pop_da.sel(
        location_id=top_5_locs[1], quantile="mean").values.item(0)
    second_upper = pop_da.sel(
        location_id=top_5_locs[1], quantile="upper").values.item(0)
    second_lower = pop_da.sel(
        location_id=top_5_locs[1], quantile="lower").values.item(0)

    third_loc = _location_id_to_name([top_5_locs[2]])
    third_val = pop_da.sel(
        location_id=top_5_locs[2], quantile="mean").values.item(0)
    third_upper = pop_da.sel(
        location_id=top_5_locs[2], quantile="upper").values.item(0)
    third_lower = pop_da.sel(
        location_id=top_5_locs[2], quantile="lower").values.item(0)

    fourth_loc = _location_id_to_name([top_5_locs[3]])
    fourth_val = pop_da.sel(
        location_id=top_5_locs[3], quantile="mean").values.item(0)
    fourth_upper = pop_da.sel(
        location_id=top_5_locs[3], quantile="upper").values.item(0)
    fourth_lower = pop_da.sel(
        location_id=top_5_locs[3], quantile="lower").values.item(0)

    fifth_loc = _location_id_to_name([top_5_locs[4]])
    fifth_val = pop_da.sel(
        location_id=top_5_locs[4], quantile="mean").values.item(0)
    fifth_upper = pop_da.sel(
        location_id=top_5_locs[4], quantile="upper").values.item(0)
    fifth_lower = pop_da.sel(
        location_id=top_5_locs[4], quantile="lower").values.item(0)

    # round
    first_val = _helper_round(first_val, 1e6)
    first_upper = _helper_round(first_upper, 1e6)
    first_lower = _helper_round(first_lower, 1e6)

    second_val = _helper_round(second_val, 1e6)
    second_upper = _helper_round(second_upper, 1e6)
    second_lower = _helper_round(second_lower, 1e6)

    third_val = _helper_round(third_val, 1e6)
    third_upper = _helper_round(third_upper, 1e6)
    third_lower = _helper_round(third_lower, 1e6)

    fourth_val = _helper_round(fourth_val, 1e6)
    fourth_upper = _helper_round(fourth_upper, 1e6)
    fourth_lower = _helper_round(fourth_lower, 1e6)

    fifth_val = _helper_round(fifth_val, 1e6)
    fifth_upper = _helper_round(fifth_upper, 1e6)
    fifth_lower = _helper_round(fifth_lower, 1e6)

    print(f"The reference projections for the 5 five largest countries in 2100 "
        f"are {first_loc} ({first_val} million [{first_lower}-{first_upper}]), "
        f"{second_loc} ({second_val} million [{second_lower}-{second_upper}]), "
        f"{third_loc} ({third_val} million [{third_lower}-{third_upper}]), "
        f"{fourth_loc} ({fourth_val} million [{fourth_lower}-{fourth_upper}]), and "
        f"{fifth_loc} ({fifth_val} million [{fifth_lower}-{fifth_upper}])."
        f"\n")


# Findings also suggest a shifting age structure in many parts of the world,
# with XX individuals above the age of 65,  and XX individuals below the age of
# 20, globally in 2100.
def age_pops():
    pop_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['population_mean_ui'].gbd_round_id}/"
        f"future/population/"
        f"{settings.BASELINE_VERSIONS['population_mean_ui'].version}")
    pop_path = pop_dir / "population_combined.nc"
    pop_da = open_xr(pop_path).data.sel(
        sex_id=BOTH_SEX_ID, scenario=0, year_id=2100, quantile="mean",
        location_id=1)

    # 65 to 69, 70 to 74, 75 to 79, 80 plus
    above_65_ages = [18, 19, 20, 21]
    above_65_da = pop_da.sel(age_group_id=above_65_ages)
    above_65_val = above_65_da.sum('age_group_id').values.item(0)

    # <20 years
    under_20_val = pop_da.sel(age_group_id=158).values.item(0)

    above_65_val = _helper_round(above_65_val, 1e9)
    under_20_val = _helper_round(under_20_val, 1e9)

    print(f"Findings also suggest a shifting age structure in many parts of "
          f"the world, with {above_65_val} billion individuals above the age "
          f"of 65, and {under_20_val} billion individuals below the age of 20, "
          f"globally in 2100.\n")


# By 2050, XX countries will have a TFR below replacement and XX below
# replacement by 2100.
def tfr_below_replacement():
    tfr_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['tfr_mean_ui'].gbd_round_id}/"
        f"future/tfr/"
        f"{settings.BASELINE_VERSIONS['tfr_mean_ui'].version}")
    tfr_path = tfr_dir / "tfr_combined.nc"
    tfr_da = open_xr(tfr_path).data.sel(scenario=0, location_id=COUNTRIES,
        quantile="mean")

    # Replacement TFR is 2.1
    below_2050 = (tfr_da.sel(year_id=2050) < 2.1).sum().values.item(0)
    below_2100 = (tfr_da.sel(year_id=2100) < 2.1).sum().values.item(0)

    print(f"By 2050, {below_2050} countries will have a TFR below replacement "
        f"and {below_2100} below replacement by 2100.\n")


# XX countries including China, XX, XX and XX in the reference scenario will
# have declines of greater than 50% from 2017 by 2100.
def pop_declines():
    forecast_pop_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['population_mean_ui'].gbd_round_id}/"
        f"future/population/"
        f"{settings.BASELINE_VERSIONS['population_mean_ui'].version}")
    forecast_pop_path = forecast_pop_dir / "population_combined.nc"
    forecast_pop_da = open_xr(forecast_pop_path).data.sel(
        sex_id=BOTH_SEX_ID, age_group_id=ALL_AGE_ID, scenario=0,
        quantile="mean", location_id=COUNTRIES)

    past_pop_dir = FBDPath(
        f"/{settings.PAST_VERSIONS['population'].gbd_round_id}/"
        f"past/population/"
        f"{settings.PAST_VERSIONS['population'].version}")
    past_pop_path = past_pop_dir / "population.nc"
    past_pop_da = open_xr(past_pop_path).data.sel(
        sex_id=BOTH_SEX_ID, age_group_id=ALL_AGE_ID, location_id=COUNTRIES)

    pct_decline_da = 1-(forecast_pop_da.sel(year_id=2100)/past_pop_da.sel(
        year_id=2017))
    decline_over_50_val = (pct_decline_da > .5).sum().values.item(0)
    decline_over_50_locs = list(pct_decline_da.where(
        pct_decline_da > .5, drop=True).location_id.values)
    decline_over_50_locs = _location_id_to_name(decline_over_50_locs)

    # compute % decline at draw level for uncertainty
    forecast_draw_pop_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['population'].gbd_round_id}/"
        f"future/population/"
        f"{settings.BASELINE_VERSIONS['population'].version}")
    forecast_draw_pop_path = forecast_draw_pop_dir / "population_agg.nc"
    forecast_draw_pop_da = open_xr(forecast_draw_pop_path).data.sel(
        sex_id=BOTH_SEX_ID, age_group_id=ALL_AGE_ID, scenario=0,
        location_id=COUNTRIES)
    past_draw_pop_dir = FBDPath(
        f"/{settings.PAST_VERSIONS['population_draw2017'].gbd_round_id}/"
        f"past/population/"
        f"{settings.PAST_VERSIONS['population_draw2017'].version}")
    past_draw_pop_path = past_draw_pop_dir / "population.nc"
    past_draw_pop_da = open_xr(past_draw_pop_path).data.sel(
        sex_id=BOTH_SEX_ID, age_group_id=ALL_AGE_ID, location_id=COUNTRIES)

    da_china_2100 = forecast_draw_pop_da.sel(year_id=2100,location_id=6)
    da_china_2017 = past_draw_pop_da.sel(year_id=2017,location_id=6)
    pct_draw_decline_da = 1-(da_china_2100/da_china_2017)

    decline_china = pct_draw_decline_da*100
    decline_china_mean = _helper_round(
        decline_china.mean("draw").values.item(0), 1)
    decline_china_lower = _helper_round(
        decline_china.quantile(dim="draw", q=[0.025]).values.item(0), 1)
    decline_china_upper = _helper_round(
        decline_china.quantile(dim="draw", q=[0.975]).values.item(0), 1)

    print(f"{decline_over_50_val} countries including {decline_over_50_locs} "
          f"in the reference scenario will have declines of greater than "
          f"50% from 2017 by 2100; China will decline by {decline_china_mean} "
          f"({decline_china_lower} to {decline_china_upper}).%\n")


# Alternative scenarios suggest meeting the SDG targets for education and
# contraceptive met need will result in a global population of 6.2 billion in
# 2100, and 6.8 billion assuming 99th percentile rates of change in educational
# attainment and met need for contraception.
def alt_scenario_pops():
    pop_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['population_mean_ui'].gbd_round_id}/"
        f"future/population/"
        f"{settings.BASELINE_VERSIONS['population_mean_ui'].version}")
    pop_path = pop_dir / "population_combined.nc"
    # Scenario 3 is SDG
    sdg_pop_da = open_xr(pop_path).data.sel(
        location_id=1, age_group_id=22, sex_id=3, scenario=3, quantile="mean")
    sdg_2100_pop = sdg_pop_da.sel(year_id=2100).values.item(0)

    # Scenario 2 is the 99
    fastest_pop_da = open_xr(pop_path).data.sel(
        location_id=1, age_group_id=22, sex_id=3, scenario=2, quantile="mean")
    fastest_2100_pop = fastest_pop_da.sel(year_id=2100).values.item(0)

    sdg_2100_pop = _helper_round(sdg_2100_pop, 1e9)
    fastest_2100_pop = _helper_round(fastest_2100_pop, 1e9)

    print(f"Alternative scenarios suggest meeting the SDG targets for "
          f"education and contraceptive met need will result in a global "
          f"population of {sdg_2100_pop} billion in 2100, and "
          f"{fastest_2100_pop} billion assuming 99th percentile rates of "
          f"change in educational attainment and met need for contraception.\n")


#  Forecasts from this study differ from the UNPD and the Wittgenstein Centre,
# which project 10.9 billion and 9.3 billion people globally in 2100,
# respectively
def wpp_witt_pops():
    wpp_pop_dir = FBDPath(
        f"/{settings.WPP_VERSIONS['population_aggs'].gbd_round_id}/"
        f"future/population/"
        f"{settings.WPP_VERSIONS['population_aggs'].version}")
    wpp_pop_path = wpp_pop_dir / "2019_fhs_agg_allage_bothsex_only.nc"
    wpp_pop_da = open_xr(wpp_pop_path).data.sel(location_id=1, year_id=2100)

    witt_pop_dir = FBDPath(
        f"/{settings.WITT_VERSIONS['population'].gbd_round_id}/"
        f"future/population/"
        f"{settings.WITT_VERSIONS['population'].version}")
    witt_pop_path = witt_pop_dir / "population_ssp2.nc"
    witt_pop_da = open_xr(witt_pop_path).data.sel(
        location_id=1, year_id=2100, sex_id=3, age_group_id=22)

    wpp_pop_val = _helper_round(wpp_pop_da.values.item(0), 1e9)
    witt_pop_val = _helper_round(witt_pop_da.values.item(0), 1e9)

    print(f"Forecasts from this study differ from the UNPD and the "
          f"Wittgenstein Centre, which project {wpp_pop_val} billion and "
          f"{witt_pop_val} billion people globally in 2100, respectively.\n")


# The difference with UNPD in 2100 is due to XX million fewer people in the
# reference scenario in sub-Saharan Africa, XX million fewer in South Asia,
# and XX million fewer in Southeast Asia, East Asia, and Oceania, primarily due
# to the level of fertility achieved in below replacement populations.
def _helper_wpp_fhs_diff(wpp_pop_da, fhs_pop_da, loc_id):
    diff = (wpp_pop_da.sel(location_id=loc_id).values.item(0) -
            fhs_pop_da.sel(location_id=loc_id).values.item(0))
    return diff
def wpp_fhs_diff():
    wpp_pop_dir = FBDPath(
        f"/{settings.WPP_VERSIONS['population_aggs'].gbd_round_id}/"
        f"future/population/"
        f"{settings.WPP_VERSIONS['population_aggs'].version}")
    wpp_pop_path = wpp_pop_dir / "2019_fhs_agg_allage_bothsex_only.nc"
    wpp_pop_da = open_xr(wpp_pop_path).data.sel(year_id=2100)

    fhs_pop_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['population_mean_ui'].gbd_round_id}/"
        f"future/population/"
        f"{settings.BASELINE_VERSIONS['population_mean_ui'].version}")
    fhs_pop_path = fhs_pop_dir / "population_combined.nc"
    fhs_pop_da = open_xr(fhs_pop_path).data.sel(
        sex_id=BOTH_SEX_ID, age_group_id=ALL_AGE_ID, scenario=0,
        quantile="mean", year_id=2100)

    # Sub-Saharan 166
    sub_saharan_diff = _helper_wpp_fhs_diff(wpp_pop_da, fhs_pop_da, 166)
    # South Asia 158
    south_asia_diff = _helper_wpp_fhs_diff(wpp_pop_da, fhs_pop_da, 158)
    # Southeast Asia, East Asia, and Oceania 4
    se_e_oceania_diff = _helper_wpp_fhs_diff(wpp_pop_da, fhs_pop_da, 4)

    sub_saharan_diff = _helper_round(sub_saharan_diff, 1e6)
    south_asia_diff = _helper_round(south_asia_diff, 1e6)
    se_e_oceania_diff = _helper_round(se_e_oceania_diff, 1e6)

    print(f"The difference with UNPD in 2100 is due to {sub_saharan_diff} "
          f"million fewer people in the reference scenario in sub-Saharan "
          f"Africa, {south_asia_diff} million fewer in South Asia, and "
          f"{se_e_oceania_diff} million fewer in Southeast Asia, East Asia, "
          f"and Oceania, primarily due to the level of fertility achieved in "
          f"below replacement populations. \n")



# China is expected to become the largest economy by XX but in the reference
# scenario the USA would once again become in the largest economy in XX.
def largest_gdp():
    gdp_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['gdp'].gbd_round_id}/"
        f"future/gdp/"
        f"{settings.BASELINE_VERSIONS['gdp'].version}")
    gdp_path = gdp_dir / "gdp.nc"
    gdp_da = open_xr(gdp_path).data.sel(scenario=0)

    max_da = gdp_da.where(gdp_da==gdp_da.max('location_id'), drop=True)
    # Find years where china is top
    china_years = max_da.sel(
        location_id=6).dropna(dim='year_id').year_id.values
    # Find years where USA is top
    usa_years = max_da.sel(
        location_id=102).dropna(dim='year_id').year_id.values
    # check no other location is ever top
    missing_years = np.setdiff1d(
        gdp_da.coords["year_id"].values,
        np.concatenate([china_years, usa_years]))
    assert missing_years.size == 0

    # find first year in future where china takes lead
    china_year = china_years.min()
    # find first year when US regains lead
    usa_year = usa_years[usa_years > china_year].min()

    print(f"China is expected to become the largest economy by {china_year} "
          f"but in the reference scenario the USA would once again become in "
          f"the largest economy in {usa_year}.\n")



if __name__ == "__main__":

    tfr_2100()
    pop_peak()
    most_populated_2100()
    age_pops()
    tfr_below_replacement()
    pop_declines()
    alt_scenario_pops()
    #wpp_witt_pops()
    #wpp_fhs_diff()
    largest_gdp()


    great_job.congratulations()

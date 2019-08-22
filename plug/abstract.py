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
achieved in below replacement populations. China is expected to become the
largest economy by XX but in the reference scenario the USA would once again
become in the largest economy in XX.
"""
import pandas as pd
import xarray as xr

from fbd_core import argparse, db, great_job
from fbd_core import great_job
from fbd_core.etl import resample
from fbd_core.etl.aggregator import Aggregator
from fbd_core.etl.transformation import expand_dimensions
from fbd_core.file_interface import FBDPath, open_xr, save_xr

import settings
import ipdb

ALL_AGE_ID = 22
BOTH_SEX_ID = 3

N_DECIMALS = 2

COUNTRIES = db.get_locations_by_level(3).location_id.values.tolist()


def _helper_round(val, divide_by_this):
    return round(val/divide_by_this, N_DECIMALS)


# The global total fertility rate (TFR) in 2100 is forecasted to be 1.66 
# (95% UI xx–xx).


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

    # Loc IDs: India 163, Nigeria 214, China 6, USA 102, DR Congo 171
    india_val = pop_da.sel(location_id=163, quantile="mean").values.item(0)
    india_upper = pop_da.sel(location_id=163, quantile="upper").values.item(0)
    india_lower = pop_da.sel(location_id=163, quantile="lower").values.item(0)

    nigeria_val = pop_da.sel(location_id=214, quantile="mean").values.item(0)
    nigeria_upper = pop_da.sel(location_id=214, quantile="upper").values.item(0)
    nigeria_lower = pop_da.sel(location_id=214, quantile="lower").values.item(0)

    china_val = pop_da.sel(location_id=6, quantile="mean").values.item(0)
    china_upper = pop_da.sel(location_id=6, quantile="upper").values.item(0)
    china_lower = pop_da.sel(location_id=6, quantile="lower").values.item(0)

    usa_val = pop_da.sel(location_id=102, quantile="mean").values.item(0)
    usa_upper = pop_da.sel(location_id=102, quantile="upper").values.item(0)
    usa_lower = pop_da.sel(location_id=102, quantile="lower").values.item(0)

    drcongo_val = pop_da.sel(location_id=171, quantile="mean").values.item(0)
    drcongo_upper = pop_da.sel(location_id=171, quantile="upper").values.item(0)
    drcongo_lower = pop_da.sel(location_id=171, quantile="lower").values.item(0)

    # round
    india_val = _helper_round(india_val, 1e6)
    india_upper = _helper_round(india_upper, 1e6)
    india_lower = _helper_round(india_lower, 1e6)

    nigeria_val = _helper_round(nigeria_val, 1e6)
    nigeria_upper = _helper_round(nigeria_upper, 1e6)
    nigeria_lower = _helper_round(nigeria_lower, 1e6)

    china_val = _helper_round(china_val, 1e6)
    china_upper = _helper_round(china_upper, 1e6)
    china_lower = _helper_round(china_lower, 1e6)

    usa_val = _helper_round(usa_val, 1e6)
    usa_upper = _helper_round(usa_upper, 1e6)
    usa_lower = _helper_round(usa_lower, 1e6)

    drcongo_val = _helper_round(drcongo_val, 1e6)
    drcongo_upper = _helper_round(drcongo_upper, 1e6)
    drcongo_lower = _helper_round(drcongo_lower, 1e6)

    print(f"The reference projections for the 5 five largest countries in 2100 "
        f"are India ({india_val} million [{india_lower}-{india_upper}]), "
        f"Nigeria ({nigeria_val} million [{nigeria_lower}-{nigeria_upper}]), "
        f"China ({china_val} million [{china_lower}-{china_upper}]), "
        f"USA ({usa_val} million [{usa_lower}-{usa_upper}]), and "
        f"DR Congo ({drcongo_val} million [{drcongo_lower}-{drcongo_upper}])."
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
          f"globally in 2100.")


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
        f"and {below_2100} below replacement by 2100.")


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

    pct_decline_da = 1-(forecast_pop_da.sel(year_id=2100)/past_pop_da.sel(year_id=2017))
    decline_over_50_val = (pct_decline_da > .4).sum().values.item(0)
    decline_over_50_locs = list(pct_decline_da.where(pct_decline_da > .5, drop=True).location_id.values)
    ipdb.set_trace()
    print(f"XX countries including China, XX, XX and XX in the reference scenario will have declines of greater than 50% from 2017 by 2100.")


# Alternative scenarios suggest meeting the SDG targets for education and
# contraceptive met need will result in a global population of 6.2 billion in
# 2100, and 6.8 billion assuming 99th percentile rates of change in educational
# attainment and met need for contraception.
def alt_scenario_pops():
    sdg_pop_dir = FBDPath(
        f"/{settings.SDG_SCENARIO_VERSIONS['population_mean_ui'].gbd_round_id}/"
        f"future/population/"
        f"{settings.SDG_SCENARIO_VERSIONS['population_mean_ui'].version}")
    sdg_pop_path = sdg_pop_dir / "population_combined.nc"
    # Scenario -1 is SDG
    sdg_pop_da = open_xr(sdg_pop_path).data.sel(
        location_id=1, age_group_id=22, sex_id=3, scenario=-1, quantile="mean")
    sdg_2100_pop = sdg_pop_da.sel(year_id=2100).values.item(0)

    fastest_pop_dir = FBDPath(
        f"/{settings.FASTEST_SCENARIO_VERSIONS['population_mean_ui'].gbd_round_id}/"
        f"future/population/"
        f"{settings.FASTEST_SCENARIO_VERSIONS['population_mean_ui'].version}")
    fastest_pop_path = fastest_pop_dir / "population_combined.nc"
    # Scenario 1 is the 99
    fastest_pop_da = open_xr(fastest_pop_path).data.sel(
        location_id=1, age_group_id=22, sex_id=3, scenario=1, quantile="mean")
    fastest_2100_pop = fastest_pop_da.sel(year_id=2100).values.item(0)

    sdg_2100_pop = _helper_round(sdg_2100_pop, 1e9)
    fastest_2100_pop = _helper_round(fastest_2100_pop, 1e9)
    
    print(f"Alternative scenarios suggest meeting the SDG targets for "
          f"education and contraceptive met need will result in a global "
          f"population of {sdg_2100_pop} billion in 2100, and "
          f"{fastest_2100_pop} billion assuming 99th percentile rates of "
          f"change in educational attainment and met need for contraception.")


#  Forecasts from this study differ from the UNPD and the Wittgenstein Centre,
# which project 10.9 billion and 9.3 billion people globally in 2100,
# respectively
def wpp_witt_pops():
    wpp_pop_dir = FBDPath(
        f"/{settings.WPP_VERSIONS['population'].gbd_round_id}/"
        f"future/population/"
        f"{settings.WPP_VERSIONS['population'].version}")
    wpp_pop_path = wpp_pop_dir / "population.nc"
    wpp_pop_da = open_xr(wpp_pop_path).data
    
    witt_pop_dir = FBDPath(
        f"/{settings.WITT_VERSIONS['population'].gbd_round_id}/"
        f"future/population/"
        f"{settings.WITT_VERSIONS['population'].version}")
    witt_pop_path = witt_pop_dir / "population_ssp2.nc"
    witt_pop_da = open_xr(witt_pop_path).data
    ipdb.set_trace()


# The difference with UNPD in 2100 is due to XX million fewer people in the
# reference scenario in sub-Saharan Africa and XX million in other regions,
# primarily due to the level of fertility achieved in below replacement
# populations.


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
    # Find 1st year where China is not nan (ie it is the max gdp for that year)
    china_year = max_da.sel(
        location_id=6).dropna(dim='year_id').year_id.values.item(0)
    # Subset to post-china_year years for when USA retakes the lead
    ok_usa_years = range(china_year, max_da.year_id.max().values.item(0) + 1)
    past_china_usa_da = max_da.sel(location_id=102, year_id=ok_usa_years)
    # Find 1st year where USA retakes lead
    usa_year = past_china_usa_da.dropna(dim='year_id').year_id.values.item(0)

    print(f"China is expected to become the largest economy by {china_year} "
          f"but in the reference scenario the USA would once again become in "
          f"the largest economy in {usa_year}.")



if __name__ == "__main__":

    #pop_peak()
    #most_populated_2100()
    #age_pops()
    #tfr_below_replacement()
    pop_declines()
    #alt_scenario_pops()
    #wpp_witt_pops()
    #largest_gdp()


    great_job.congratulations()
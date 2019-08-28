"""
Discussion: In contrast, positive incentives have had little effect in Singapore
and Taiwan, where 2017 TFR levels were 1·26 (XX–XX) and 1·04 (XX–XX),

Discussion: Japan, Hungary, Slovakia, the Baltic states, and others are facing
substantial declines in population but have not adopted immigration as a
strategy. 

Limitations: But for countries such as Niger (190 [95% UI XX–XX] million in
2100), Chad (128 [XX–XX] million in 2100), and South Sudan (83 [XX–XX] million
in 2100)  who will experience large population growth and while remaining
low-income, there is a real possibility that such levels of population will not
be sustainable

Twenty five countries, including Japan, Thailand, Italy, Spain and Ukraine, are
expected to decline in population by 50% or more. Another 33 countries will
likely decline by 25 to 50%, including China with a 49% decline. Population
percentage declines do not convey the associated profound shifts in age
structure in these nations. Our findings suggest that the ratio of the
population over 80 to the population under 15 will increase in these countries
from XX today to XX (XX–XX) in 2100. 
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

def _location_id_to_name(location_ids):
    locs = db.get_locations_by_max_level(3)[['location_id', 'location_name']]
    locs = locs[locs['location_id'].isin(location_ids)]
    location_names = locs['location_name'].tolist()

    return location_names

def _add_location_name(df):
    locs = db.get_locations_by_max_level(3)[['location_id', 'location_name']]
    df = df.merge(locs, how='left', on='location_id')
    
    return df

# Discussion: In contrast, positive incentives have had little effect in
# Singapore and Taiwan, where 2017 TFR levels were 1·26 (XX–XX) and 1·04 (XX–XX)
def sing_tai_tfr():
    tfr_dir = FBDPath(
        f"/{settings.PAST_VERSIONS['tfr'].gbd_round_id}/"
        f"past/tfr/"
        f"{settings.PAST_VERSIONS['tfr'].version}")
    tfr_path = tfr_dir / "tfr.nc"
    tfr_da = open_xr(tfr_path).data.sel(year_id=2017)

    singapore_tfr = tfr_da.sel(location_id=69).mean('draw').values.item(0)
    taiwan_tfr = tfr_da.sel(location_id=8).mean('draw').values.item(0)

    singapore_upper = tfr_da.sel(location_id=69).quantile(
        .975, dim='draw').values.item(0)
    singapore_lower = tfr_da.sel(location_id=69).quantile(
        .025, dim='draw').values.item(0)

    taiwan_upper = tfr_da.sel(location_id=8).quantile(
        .975, dim='draw').values.item(0)
    taiwan_lower = tfr_da.sel(location_id=8).quantile(
        .025, dim='draw').values.item(0)

    singapore_tfr = _helper_round(singapore_tfr, 1)
    taiwan_tfr = _helper_round(taiwan_tfr, 1)
    singapore_upper = _helper_round(singapore_upper, 1)
    singapore_lower = _helper_round(singapore_lower, 1)
    taiwan_upper = _helper_round(taiwan_upper, 1)
    taiwan_lower = _helper_round(taiwan_lower, 1)

    print(f"Discussion: In contrast, positive incentives have had little "
          f"effect in Singapore and Taiwan, where 2017 TFR levels were "
          f"{singapore_tfr} ({singapore_lower}–{singapore_upper}) and "
          f"{taiwan_tfr} ({taiwan_lower}–{taiwan_upper})\n")

# Limitations: Niger (190 [95% UI XX–XX] million in 2100), Chad (128 [XX–XX]
# million in 2100), and South Sudan (83 [XX–XX] million in 2100)
def pops_2100():
    pop_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['population_mean_ui'].gbd_round_id}/"
        f"future/population/"
        f"{settings.BASELINE_VERSIONS['population_mean_ui'].version}")
    pop_path = pop_dir / "population_combined.nc"
    pop_da = open_xr(pop_path).data.sel(
        age_group_id=ALL_AGE_ID, sex_id=BOTH_SEX_ID, scenario=0, year_id=2100)

    niger_pop = pop_da.sel(location_id=213, quantile="mean").values.item(0)
    niger_upper = pop_da.sel(location_id=213, quantile="upper").values.item(0)
    niger_lower = pop_da.sel(location_id=213, quantile="lower").values.item(0)

    chad_pop = pop_da.sel(location_id=204, quantile="mean").values.item(0)
    chad_upper = pop_da.sel(location_id=204, quantile="upper").values.item(0)
    chad_lower = pop_da.sel(location_id=204, quantile="lower").values.item(0)

    south_sudan_pop = pop_da.sel(
        location_id=435, quantile="mean").values.item(0)
    south_sudan_upper = pop_da.sel(
        location_id=435, quantile="upper").values.item(0)
    south_sudan_lower = pop_da.sel(
        location_id=435, quantile="lower").values.item(0)

    niger_pop = _helper_round(niger_pop, 1e6)
    niger_upper = _helper_round(niger_upper, 1e6)
    niger_lower = _helper_round(niger_lower, 1e6)

    chad_pop = _helper_round(chad_pop, 1e6)
    chad_upper = _helper_round(chad_upper, 1e6)
    chad_lower = _helper_round(chad_lower, 1e6)

    south_sudan_pop = _helper_round(south_sudan_pop, 1e6)
    south_sudan_upper = _helper_round(south_sudan_upper, 1e6)
    south_sudan_lower = _helper_round(south_sudan_lower, 1e6)


    print(f"Niger ({niger_pop} [95% UI {niger_lower}–{niger_upper}] million in "
          f"2100), Chad ({chad_pop} [{chad_lower}–{chad_upper}] million in "
          f"2100), and South Sudan ({south_sudan_pop} [{south_sudan_lower}–"
          f"{south_sudan_upper}] million in 2100)\n")


# Discussion: Japan, Hungary, Slovakia, the Baltic states, and others are facing
# substantial declines in population but have not adopted immigration as a
# strategy.
def pop_declines():
    pop_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['population_mean_ui'].gbd_round_id}/"
        f"future/population/"
        f"{settings.BASELINE_VERSIONS['population_mean_ui'].version}")
    pop_path = pop_dir / "population_combined.nc"
    pop_da = open_xr(pop_path).data.sel(
        age_group_id=ALL_AGE_ID, sex_id=BOTH_SEX_ID, scenario=0,
        quantile="mean")
    
    pct_decline_da = 1-(pop_da.sel(year_id=2100)/pop_da.sel(year_id=2018))
    japan_val = pct_decline_da.sel(location_id=67).values.item(0)
    hungary_val = pct_decline_da.sel(location_id=48).values.item(0)
    slovakia_val = pct_decline_da.sel(location_id=54).values.item(0)
    estonia_val = pct_decline_da.sel(location_id=58).values.item(0)
    latvia_val = pct_decline_da.sel(location_id=59).values.item(0)
    lithuania_val = pct_decline_da.sel(location_id=60).values.item(0)

    print(f"Discussion: Japan ({japan_val}), Hungary ({hungary_val}), "
          f"Slovakia ({slovakia_val}), the Baltic states ({estonia_val}, "
          f"{latvia_val}, {lithuania_val}), and others are facing substantial "
          f"declines in population but have not adopted immigration as a "
          f"strategy.\n")

"""
Twenty five countries, including Japan, Thailand, Italy, Spain and Ukraine, are
expected to decline in population by 50% or more. Another 33 countries will
likely decline by 25 to 50%, including China with a 49% decline. Population
percentage declines do not convey the associated profound shifts in age
structure in these nations. Our findings suggest that the ratio of the
population over 80 to the population under 15 will increase in these countries
from XX today to XX (XX–XX) in 2100.
"""
def pop_decline_pcts():
    forecast_pop_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['population_mean_ui'].gbd_round_id}/"
        f"future/population/"
        f"{settings.BASELINE_VERSIONS['population_mean_ui'].version}")
    forecast_pop_path = forecast_pop_dir / "population_combined.nc"
    forecast_pop_da = open_xr(forecast_pop_path).data.sel(
        sex_id=BOTH_SEX_ID, scenario=0, quantile="mean", location_id=COUNTRIES)

    past_pop_dir = FBDPath(
        f"/{settings.PAST_VERSIONS['population'].gbd_round_id}/"
        f"past/population/"
        f"{settings.PAST_VERSIONS['population'].version}")
    past_pop_path = past_pop_dir / "population.nc"
    past_pop_da = open_xr(past_pop_path).data.sel(
        sex_id=BOTH_SEX_ID, location_id=COUNTRIES)

    pct_decline_da = 1-(forecast_pop_da.sel(age_group_id=ALL_AGE_ID, year_id=2100)/past_pop_da.sel(
        age_group_id=ALL_AGE_ID, year_id=2017))
    decline_over_50_val = (pct_decline_da >= .5).sum().values.item(0)
    decline_over_50_locs = list(pct_decline_da.where(
        pct_decline_da >= .5, drop=True).location_id.values)
    decline_25_to_50_val = ((pct_decline_da > .25) & (pct_decline_da < .5)).sum(
        ).values.item(0)
    decline_25_to_50_locs = list(pct_decline_da.where(
        (pct_decline_da > .25) & (pct_decline_da < .5), drop=True).location_id.values)
    decline_over_25_locs = decline_over_50_locs + decline_25_to_50_locs

    forecast_pop_over_50 = forecast_pop_da.sel(
        location_id=decline_over_50_locs, year_id=2100)
    past_pop_over_50 = past_pop_da.sel(
        location_id=decline_over_50_locs, year_id=2017)
    forecast_pop_25_to_50 = forecast_pop_da.sel(
        location_id=decline_25_to_50_locs, year_id=2100)
    past_pop_25_to_50 = past_pop_da.sel(
        location_id=decline_25_to_50_locs, year_id=2017)
    forecast_pop_over_25 = forecast_pop_da.sel(
        location_id=decline_over_25_locs, year_id=2100)
    past_pop_over_25 = past_pop_da.sel(
        location_id=decline_over_25_locs, year_id=2017)

    over_80_age_ids = [30, 31, 32, 235]
    under_15_age_ids = [2, 3, 4, 5, 6, 7]
    
    ratio_2100 = (forecast_pop_over_25.sel(age_group_id=over_80_age_ids).sum('age_group_id').sum('location_id') /
        forecast_pop_over_25.sel(age_group_id=under_15_age_ids).sum('age_group_id').sum('location_id')).values.item(0)
    ratio_2017 = (past_pop_over_25.sel(age_group_id=over_80_age_ids).sum('age_group_id').sum('location_id') /
        past_pop_over_25.sel(age_group_id=under_15_age_ids).sum('age_group_id').sum('location_id')).values.item(0)

    forecast_pop_upper = open_xr(forecast_pop_path).data.sel(
        sex_id=BOTH_SEX_ID, scenario=0, quantile="upper",
        location_id=decline_over_50_locs)
    forecast_pop_lower = open_xr(forecast_pop_path).data.sel(
        sex_id=BOTH_SEX_ID, scenario=0, quantile="lower",
        location_id=decline_over_50_locs)

    ratio_2100_lower = (forecast_pop_lower.sel(age_group_id=over_80_age_ids).sum('age_group_id').sum('location_id') /
        forecast_pop_lower.sel(age_group_id=under_15_age_ids).sum('age_group_id').sum('location_id')).values.item(0)
    ratio_2100_upper = (forecast_pop_upper.sel(age_group_id=over_80_age_ids).sum('age_group_id').sum('location_id') /
        forecast_pop_upper.sel(age_group_id=under_15_age_ids).sum('age_group_id').sum('location_id')).values.item(0)

    forecast_pop_draw_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['population'].gbd_round_id}/"
        f"future/population/"
        f"{settings.BASELINE_VERSIONS['population'].version}")
    forecast_pop_draw_path = forecast_pop_draw_dir / "population.nc"
    forecast_pop_draw = open_xr(forecast_pop_draw_path).data.sel(scenario=0,
        location_id=decline_over_25_locs, year_id=2100).sum('sex_id')
    
    ratio_draws = (forecast_pop_draw.sel(age_group_id=over_80_age_ids).sum(
        'age_group_id').sum('location_id') /
        forecast_pop_draw.sel(age_group_id=under_15_age_ids).sum(
            'age_group_id').sum('location_id'))
    
    ratio_2100_lower = ratio_draws.quantile(q=.025).values.item(0)
    ratio_2100_upper = ratio_draws.quantile(q=.975).values.item(0)

    ratio_2017 = _helper_round(ratio_2017, 1)
    ratio_2100 = _helper_round(ratio_2100, 1)
    ratio_2100_lower = _helper_round(ratio_2100_lower, 1)
    ratio_2100_upper = _helper_round(ratio_2100_upper, 1)


    print(f"{decline_over_50_val} countries, including Japan, Thailand, Italy, "
          f"Spain and Ukraine, are expected to decline in population by 50% or "
          f"more. Another {decline_25_to_50_val} countries will likely decline "
          f"by 25 to 50%, including China with a 49% decline. Population "
          f"percentage declines do not convey the associated profound shifts "
          f"in age structure in these nations. Our findings suggest that the "
          f"ratio of the population over 80 to the population under 15 will "
          f"increase in these countries from {ratio_2017} today to "
          f"{ratio_2100} ({ratio_2100_lower}–{ratio_2100_upper}) in 2100.")


# Japan and Russia pop rank 2017 vs 2100
def japan_russia_pop():
    # This one isn't direct output needs user to read index
    past = xr.open_dataarray(
        f"/ihme/forecasting/data/5/past/population/"
        f"{settings.PAST_VERSIONS['population'].version}/population.nc")
    past = past.sel(age_group_id=22, sex_id=3, location_id=COUNTRIES,
        year_id=2017)
    pastdf = past.to_dataframe()
    pastdf = _add_location_name(pastdf)
    pastdf = pastdf.sort_values(by="population", ascending=False).reset_index()
    pastdf[pastdf['location_name']=="Japan"]
    pastdf[pastdf['location_name']=="Russia"]

    forecast = xr.open_dataarray(
        f"/ihme/forecasting/data/5/future/population/"
        f"{settings.BASELINE_VERSIONS['population_mean_ui'].version}/"
        f"population.nc")
    forecast = forecast.sel(age_group_id=22, sex_id=3, location_id=COUNTRIES,
        year_id=2100)
    forecastdf = forecast.to_dataframe()
    forecastdf = _add_location_name(forecastdf)
    forecastdf = forecastdf.sort_values(
        by="value", ascending=False).reset_index()
    forecastdf[forecastdf['location_name']=="Japan"]
    forecastdf[forecastdf['location_name']=="Russia"]



if __name__ == "__main__":

    # sing_tai_tfr()
    # pops_2100()
    # pop_declines()
    pop_decline_pcts()

    great_job.congratulations()
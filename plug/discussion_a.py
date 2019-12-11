'''
A decline in total population in the latter half of the century, such that the
world will have roughly the same number of people in 2100 as today, is
potentially good news for the global environment.

While good for the environment, population decline and associated shifts in
age structure in many nations may have profound and negative consequences.
25 countries, including Japan, Thailand, Italy, Spain and Ukraine, are expected
to decline in population by 50% or more. Another 33 countries will likely
decline by 25 to 50%.

Our findings suggest that the ratio of the population over 80 to the population
under 15 will increase in these countries from XX today to XX (XX–XX) in 2100.

The XX% (95% UI XX–XX%) decline in China’s population 20-24 is a factor that
cannot be ignored when considering possible shifts in global power later in the
century.
'''

import xarray as xr
import pandas as pd
import numpy as np

from fbd_core import db
from fbd_core.file_interface import FBDPath, open_xr
from fbd_core.etl import expand_dimensions

import settings


gbd_round_id = settings.BASELINE_VERSIONS["population_mean_ui"].gbd_round_id

# population versions
PAST = settings.PAST_VERSIONS["population"].version
PAST_DRAWS = settings.PAST_VERSIONS["population_draw2017"].version
REF = settings.BASELINE_VERSIONS["population_mean_ui"].version
REF_DRAWS = settings.BASELINE_VERSIONS["population"].version


def decline_pattern(ref):
    ref_da = ref.sel(sex_id=3,
                     age_group_id=22,
                     scenario=0,
                     location_id=1)

    ref_2018 = (ref_da.sel(year_id=2018) / 1e9).mean('draw').values
    ref_2100 = (ref_da.sel(year_id=2100) / 1e9).mean('draw').values
    diff = np.mean(ref_2100 - ref_2018)

    print(f"2018 Population: {ref_2018.round(2)} B \n"
          f"2100 Population: {ref_2100.round(2)} B \n"
          f"Difference: {diff.round(2)} B")


def decline_50_plus(ref, past_pop):
    loc_list = list(db.get_locations_by_level(3).location_id)
    
    ref_2017 = past_pop.sel(year_id=2017,
                       age_group_id=22,
                       location_id=loc_list,
                       sex_id=3) / 1e9

    ref_2100 = ref.sel(year_id=2100,
                       age_group_id=22,
                       location_id=loc_list,
                       sex_id=3).sel(scenario=0, drop=True) / 1e9

    change = (ref_2100 - ref_2017) / ref_2017
    mean_change = change.mean('draw')

    fifty_plus = change.mean('draw').where(change.mean('draw') < -0.5).dropna(
        "location_id")

    china = change.sel(location_id=6).mean('draw')
    japan = change.sel(location_id=67).mean('draw')

    twentyfive_to_fifty = mean_change\
        .where(mean_change < -0.25).where(mean_change > -0.50)\
        .dropna("location_id")

    ref_slice = past_pop.sel(location_id=fifty_plus.location_id.values,
                             age_group_id=22,
                             sex_id=3,
                             year_id=2017)

    print(f"Number of countries decline greater than 50%: {len(fifty_plus)}"\
          f"China percent decline: {china.values.round(2)}"\
          f"Japan percent decline: {japan.values.round(2)}"\
          f"Number of countries decline between 25% - 50%: "\
          f"{len(twentyfive_to_fifty)}")

    return change


def old_young_ratio(change, ref_draws, past_pop):
    twentyfive_plus = change.where(change < -0.25).dropna('location_id')

    young = range(2, 8, 1)
    old = [21]

    old_2017 = past_pop.sel(location_id=twentyfive_plus.location_id,
                            age_group_id=list(old),
                            sex_id=3,
                            year_id=2017).sum("age_group_id")
    young_2017 = past_pop.sel(location_id=twentyfive_plus.location_id,
                              age_group_id=list(young),
                              sex_id=3,
                              year_id=2017).sum("age_group_id")

    ratio_2017 = (old_2017.sum("location_id") / \
                 young_2017.sum("location_id")).mean('draw').values

    old_2100 = ref_draws.sel(location_id=twentyfive_plus.location_id,
                             age_group_id=list(old),
                             sex_id=3,
                             scenario=0,
                             year_id=2100).sum("age_group_id")
    young_2100 = ref_draws.sel(location_id=twentyfive_plus.location_id,
                               age_group_id=list(young),
                               sex_id=3,
                               scenario=0,
                               year_id=2100).sum("age_group_id")

    ratio_2100 = (old_2100.sum("location_id") /
                  young_2100.sum("location_id"))\
                    .mean('draw').values
    ratio_2100_lower = (old_2100.sum("location_id") /
                        young_2100.sum("location_id"))\
                            .quantile(0.25, 'draw').values
    ratio_2100_upper = (old_2100.sum("location_id") /
                        young_2100.sum("location_id"))\
                            .quantile(0.975, 'draw').values

    print(f"Our findings suggest that the ratio of the population over 80 to "
          f"the population under 15 will increase in countries with more than "
          f"a 25% population decline, from {ratio_2017.round(2)} today to"
          f" {ratio_2100.round(2)} "
          f"(95% UI {ratio_2100_lower.round(2)}– {ratio_2100_upper.round(2)}) "
          f"in 2100.")


'''The XX% (95% UI XX–XX%) decline in China’s population 20-24 is a factor that 
cannot be ignored when considering possible shifts in global power later in the 
century.'''
def china_decline(past_pop, ref_draws):
    ref_2017 = past_pop.sel(year_id=2017,
                       age_group_id=9,
                       location_id=6,
                       sex_id=3) / 1e9

    ref_2100 = ref_draws.sel(year_id=2100,
                       scenario=0,
                       age_group_id=9,
                       location_id=6,
                       sex_id=3) / 1e9

    change = (ref_2100 - ref_2017) / ref_2017

    mean_change = change.mean('draw').values.round(2)
    change_lower = change.quantile(0.25, 'draw').values.round(2)
    change_upper = change.quantile(0.975, 'draw').values.round(2)

    print(f"The {mean_change}% (95% UI {change_lower}, {change_upper}%) "
          f"decline in China’s population 20-24 is a factor that"
          f"cannot be ignored when considering possible shifts in global "
          f"power later in the century.")


if __name__ == "__main__":
    ref_file = FBDPath(
        f"/{gbd_round_id}/future/population/{REF}/population_combined.nc")
    ref = open_xr(ref_file).data

    past_pop_file = FBDPath(
        f"/{gbd_round_id}/past/population/{PAST}/population.nc")
    past_pop = open_xr(past_pop_file).data

    past_draw_ver = FBDPath(
        f"/{gbd_round_id}/past/population/{PAST_DRAWS}/population.nc")
    past_draw = open_xr(past_draw_ver).data

    ref_draws_ver = FBDPath(
        f"/{gbd_round_id}/future/population/{REF_DRAWS}/population_agg.nc")
    ref_draws = open_xr(ref_draws_ver).data

    decline_pattern(ref_draws)
    change = decline_50_plus(ref_draws, past_draw)
    old_young_ratio(change, ref_draws, past_draw)
    china_decline(past_draw, ref_draws)
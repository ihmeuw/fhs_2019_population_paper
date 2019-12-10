"""
We forecasted 108 countries will have net migration rates between -1 and 1 per
1000 population in 2100, and 164 countries between -2 and 2. The countries with
the largest in-migration in absolute numbers in 2100 are the United States,
India, and China whereas out-migration will be largest in Somalia, Philippines,
and Bangladesh. Net in-migration rates are forecasted to be high in Timor-Leste,
Israel, and Canada, while out-migration rates are highest in Samoa, El Salvador,
Jamaica, and Syria. Detailed results from our migration model can be downloaded
from the Global Health Data Exchange (GHDx).
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

def _load_migration_rate():
    pop_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['population_mean_ui'].gbd_round_id}/"
        f"future/population/"
        f"{settings.BASELINE_VERSIONS['population_mean_ui'].version}")
    pop_path = pop_dir / "population_combined.nc"
    pop_da = open_xr(pop_path).data.sel(
        scenario=0, quantile="mean", location_id=COUNTRIES, sex_id=BOTH_SEX_ID,
        age_group_id=ALL_AGE_ID)

    mig_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['migration'].gbd_round_id}/"
        f"future/migration/{settings.BASELINE_VERSIONS['migration'].version}")
    mig_path = mig_dir / "migration.nc"
    mig_da = open_xr(mig_path).data
    mig_da = mig_da.mean('draw').sum(['sex_id', 'age_group_id'])
    
    mig_rate_da = mig_da / pop_da
    return mig_rate_da

# We forecasted 108 countries will have net migration rates between -1 and 1 per
# 1000 population in 2100, and 164 countries between -2 and 2.
def mig_rate_range(mig_rate_da):
    mig_rate_da = mig_rate_da.sel(year_id=2100)
    count_between_one = len(mig_rate_da.where(
        (mig_rate_da<.001) & (mig_rate_da>-.001), drop=True).location_id)
    count_between_two = len(mig_rate_da.where(
        (mig_rate_da<.002) & (mig_rate_da>-.002), drop=True).location_id)
    
    print(f"We forecasted {count_between_one} countries will have net "
          f"migration rates between -1 and 1 per 1000 population in 2100, and "
          f"{count_between_two} countries between -2 and 2.")

# The countries with the largest in-migration in absolute numbers in 2100 are
# the United States, India, and China whereas out-migration will be largest in
# Somalia, Philippines, and Bangladesh.
def largest_counts():
    mig_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['migration'].gbd_round_id}/"
        f"future/migration/{settings.BASELINE_VERSIONS['migration'].version}")
    mig_path = mig_dir / "migration.nc"
    mig_da = open_xr(mig_path).data
    mig_da = mig_da.mean('draw').sum(['sex_id', 'age_group_id'])
    mig_da = mig_da.sel(year_id=2100)
    mig_df = mig_da.to_dataframe()
    mig_df = _add_location_name(mig_df)
    mig_df = mig_df.sort_values(
        by='value', ascending=False).reset_index()

    first = mig_df.location_name[0]
    second = mig_df.location_name[1]
    third = mig_df.location_name[2]
    
    last = mig_df.location_name.iloc[-1]
    second_last = mig_df.location_name.iloc[-2]
    third_last = mig_df.location_name.iloc[-3]

    print(f"The countries with the largest in-migration in absolute numbers in "
          f"2100 are {first}, {second}, and {third} whereas out-migration will "
          f"be largest in {last}, {second_last}, and {third_last}.")

# Net in-migration rates are forecasted to be high in Timor-Leste, Israel, and
# Canada, while out-migration rates are highest in Samoa, El Salvador, Jamaica,
# and Syria.
def largest_rates(mig_rate_da):
    mig_rate_da = mig_rate_da.sel(year_id=2100)
    mig_rate_df = mig_rate_da.to_dataframe()
    mig_rate_df = _add_location_name(mig_rate_df)
    mig_rate_df = mig_rate_df.sort_values(
        by='value', ascending=False).reset_index()

    first = mig_rate_df.location_name[0]
    second = mig_rate_df.location_name[1]
    third = mig_rate_df.location_name[2]

    last = mig_rate_df.location_name.iloc[-1]
    second_last = mig_rate_df.location_name.iloc[-2]
    third_last = mig_rate_df.location_name.iloc[-3]
    
    print(f"Net in-migration rates are forecasted to be high in {first}, "
          f"{second}, and {third}, while out-migration rates are highest in "
          f"{last}, {second_last}, and {third_last}.")

if __name__ == "__main__":

    mig_rate_da = _load_migration_rate()
    mig_rate_range(mig_rate_da)
    largest_counts()
    largest_rates(mig_rate_da)

    great_job.congratulations()

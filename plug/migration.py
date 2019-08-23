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


if __name__ == "__main__":

	

    great_job.congratulations()

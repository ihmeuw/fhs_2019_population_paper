"""
a location like Taiwan (province of China), with a current TFR of 1.04, will,
in this scenario, experience a steady increase in fertility over the century
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

# country like Taiwan, with a current TFR of 1.1 (loc id 8)
def taiwan_tfr():
    tfr_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['tfr_mean_ui'].gbd_round_id}/"
        f"future/tfr/"
        f"{settings.BASELINE_VERSIONS['tfr_mean_ui'].version}")
    tfr_path = tfr_dir / "tfr_combined.nc"
    tfr_da = open_xr(tfr_path).data.sel(scenario=0, location_id=8,
        quantile="mean", year_id=2017)
    
    taiwan_tfr = tfr_da.values.item(0)

    print(f"country like Taiwan, with a current TFR of {taiwan_tfr}\n")

if __name__ == "__main__":

    taiwan_tfr()


    great_job.congratulations()

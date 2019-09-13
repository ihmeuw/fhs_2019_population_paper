
"""
Combine the separate location files from the age-sex splitting of migration.
Balance the migration data to be zero at each age-sex-draw combo.
Save.

Example:

.. code:: bash
    python balance_migration.py \
        --version 20190408_test_model6_1000d_split \
        --gbd_round_id 5
"""
import logging

import pandas as pd
import xarray as xr

from fbd_core import argparse, great_job
from fbd_core.etl import df_to_xr
from fbd_core.file_interface import FBDPath, save_xr, open_xr

LOGGER = logging.getLogger(__name__)

ID_VARS = ["location_id", "year_id", "age_group_id", "sex_id", "draw"]
WPP_LOCATION_IDS = pd.read_csv(
                         "filepath")["location_id"].unique().tolist()

def combine_and_save_mig(version):
    """
    Load location csvs of migration files and combine into an xarray dataarray.

    Args:
        version (str):
            The version of migration to combine and save

    Returns:
        xarray.DataArray: The combined migration data xarray dataarray.
    """
    LOGGER.debug("Combining migration csvs to xarray")
    all_locs_xr_list = []
    # Put dataframes for each location into a list
    for loc in WPP_LOCATION_IDS:
        temp = pd.read_csv(f'filepath')
        #temp = temp.set_index(ID_VARS)
        temp = df_to_xr(temp, dims=ID_VARS)
        all_locs_xr_list.append(temp)
    # Concat all locations together
    result = xr.concat(all_locs_xr_list, dim='location_id')

    # Save to forecasting directory
    result.to_netcdf(f'filepath')
    return result
    
def balance_migration(mig_da):
    """
    Ensure that net migration is zero at each loc-sex-age combo.
    Calculate K = sqrt(-sum of positive values/sum of negative values)
    Divide positive values by K
    Multiply negative values by K

    Args:
        mig_da (xarray.DataArray):
            The input migration xarray dataarray that is being balanced

    Returns:
        xarray.DataArray: The balanced migration data,
            in dataarray.
    """
    LOGGER.debug("Entered balancing step")
    negatives = mig_da.where(mig_da < 0)
    positives = mig_da.where(mig_da > 0)
    zeros = mig_da.where(mig_da == 0)
    
    sum_dims = [dim for dim in mig_da.dims if dim not in (
        "draw", "age_group_id", "sex_id", "year_id")]
    k = positives.sum(sum_dims) / negatives.sum(sum_dims)

    # Multiply constant by positives
    adjusted_positives = xr.ufuncs.sqrt(1 / -k) * positives
    adjusted_negatives = xr.ufuncs.sqrt(-k) * negatives

    # Combine
    balanced_mig_da = adjusted_positives.combine_first(adjusted_negatives)
    balanced_mig_da = balanced_mig_da.combine_first(zeros)
    
    LOGGER.debug("Balanced migration")
    return balanced_mig_da

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", type=str,
        help="Which version of migration to balance.")
    parser.add_argument(
        "--gbd_round_id", type=int, required=True,
        help="Which gbd_round_id to use in file loading and saving")
    args = parser.parse_args()

    # Try to load data, else combine csvs into dataarray
    try:
        mig_dir = FBDPath(f"/{args.gbd_round_id}/future/migration/{args.version}/")
        mig_path = mig_dir / "migration_split.nc"
        mig_da = open_xr(mig_path).data
    except: # Data doesn't yet exist
        mig_da = combine_and_save_mig(version=args.version)

    balanced_mig_da = balance_migration(mig_da)

    # Save to forecasting directory
    balanced_path = mig_dir / "migration.nc"
    save_xr(balanced_mig_da, balanced_path, metric="number", space="identity")

    great_job.congratulations() # You did it!

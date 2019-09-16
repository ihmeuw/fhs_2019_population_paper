"""
This script works to split the migration into separate age-sex groups in a
faster way than the R script by converting the Eurostat data to xarray.

Example:

.. code:: bash

    python age_sex_split.py \
    --migration_version 20190408_test_model6_1000d \
    --gbd_round_id 5
"""

import logging

import pandas as pd
import xarray as xr

from fbd_core import argparse, great_job
from fbd_core.db import get_locations_by_level
from fbd_core.etl import df_to_xr
from fbd_core.etl.transformation import expand_dimensions
from fbd_core.file_interface import FBDPath, open_xr, save_xr

LOGGER = logging.getLogger(__name__)

# location of the pattern for the Qatar-modeled countries
QATAR_PATTERN = "filepath"
# Location of the pattern for other countries
EUROSTAT_PATTERN = "filepath"
# Version of pattern in xarray format to save/load
PATTERN_VERSION = "filepath"
# Which location ids to use
WPP_LOCATION_IDS = pd.read_csv(
                         "filepath"
                         )["location_id"].unique().tolist()
PATTERN_ID_VARS = ["age_group_id", "sex_id"]
QATAR_LOCS = [151, 152, 140, 156, 150] #'QAT', 'SAU', 'BHR', 'ARE', 'OMN'

def create_age_sex_xarray():
    LOGGER.debug("Creating xarray of age-sex patterns for migration")
    # load patterns
    qatar = pd.read_csv(QATAR_PATTERN)
    eurostat = pd.read_csv(EUROSTAT_PATTERN)
    # convert to xarrays
    qatar = df_to_xr(qatar, dims=PATTERN_ID_VARS)
    eurostat = df_to_xr(eurostat, dims=PATTERN_ID_VARS)
    # create superarray to hold all locs
    all_locs_xr_list = []
    # Put dataframes for each location into a list
    for loc in WPP_LOCATION_IDS:
        if loc in QATAR_LOCS:
            data = qatar
        else:
            data = eurostat
        data = expand_dimensions(data, location_id=[loc])
        all_locs_xr_list.append(data)
    # Concat all locations together
    result = xr.concat(all_locs_xr_list, dim='location_id')
    # Save all locs pattern
    LOGGER.debug("Saving age-sex pattern xarray")
    pattern_dir = FBDPath(f'/{gbd_round_id}/future/migration/'
            f'{PATTERN_VERSION}')
    pattern_path = pattern_dir / f"combined_age_sex_pattern.nc"
    save_xr(pattern, pattern_path, metric="percent", space="identity")
    LOGGER.debug("Saved age-sex pattern xarray")
    return result

def main(migration_version, gbd_round_id):
    # load age-sex pattern (loc, draw, age, sex)
    LOGGER.debug("Loading age-sex migration pattern")
    try:
        pattern_dir = FBDPath(f'/{gbd_round_id}/future/migration/'
            f'{PATTERN_VERSION}')
        pattern_path = pattern_dir / "combined_age_sex_pattern.nc"
        pattern = open_xr(pattern_path).data
    except  FileNotFoundError: # Data doesn't yet exist
        pattern = create_age_sex_xarray()
    # load migration counts (loc, draw, year)
    LOGGER.debug("Loading migration data")
    mig_dir = FBDPath(f"/{gbd_round_id}/future/migration/{migration_version}/")
    mig_path = mig_dir / "mig_counts.nc"
    migration = open_xr(mig_path).data
    migration = migration.squeeze(drop=True)
    # end up with migration counts with age and sex (loc, draw, year, age, sex)
    split_data = migration * pattern
    # Save it!
    LOGGER.debug("Saving age-sex split migration data")
    
    split_path = mig_dir / "migration_split.nc"
    save_xr(split_data, split_path, metric="number", space="identity")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--migration_version", type=str, required=True,
        help="Which version of migrations to use in WPP directory")
    parser.add_argument(
        "--gbd_round_id", type=int, required=True,
        help="Which gbd_round_id to use in file loading and saving")
    args = parser.parse_args()

    main(migration_version=args.migration_version,
        gbd_round_id=args.gbd_round_id)
    great_job.congratulations() # You did it!

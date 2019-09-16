"""
Convert migration rates output by the random walk step to counts for use in
age-sex splitting step.

Example:

.. code:: bash
    python migration_rate_to_count.py \
        --migration_version 20190408_test_model6_1000d \
        --past_pop_version 20181206_pop_1950_2017 \
        --forecast_pop_version 20190317_gkshift_fix_squeezed_agg \
        --gbd_round_id 5 \
        --draws 1000 \
        --years 1950:2018:2100
"""
import pandas as pd
import xarray as xr

from fbd_core import argparse, great_job
from fbd_core import great_job
from fbd_core.etl import resample
from fbd_core.etl.aggregator import Aggregator
from fbd_core.etl.transformation import expand_dimensions
from fbd_core.file_interface import FBDPath, open_xr, save_xr

SCALE_FACTOR = 1000


def load_past_pop(gbd_round_id, version):
    """
    Load past population data. This will generally be from 1950 to the start of
    the forecasts.

    Args:
        gbd_round_id (int):
            The gbd round ID that the past population is from
        version (str):
            The version of past population to read from

    Returns:
        xarray.DataArray: The past population xarray dataarray
    """
    past_pop_dir = FBDPath(f"/{gbd_round_id}/past/population/{version}")
    past_pop_path = past_pop_dir / "population.nc"
    past_pop_da = open_xr(past_pop_path).data

    return past_pop_da


def load_forecast_pop(gbd_round_id, version, years, draws):
    """
    Load forecast population data. Aggregates if necessary.

    Args:
        gbd_round_id (int):
            The gbd round ID that the past population is from
        version (str):
            The version of forecast population to read from
        years (YearRange):
            The Forecasting format years to use.

    Returns:
        xarray.DataArray: The past population xarray dataarray
    """
    forecast_pop_dir = FBDPath(f"/{gbd_round_id}/future/population/{version}")
    try:
        forecast_pop_path = forecast_pop_dir / "population_agg.nc"
        forecast_pop_da = open_xr(forecast_pop_path).data
    except: # Need to make agg version
        forecast_pop_path = forecast_pop_dir / "population.nc"
        forecast_pop_da = open_xr(forecast_pop_path).data
        forecast_pop_da = Aggregator.aggregate_everything(forecast_pop_da, gbd_round_id).pop
        forecast_pop_out_path = forecast_pop_dir / "population_agg.nc"
        save_xr(forecast_pop_da, forecast_pop_out_path, metric="number",
            space="identity")

    # slice to correct years and number of draws
    forecast_pop_da = forecast_pop_da.sel(year_id=years.forecast_years)
    forecast_pop_da = resample(forecast_pop_da, draws)

    return forecast_pop_da


def main(migration_version, past_pop_version, forecast_pop_version,
    gbd_round_id, draws, years):
    """
    Load pops and migration rate, multiply to get counts
    """
    # Load migration data
    mig_dir = FBDPath(f"/{gbd_round_id}/future/migration/{migration_version}/")
    mig_path = mig_dir / "mig_star.nc"
    mig_da = open_xr(mig_path).data

    # Load pops
    past_pop_da = load_past_pop(gbd_round_id, past_pop_version)
    forecast_pop_da = load_forecast_pop(gbd_round_id, forecast_pop_version,
        years, draws)
    
    # Give past populations dummy draws/scenarios to be concatenated with
    # forecast pops
    past_pop_da =  expand_dimensions(past_pop_da,
        draw=forecast_pop_da["draw"].values)
    past_pop_da =  expand_dimensions(past_pop_da,
        scenario=forecast_pop_da["scenario"].values)

    # Subset to coordinates relevant to mig_da
    forecast_pop_da = forecast_pop_da.sel(sex_id=3, age_group_id=22,
        location_id=mig_da.location_id.values, scenario=0)
    past_pop_da = past_pop_da.sel(sex_id=3, age_group_id=22,
        location_id=mig_da.location_id.values, scenario=0)

    # Combine past and forecast pop
    pop_da = past_pop_da.combine_first(forecast_pop_da)

    # Multiply rates by pop to get counts
    mig_counts = mig_da * pop_da
    mig_counts = mig_counts/SCALE_FACTOR
    
    # Save out
    mig_counts_path = mig_dir / "mig_counts.nc"
    save_xr(mig_counts, mig_counts_path, metric="number", space="identity")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--migration_version", type=str, required=True,
        help="Which version of migration to convert")
    parser.add_argument("--past_pop_version", type=str, required=True,
        help="Which version of past populations to use")
    parser.add_argument("--forecast_pop_version", type=str, required=True,
        help="Which version of forecasted populations to use")
    parser.add_argument("--gbd_round_id", type=int, required=True,
        help="Which gbd round id to use for populations")
    parser.add_argument("--draws", type=int, required=True,
        help="How many draws to use")
    parser.add_arg_years(required=True)
    args = parser.parse_args()

    main(migration_version=args.migration_version,
        past_pop_version=args.past_pop_version,
        forecast_pop_version=args.forecast_pop_version,
        gbd_round_id=args.gbd_round_id,
        draws=args.draws,
        years=args.years)
    great_job.congratulations() # You did it!


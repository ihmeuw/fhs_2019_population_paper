"""
Calculate number of deaths and number of births
"""
import logging

import pandas as pd
import xarray as xr

from fbd_core import argparse, db
from fbd_core.file_interface import FBDPath, open_xr, save_xr
from fbd_core.etl import aggregator, resample, expand_dimensions


LOGGER = logging.getLogger(__name__)

ALL_FERTILE_AGES_ID = 169
FERTILE_AGE_GROUP_IDS = tuple(range(7, 15 + 1))

ALL_NEONATAL_AGES_ID = 42
NEONATAL_AGE_IDS = (2, 3)

ALL_UNDER_5_AGES_ID = 1
UNDER_5_AGE_IDS = (2, 3, 4, 5)

BOTH_SEX_ID = 3
FEMALE_ID = 2

COUNTRY_LEVEL = 3


def main(forecast_pop_version, input_version, output_version,
         draws, gbd_round_id, measure, years, past_pop_version,
         past_asfr_version, past_mortality_version):
    """Compute deaths or births aggregations and save the data

    Args:
        forecast_pop_version (str):
            The version name of the population forecasts.
        gbd_round_id (int):
            The GBD round fed into FBDPath to pull the correct version of ASFR.
        input_version (str):
            The version name of the input mortality or ASFR
        output_version (str):
            The version name of the output deaths or births to be saved.
        measure (str):
            Death or live_births.
        draws (int):
            The number of desired draws.
    Returns: None.
    """
    forecast_pop = get_pop(forecast_pop_version, gbd_round_id, measure, draws,
                           years, past_pop_version)

    if measure == "live_births":
        forecast_da = get_asfr(input_version, gbd_round_id, draws, years,
                               past_asfr_version)
    else:
        forecast_da = get_mortality(input_version, gbd_round_id, draws, years,
                                    past_mortality_version)

    agg_output = get_agg(forecast_pop, forecast_da, gbd_round_id)

    data_path = FBDPath(
        f"{gbd_round_id}/future/{measure}/{output_version}")
    save_xr(agg_output, f"{data_path}/{measure}.nc", metric="number", space="identity")

    LOGGER.info(f"{measure} have been calculated")


def get_agg(forecast_pop, da, gbd_round_id):
    """Use aggregator function to get aggregated births or deaths.

    Args:
        forecast_pop (xarray.DataArray):
            Forecast populations.
        da (xarray.DataArray):
            Mortality or ASFR forecasts
        gbd_round_id (int):
            The GBD round fed into FBDPath.
    Returns:
        (xarray.DataArray):
            Livebirths or deaths.
    """
    loc = db.get_locations_by_max_level(3)
    loc_hierarchy = loc.set_index("location_id").to_xarray()["parent_id"]
    da.attrs["metric"] = "rate"
    region_scalars = aggregator._get_regional_correction_factors(gbd_round_id)
    agg = aggregator.Aggregator(forecast_pop)
    output = agg.aggregate_locations(loc_hierarchy, data=da,
                                     correction_factor=region_scalars)
    output = output.number
    return output


def concat_past_future(past_da, forecast_da, draws, years):
    """Combine past at last past year and future data.

    Args:
        past_da (xarray.DataArray):
            Past data.
        forecast_da (xarray.DataArray):
            Forecast data.
        gbd_round_id (int):
            The GBD round fed into FBDPath.
        draws (int):
            Number of draws.
    Returns: (xarray.DataArray):
        Combined past and future data.
    """
    past_da = past_da.sel(year_id=years.past_end,
                          age_group_id=forecast_da.age_group_id.values,
                          location_id=forecast_da.location_id.values)
    forecast_da = forecast_da.sel(year_id=years.forecast_years,
                                  scenario=0).drop("scenario")
    forecast_da = resample(forecast_da, draws)
    past_da = resample(past_da, draws)
    combined_da = xr.concat([past_da, forecast_da], dim="year_id")
    return combined_da


def get_pop(forecast_pop_version, gbd_round_id, measure, draws, years,
            past_pop_version):
    """Pulls specified version of populations, subsets to fertile age groups
    and females only if meausre is live_births.

    Args:
        gbd_round_id (int):
            The GBD round fed into FBDPath to pull the correct version of pops
        forecast_pop_version (str):
            The version name of the populations file used in FBDPath.
        draws (int):
            The number of desired draws. This goes into resample, so we get
            pops with the correct number of draws.
    Returns:
        (xarray.DataArray):
            Fertile forecast population. The ``age_group_id`` dimension
            includes coordinates for each of the fertile age-groups.
    """
    forecast_pop_path = FBDPath(
        f"{gbd_round_id}/future/population/{forecast_pop_version}")
    forecast_pop_file = forecast_pop_path / "population.nc"
    forecast_pop = open_xr(forecast_pop_file).data
    past_pop_path = FBDPath(
        f"{gbd_round_id}/past/population/{past_pop_version}")
    past_pop_file = past_pop_path / "population.nc"
    past_pop = open_xr(past_pop_file).data
    past_pop = past_pop.sel(sex_id=forecast_pop.sex_id.values)
    past_pop = expand_dimensions(past_pop, draw=range(draws))
    forecast_pop = concat_past_future(past_pop, forecast_pop, draws, years)

    if measure == "live_births":
        forecast_pop = forecast_pop.sel(
            age_group_id=list(FERTILE_AGE_GROUP_IDS),
            sex_id=2
        ).drop(["sex_id"])
    else:
        forecast_pop = forecast_pop.sel(sex_id=[1, 2])

    return forecast_pop


def get_asfr(forecast_asfr_version, gbd_round_id, draws, years, past_asfr_version):
    """Pulls specified version of ASFR, subsets to females only if sex_id
        is a dimension.

    Args:
        forecast_asfr_version (str):
            The future version name of the ASFR file used in FBDPath.
        past_asfr_version (str):
            The past version name of the ASFR file used in FBDPath.
        gbd_round_id (int):
            The GBD round fed into FBDPath to pull the correct version of ASFR.
        draws (int):
            The number of desired draws.
    Returns:
        (xarray.DataArray):
            Age-specific fertility rate.
    """
    forecast_asfr_path = FBDPath(
        f"{gbd_round_id}/future/asfr/{forecast_asfr_version}")
    forecast_asfr = open_xr(forecast_asfr_path / "asfr.nc").data
    past_asfr_path = FBDPath(
        f"{gbd_round_id}/past/asfr/{past_asfr_version}")
    past_asfr_file = past_asfr_path / "asfr.nc"
    past_asfr = open_xr(past_asfr_file).data
    forecast_asfr = concat_past_future(past_asfr, forecast_asfr, draws, years)

    if 'sex_id' in forecast_asfr.dims:  # sex_id is dimension
        forecast_asfr = forecast_asfr.sel(sex_id=2, drop=True)
    elif 'sex_id' in forecast_asfr.coords:  #sex-id is point coordinate
        forecast_asfr = forecast_asfr.drop('sex_id')
    else:
        pass  # do nothing -- sex_id doesn't exist

    return forecast_asfr


def get_mortality(forecast_mortality_version, gbd_round_id, draws, years,
                  past_mortality_version):
    """Pulls specified version of mortality.

    Args:
        forecast_mortality_version (str):
            The version name of the future mortality file used in FBDPath.
        past_mortality_version (str):
            The version name of the past mortality file used in FBDPath.
        gbd_round_id (int):
            The GBD round fed into FBDPath to pull the correct version of mortality.
        draws (int):
            The number of desired draws.
    Returns:
        (xarray.DataArray):
            Mortality rate.
    """
    forecast_mot_path = FBDPath(
        f"{gbd_round_id}/future/death/{forecast_mortality_version}")
    forecast_mot = open_xr(forecast_mot_path / "_all.nc").data
    past_mot_path = FBDPath(
        f"{gbd_round_id}/past/death/{past_mortality_version}")
    past_mot_file = past_mot_path / "_all.nc"
    past_mot = open_xr(past_mot_file).data
    past_mot = past_mot.drop("acause")
    forecast_mot = concat_past_future(past_mot, forecast_mot, draws, years)
    return forecast_mot



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--input-version",
        type=str,
        required=True,
        help="Version of input future mortaltiy or ASFR.")
    parser.add_argument(
        "--forecast-pop-version",
        type=str,
        required=True,
        help="Version of input future population")
    parser.add_argument(
        "--output-version",
        type=str,
        required=True,
        help="Version of outputs")
    parser.add_argument(
        "--gbd-round-id",
        type=int,
        default=5,
        help="Numeric ID for the GBD round.")
    parser.add_argument(
        "--measure",
        type=str,
        required=True,
        help="death or live_births")
    parser.add_argument(
        "--past-pop-version",
        type=str,
        default="20181206_pop_1950_2017",
        help="Version of input past population")
    parser.add_argument(
        "--past-asfr-version",
        type=str,
        default="20190109_va84",
        help="Version of input past ASFR")
    parser.add_argument(
        "--past-mortality-version",
        type=str,
        default="20190424_fix_rota_agg",
        help="Version of input past mortality")
    parser.add_arg_draws(required=True)
    parser.add_arg_years(required=True)
    args = parser.parse_args()

    main(args.forecast_pop_version, args.input_version, args.output_version,
         args.draws, args.gbd_round_id, args.measure, args.years,
         args.past_pop_version, args.past_asfr_version,
         args.past_mortality_version)

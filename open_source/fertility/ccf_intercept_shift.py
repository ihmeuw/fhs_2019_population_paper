r"""
ASFRs are intercept-shifted through the forecast years while keeping CCF50
(cohort 15-49 ASFR sum) constant:

1. Read in location-specific draws of raw period ASFR from CCF stage
   (only has age group ids 8-14).
2. Intercept shift asfr by holding CCF50 constant.
3. Append terminal age group ASFR's (age group ids 7 and 15,
   because the pop code expects them)
4. Export location-specific intercept-shifted ASFR in netCDF

Example call:

.. code:: bash
    python ccf_intercept_shift.py --asfr-version ASFR_VERSION
    --past-asfr-version PAST_ASFR_VERSION --location-id 102 --gbd-round-id 5
    --years 1990:2018:2100 --granularity 1
"""
import gc
import logging

import numpy as np
import pandas as pd
import xarray as xr

from fbd_core import argparse, db
from fbd_core.etl import df_to_xr, expand_dimensions
from fbd_core.file_interface import FBDPath, open_xr, save_xr

LOGGER = logging.getLogger(__name__)

CCF50_LOWER_AGE = 15  # first age for CCF50
CCF50_UPPER_AGE = 49  # last age for CCF50
ASFR_NON_AGE_DIMS = ("location_id", "year_id", "scenario", "draw")
RTOL = 1e-3  # acceptable relative tolerance for CCF50 numerical error


def read_to_xr(location_id, ccf_asfr_fbd_path, dims):
    """
    Reads location-specific csv file into pd.DataFrame, and then return as
    xr.DataArray.

    Args:
        location_id (int): location id.
        ccf_asfr_fbd_path (FBDPath): folder path within ccf stage.  Contains an
            "asfr_single_year" or "asfr" sub-folder, where location-specific
            csv files are stored.
        dims (list[str]): list of dims expected from upstream asfr csv files.

    Returns:
        (xr.DataArray): location-specific asfr, not yet intercept-shifted.
    """
    file_fbd_path = ccf_asfr_fbd_path / f"{location_id}.csv"

    asfr = pd.read_csv(file_fbd_path)
    asfr = df_to_xr(asfr, dims=dims).sel(location_id=location_id)
    return asfr


def extrapolate_terminal_asfr_age_groups(past_asfr, future_asfr, last_year):
    """
    Extrapolates age groups ids 7 and 15 for future asfr.  The extrapolation is
    done by keeping the 7-to-8 and 15-to-14 asfr ratios the same as
    those of the last past year provided.

    Args:
        past_asfr (xr.DataArray): past asfr dataarray.
        future_asfr (xr.DataArray): forecasted asfr dataarray.
        last_year (int): last past year.

    Returns:
        (xr.DataArray): asfr with age group ids 7-15.
    """
    past_young_ratio = (past_asfr.sel(year_id=last_year, age_group_id=7) /
                        past_asfr.sel(year_id=last_year, age_group_id=8))
    future_7 = future_asfr.sel(age_group_id=8) * past_young_ratio
    future_7["age_group_id"] = 7

    past_old_ratio = (past_asfr.sel(year_id=last_year, age_group_id=15) /
                      past_asfr.sel(year_id=last_year, age_group_id=14))
    future_15 = future_asfr.sel(age_group_id=14) * past_old_ratio
    future_15["age_group_id"] = 15

    out = xr.concat([future_7, future_asfr, future_15],
                    dim="age_group_id").sortby("age_group_id")
    assert xr.ufuncs.isfinite(out).all(), "Non-finite values in extended ASFR"

    return out


def _expand_5yr_age_groups_to_1yr_ages(da, ages_df):
    """
    Converts 5-year age groups to 1-year ages, by simply repeating the same
    value.

    Args:
        da (xr.DataArray): da with "age_group_id" dim.
        ages_df (pd.DataFrame): df with age group metadata.

    Returns:
        (xr.DataArray) da where "age_group_id" dim is replaced with "age".
    """
    assert "age_group_id" in da.dims, "Missing age_group_id dim"
    for col in ["age_group_id", "age_group_years_start",
                "age_group_years_end"]:
        assert col in ages_df.columns, f"Missing {col} column"
    assert da["age_group_id"].isin(ages_df["age_group_id"]).all(),\
        "Not all age group ids are available in metadata"
    das = []
    for age_group_id in da["age_group_id"].values:
        lower_age = int(ages_df.query("age_group_id == @age_group_id")
                        ["age_group_years_start"])
        upper_age = int(ages_df.query("age_group_id == @age_group_id")
                        ["age_group_years_end"])
        sub_da = da.sel(age_group_id=age_group_id).drop("age_group_id")
        sub_da = expand_dimensions(sub_da, age=range(int(lower_age),
                                                     int(upper_age)))
        das.append(sub_da)

    return xr.concat(das, dim="age")


def _convert_ages_to_5_year_age_groups_by_mean(da, ages_df):
    """
    Converts the ages dimension to 5-year age groups, by taking the mean
    of the values within the age group years.

    Args:
        da (xr.DataArray): da with "age" dim.
        ages_df (pd.DataFrame): df with age group metadata.

    Returns:
        (xr.DataArray) da where "age" dim is replaced with "age_group_id".
    """
    min_age, max_age = da["age"].min(), da["age"].max()
    ages_df = ages_df.sort_values(by="age_group_years_start")
    min_age_start = int(ages_df.
                        query("@min_age >= age_group_years_start & "
                              "@min_age < age_group_years_end")
                        ["age_group_years_start"])
    max_age_start = int(ages_df.
                        query("@max_age >= age_group_years_start & "
                              "@max_age < age_group_years_end")
                        ["age_group_years_start"])

    das = []
    for age_start in range(min_age_start, max_age_start + 1, 5):
        # in case where not all 5 age years are present, take what's there
        ages = [a for a in da["age"].values.tolist()
                if a in range(age_start, age_start + 5)]
        da_sub = da.sel(age=ages).mean("age")  # yes we mean over the 5 yrs
        da_sub["age_group_id"] =\
            int(ages_df.query("age_group_years_start == @age_start")
                ["age_group_id"])
        das.append(da_sub)

    return xr.concat(das, dim="age_group_id").sortby("age_group_id")


def ccf50_intercept_shift_lpf(asfr, gbd_round_id, years, iterations,
                              window_size=3):
    """
    CCF50 intercept-shift of future years using a rolling window average to
    smooth out the kinks.  Window defaults to centered with size=window_size.

    Boundary conditions:
    1.) the past does not change.  Smoothing starts with years.forecast_start
        as the center of the left-most window.
    2.) years.forecast_end will only have 2 points contributing to its rolled
        average, because there's no year after it.

    The rolling average is done n=hyperparam times before raking back all
    cohort values to the CCF50 they started out with.

    Prior to any smoothing, the CCF50 of every cohort is computed and stored.

    Args:
        asfr (xr.DataArray): only has age group ids 8-14.  Should also
            have year_ids > years.forecast_start - CCF_UPPER_AGE.
            Has dims [location_id, age_group_id, scenario, year_id, draw]
        gbd_round_id (int): gbd round id.
        years (YearRange): past_start:forecast_start:forecast_end.
        iterations (int): number of time to perform moving average.
        window_size (int, optional): window size.  Defaults to 3 to maintain
        a minimum centered window.  This is to smooth while preserving local
        structure.  Unless very good reasons are given, it should stay at 3.

    Returns:
        (xr.DataArray): corrected asfr.
    """
    # create cohort non-index dim.  A cohort is defined by year of birth.
    asfr = asfr.assign_coords(cohort=asfr["year_id"] - asfr["age"])

    # CCF50 is kept constant for every cohort.
    # Need to store what each cohort's asfr sum should be in the future years.
    asfr_f = asfr.sel(year_id=years.forecast_years)
    ccf50_cohorts = []
    for cohort in np.unique(asfr_f["cohort"].values).tolist():
        ccf50 = asfr_f.where(asfr["cohort"] == cohort).sum(["age", "year_id"])
        ccf50["cohort"] = cohort  # dataarray indexed by location/scenario/draw
        ccf50_cohorts.append(ccf50)

    # now a dataarray indexed by cohort/location/scenario/draw
    ccf50_cohorts = xr.concat(ccf50_cohorts, dim="cohort")

    del asfr_f, ccf50
    gc.collect()

    for _ in range(iterations):
        # now asfrs from years.past_end to forecast_end
        smoothed_asfr = asfr.sel(year_id=range(years.past_end,
                                               years.forecast_end + 1))
        # replace years.forecast_start:forecast_end - 1 with rolling avg.
        # NOTE the window size of 3 and the years.past_end are tied together
        smoothed_asfr =\
            smoothed_asfr.rolling(year_id=window_size,
                                  min_periods=2,
                                  center=True)\
            .mean().combine_first(smoothed_asfr)

        asfr.loc[dict(year_id=years.forecast_years)] =\
            smoothed_asfr.sel(year_id=years.forecast_years)

    gc.collect()

    # now go through each cohort in future years to "rake"
    smoothed_asfr = smoothed_asfr.sel(year_id=years.forecast_years)
    for cohort in np.unique(smoothed_asfr["cohort"].values).tolist():
        cohort_future_asfr =\
            smoothed_asfr.where(smoothed_asfr["cohort"] == cohort)
        # the following are in location/scenario/draw
        actual = cohort_future_asfr.sum(["year_id", "age"])
        expected = ccf50_cohorts.sel(cohort=cohort)
        cohort_future_asfr *= expected / actual  # raking

        smoothed_asfr = cohort_future_asfr.combine_first(smoothed_asfr)

    del actual, expected, cohort_future_asfr
    gc.collect()

    # now assign the modified asfrs back to the original data
    asfr.loc[dict(year_id=years.forecast_years)] = smoothed_asfr

    asfr.attrs["iterations"] = iterations

    return asfr


def main(asfr_version, past_asfr_version, location_id, gbd_round_id, years,
         granularity, iterations, **kwargs):
    """
    1. Read in location-specific draws of period ASFR from CCF stage
    2. Add terminal age group ASFR's
    3. Intercept shift asfr by holding CCF50 constant.
    4. Export location-specific intercept-shifted ASFR in .nc

    Args:
        asfr_version (str): version name of future ccf/asfr.
        past_asfr_version (str): asfr version from past.
        location_id (int): location_id.
        gbd_round_id (int): gbd round id.
        years (YearRange): past_start:forecast_start:forecast_end
        iterations (int): number of times to intercept-shift.
    """
    ages_df = db.get_ages(gbd_round_id)[["age_group_id",
                                         "age_group_years_start",
                                         "age_group_years_end"]]

    # read the location-specific asfr .csv into dataarray
    # the raw forecasted ASFR are stored in the CCF stage of the same
    ccf_fbd_path = FBDPath(gbd_round_id=gbd_round_id, past_or_future="future",
                           stage="ccf", version=asfr_version)
    if granularity == 1:
        sub_folder = "asfr_single_year"
        ccf_asfr_fbd_path = ccf_fbd_path / sub_folder
        future_asfr = read_to_xr(location_id, ccf_asfr_fbd_path,
                                 dims=list(ASFR_NON_AGE_DIMS + ("age",)))
    else:
        sub_folder = "asfr"
        ccf_asfr_fbd_path = ccf_fbd_path / sub_folder
        future_asfr =\
            read_to_xr(location_id, ccf_asfr_fbd_path,
                       dims=list(ASFR_NON_AGE_DIMS + ("age_group_id",)))
        # we intercept-shift in 1-year ages, so convert to single years
        future_asfr = _expand_5yr_age_groups_to_1yr_ages(future_asfr, ages_df)

    if "sex_id" in future_asfr.dims:
        raise ValueError("Found sex_id dim in future asfr")

    # now etl the past asfr data
    past_asfr_fbd_path = FBDPath(gbd_round_id=gbd_round_id,
                                 past_or_future="past",
                                 stage="asfr",
                                 version=past_asfr_version)
    past_asfr =\
        open_xr(past_asfr_fbd_path /
                "asfr.nc").data.sel(location_id=location_id)

    if "sex_id" in past_asfr.dims:
        raise ValueError("Found sex_id dim in past asfr")

    # past has no scenarios, so we need to expand it for merging
    past_asfr = expand_dimensions(past_asfr, scenario=future_asfr["scenario"])

    # past asfr has age group ids 7-15, but future asfr in ccf only has 8-14.
    # we only need age groups 8-14 for intercept shift
    past_asfr_1yrs = _expand_5yr_age_groups_to_1yr_ages(
        past_asfr.sel(age_group_id=range(8, 15)), ages_df)

    # now ready to concat past and future together for intercept shift
    asfr = xr.concat([past_asfr_1yrs.sel(year_id=years.past_years),
                      future_asfr.sel(year_id=years.forecast_years)],
                     dim="year_id")

    del past_asfr_1yrs, future_asfr 
    gc.collect()

    # the intercept-shift should keep ccf50 (asfr sum) constant
    pre_fix_asfr_sum = asfr.sum()  # sum of all asfr values before shift

    asfr = ccf50_intercept_shift_lpf(asfr, gbd_round_id, years, iterations)

    post_fix_asfr_sum = asfr.sum()  # asfr sum post-shift should stay the same

    assert np.isclose(post_fix_asfr_sum, pre_fix_asfr_sum, rtol=RTOL),\
        f"The intercept shift changed total asfr sum by more than rtol={RTOL}"

    # need to save years.past_end for cohort-component model
    save_years = [years.past_end] + years.forecast_years.tolist()
    asfr = asfr.sel(year_id=save_years)  # only do forecast
    # convert forecasted asfr back to 5-year age groups
    asfr = _convert_ages_to_5_year_age_groups_by_mean(asfr, ages_df)
    # add 10-15 (7) and 50-55 (15) age groups for forecasted asfr
    asfr = extrapolate_terminal_asfr_age_groups(past_asfr, asfr,
                                                last_year=years.past_end)
    asfr["location_id"] = location_id
    asfr.name = "value"

    del past_asfr
    gc.collect()

    LOGGER.info("Finished CCF50 intercept-shift")

    asfr_fbd_path = FBDPath(gbd_round_id=gbd_round_id, past_or_future="future",
                            stage="asfr", version=asfr_version)

    save_xr(asfr, asfr_fbd_path / f"{location_id}.nc", metric="rate",
            space="identity", version=asfr_version,
            past_asfr_version=past_asfr_version, iterations=iterations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--asfr-version", required=True,
                        help="Age-specific fertility rate")
    parser.add_argument("--past-asfr-version", required=True,
                        help="Past age-specific fertility rate")
    parser.add_argument(
        "--gbd-round-id", type=int, required=True,
        help="The gbd round id associated with the data.")
    parser.add_argument("--location-id", type=int)
    parser.add_argument("--granularity", type=int, choices={1, 5}, default=1,
                        help="Age granularity of upstream period ASFR files. "
                             "Some times the upstream could provide asfr in "
                             "5-year age groups, it's up to the user to "
                             "inform the script on the upstream format. "
                             "Default is 1-year ages.")
    parser.add_argument("--iterations", required=True, type=int,
                        help="# of iterations to run rolling average.")
    parser.add_arg_years(required=True)

    args = parser.parse_args()

    main(**args.__dict__)

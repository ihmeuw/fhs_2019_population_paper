r"""
From asfr and pop, make asfr_agg, tfr, and tfr_agg, and export
files for pipeline and plotting needs.

Example call:

.. code:: bash
    python make_tfr_and_agg.py --asfr-version ASFR_VERSION
    --pop-version POP_VERSION --gbd-round-id 5 --years 1990:2018:2100
"""
import logging
import xarray as xr

from fbd_core import argparse, db
from fbd_core.etl import Aggregator
from fbd_core.file_interface import FBDPath, open_xr, save_xr

LOGGER = logging.getLogger(__name__)

ASFR_AGE_GROUP_IDS = range(7, 16)


def calc_tfr_from_asfr(data):
    """Compute TFR from ASFR.
    TFR is defined as the sum of single-year-age asfr values
    over all female fertile ages (10-54 yrs of age), which
    spans age group ids 7-15.  This is done by simply
    multiplying the age-group asfrs by 5 and collect the sum.

    Args:
        data (xr.DataArray): asfr with age group ids 7-15.

    Returns:
        (xr.DataArray): tfr values.
    """
    assert "age_group_id" in data.dims, "Missing age_group_id dim in input"
    # ensure age group ids span from 7-15, as required.
    assert data["age_group_id"].\
        equals(xr.DataArray(ASFR_AGE_GROUP_IDS,
                            coords=[('age_group_id', ASFR_AGE_GROUP_IDS)])),\
        "Input must have age group ids 7-15"

    tfr = 5 * data.sum("age_group_id", skipna=False)
    return tfr


def make_tfr_and_agg(asfr_version, pop_version, gbd_round_id, years,
                     model, hyperparam, **kwargs):
    """
    From asfr and pop, make asfr_agg, tfr, and tfr_agg, and export
    files for pipeline and plotting needs.

    Args:
        asfr_version (str): intercept-shifted asfr version where an "asfr.nc"
            with both past and future is present.
        pop_version (str): future pop version to use for agg.
        gbd_round_id (int): gbd round id.
        years (YearRange): past_start:forecast_start:forecast_end.
    """
    pop_fbd_path = FBDPath(gbd_round_id=gbd_round_id,
                           past_or_future="future",
                           stage="population",
                           version=pop_version)
    # only need females for fertility studies
    pop = open_xr(pop_fbd_path / "population.nc").data.\
        sel(sex_id=2, year_id=years.forecast_years)

    agg = Aggregator(pop)
    locs = db.get_locations_by_max_level(3) 
    hierarchy = locs[["location_id", "parent_id"]].\
        set_index("location_id").to_xarray().parent_id

    asfr_fbd_path = FBDPath(gbd_round_id=gbd_round_id,
                            past_or_future="future",
                            stage="asfr",
                            version=asfr_version)
    asfr = open_xr(asfr_fbd_path / "asfr.nc").data.\
        sel(year_id=years.forecast_years)
    asfr_agg = agg.aggregate_locations(hierarchy, data=asfr).rate

    # Calculate TFR
    tfr = calc_tfr_from_asfr(asfr)
    tfr_agg = calc_tfr_from_asfr(asfr_agg)

    # Saving to .nc files
    asfr.name = "value"
    tfr.name = "value"
    asfr_agg.name = "value"
    tfr_agg.name = "value"

    LOGGER.info("saving asfr_agg, tfr, tfr_agg to .nc")
    save_xr(asfr_agg, asfr_fbd_path / "asfr_agg_based_on_preliminary_pop.nc",
            metric="rate", space="identity",
            asfr_version=asfr_version, pop_version=pop_version)

    tfr_fbd_path = FBDPath(gbd_round_id=gbd_round_id,
                           past_or_future="future",
                           stage="tfr",
                           version=asfr_version)
    save_xr(tfr, tfr_fbd_path / "tfr.nc", metric="rate", space="identity",
            asfr_version=asfr_version)

    save_xr(tfr_agg, tfr_fbd_path / "tfr_agg_based_on_preliminary_pop.nc",
            metric="rate", space="identity",
            asfr_version=asfr_version, pop_version=pop_version)

    print("Saving Quantiles and Means to .csv")
    asfr.mean("draw").to_dataframe().reset_index().\
        to_csv(asfr_fbd_path / "asfr_mean.csv", index=False)
    asfr_quantiles = asfr.quantile([0.025, 0.975], "draw")
    asfr_quantiles.sel(quantile=0.025).to_dataframe().reset_index().\
        to_csv(asfr_fbd_path / "asfr_lower.csv", index=False)
    asfr_quantiles.sel(quantile=0.975).to_dataframe().reset_index().\
        to_csv(asfr_fbd_path / "asfr_upper.csv", index=False)

    asfr_agg.mean("draw").to_dataframe().reset_index().\
        to_csv(asfr_fbd_path / "asfr_agg_based_on_preliminary_pop_mean.csv",
               index=False)
    asfr_agg_quantiles = asfr_agg.quantile([0.025, 0.975], "draw")
    asfr_agg_quantiles.sel(quantile=0.025).to_dataframe().reset_index().\
        to_csv(asfr_fbd_path / "asfr_agg_based_on_preliminary_pop_lower.csv",
               index=False)
    asfr_agg_quantiles.sel(quantile=0.975).to_dataframe().reset_index().\
        to_csv(asfr_fbd_path / "asfr_agg_based_on_preliminary_pop_upper.csv",
               index=False)

    tfr.mean("draw").to_dataframe().reset_index().\
        to_csv(tfr_fbd_path / "tfr_mean.csv", index=False)
    tfr_quantiles = tfr.quantile([0.025, 0.975], "draw")
    tfr_quantiles.sel(quantile=0.025).to_dataframe().reset_index().\
        to_csv(tfr_fbd_path / "tfr_lower.csv", index=False)
    tfr_quantiles.sel(quantile=0.975).to_dataframe().reset_index().\
        to_csv(tfr_fbd_path / "tfr_upper.csv", index=False)

    tfr_agg.mean("draw").to_dataframe().reset_index().\
        to_csv(tfr_fbd_path / "tfr_agg_based_on_preliminary_pop_mean.csv",
               index=False)
    tfr_agg_quantiles = tfr_agg.quantile([0.025, 0.975], "draw")
    tfr_agg_quantiles.sel(quantile=0.025).to_dataframe().reset_index().\
        to_csv(tfr_fbd_path / "tfr_agg_based_on_preliminary_pop_lower.csv",
               index=False)
    tfr_agg_quantiles.sel(quantile=0.975).to_dataframe().reset_index().\
        to_csv(tfr_fbd_path / "tfr_agg_based_on_preliminary_pop_upper.csv",
               index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make asfr_agg, tfr, tfr_agg")
    parser.add_argument("--asfr-version", required=True,
                        help="Version of the loc-specific asfrs to collect.")
    parser.add_argument("--pop-version", type=str, required=True,
                        help="needed for aggregation")
    parser.add_argument("--gbd-round-id", type=int, required=True,
                        help="GBD round id")
    parser.add_argument("--model", required=True, type=str,
                        help="algo for intercept-shift")
    parser.add_argument("--hyperparam", required=True, type=int,
                        help="the one hyperparameter for the intercept-shift "
                             "code.  If # of iterations, will be taken as is. "
                             "If fraction, give 75 for 0.75.")
    parser.add_arg_years(required=True)

    args = parser.parse_args()

    make_tfr_and_agg(**args.__dict__)

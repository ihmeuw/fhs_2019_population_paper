"""This module aggregates the expected value of mortality or ylds up the cause
hierarchy and computes the following at each step:

    y_hat = expected value of mortality or ylds
    y_past = past mortality or ylds
    epsilon_past = y_hat - y_past
    epsilon_hat = RW(epsilon_past)
    y_star = y_hat + epsilon_hat

The following results are saved in their own directory:
    - y_hat in log rate space
    - epsilon_hat in log rate space
    - y_star in log rate space

It is assumed that the inputs (modeled cause specific mortality or ylds with
version <version>) contain y_hat data in log space. That is, only draws of the
expected value.

Spaces:
    - Modeled cause results split by sex are in log rate space.
    - Modeled cause results not split by sex are in normal rate
      space.
    - Past cause mortality or ylds are in normal space.
    - Summing to aggregate causes happens in normal rate space.
    - Computing the difference between y_bar and y_past happens in log rate
      space.
    - ARIMA for epsilon happens in log rate space for the mean of the baseline
      scenario.
    - Summing y_hat and epsilon_hat happens in log rate space.
    - All results produced by this script are saved in log rate space.
"""

import logging

import numpy as np
import pandas as pd
from sqlalchemy.orm import sessionmaker
import xarray as xr

from fbd_core import db, argparse
from fbd_core.etl.transformation import resample, bias_exp
from fbd_core.file_interface import FBDPath
from fbd_core.file_interface import save_xr
from fbd_core.strategy_set import get_hierarchy
from fbd_core.strategy_set.strategy import get_strategy_set
from fbd_model.model.PooledRandomWalk import PooledRandomWalk
from fbd_model.model import RandomWalk as rw
from fbd_scenarios import ar1_utils, remove_drift, subset_fatal
from fbd_scenarios import gk_intercept_shift as gis


__module__ = "fbd_scenarios.aggregate_with_error_arima"
logger = logging.getLogger(__module__)

NTD_CAUSES = ("_ntd", "malaria", "ntd_chagas", "ntd_leish", "ntd_leish_visc",
              "ntd_leish_cut", "ntd_afrtryp", "ntd_schisto", "ntd_cysticer",
              "ntd_echino", "ntd_lf", "ntd_oncho", "ntd_trachoma", "ntd_dengue",
              "ntd_yellowfever", "ntd_rabies", "ntd_nema", "ntd_nema_ascar",
              "ntd_nema_trichur", "ntd_nema_hook", "ntd_foodborne", "ntd_other")


def aggregate_yhats(acause, input_version, agg_version, measure, period,
                    draws, gbd_round_id, dryrun=False):
    """Computes y hats.

    If acause is a modeled cause, then it is simply moved into the right
    folder. Otherwise, the subcauses of acause are aggregated.

    :param str acause: name of the target acause to aggregate to.
    :param str input_version: name of the mort or yld version the aggregate is
        based on.
    :param str agg_version: name of the aggregate version.
    :param str measure: death or yld
    :param str period: future or past
    :param int gbd_round_id: the gbd round providing the past estimates
    :param bool dryrun: dryrun flag. This is a test run if True.
    """
    y_hat = _get_y_hat(acause, input_version, agg_version, measure, period,
                       draws, gbd_round_id)
    num_draws = len(y_hat["draw"].values)
    logger.info("The data has {} draws.".format(num_draws))
    if not dryrun:
        # Save data!
        logger.info("Saving y_hat to {}".format(FILEPATH))
        _save_netcdf(y_hat, FILEPATH)


def arima_and_ystar(acause, agg_version, arima_version, smoothing, years,
                    measure, intercept_shift, gbd_round_id, draws, decay,
                    dryrun=False, no_correction=False,
                    past_version="best", no_arima=False, **kwargs):
    r"""Samples mortality residuals from an ARIMA and forms
    $y^* = \hat{y} + \hat{\epsilon}$.

    :param str acause: name of the target acause to aggregate to.
    :param str agg_version: name of the aggregate version.
    :param str arima_version: name of the arima version.
    :param list[str] smoothing: what dimensions to smooth over during the ARIMA
        step.
    :param fbd_core.argparse.YearRange years: a container for the three years
        which define our forecast.
    :param int draws: number of draws to take.
    :param bool dryrun: dryrun flag. This is a test run if True.
    :param bool bias: Perform log bias correction.
    """
    logger.debug("Opening: {}".format(FILEPATH))
    y_hat = xr.open_dataarray(str(FILEPATH))

    # GK intercept shift
    y_hat = gis.intercept_shift_at_draw(y_hat, acause, past_version,
                                        gbd_round_id, years, draws)
    save_xr(y_hat, FILEPATH, root_dir = "scratch", metric = "rate", space = "log")

    y_past = _get_y_past(acause, years, measure, gbd_round_id,
                         past_version=past_version)

    past_years = years.past_years

    if not no_arima:
        # ARIMA for everything except NTDs
        logger.info("Computing epsilon_past.")
        epsilon_past_with_scenarios_and_draws = (
            y_past.loc[dict(year_id=past_years)] -
            y_hat.loc[dict(year_id=past_years)])
        epsilon_past = epsilon_past_with_scenarios_and_draws.loc[
            dict(scenario=0)].mean("draw")

        try:
            epsilon_hat = xr.open_dataarray(str(FILEPATH))
        except:
            epsilon_hat = _draw_epsilons(
                    epsilon_past, draws, smoothing, years, acause, decay,
                    gbd_round_id=gbd_round_id)
        if not dryrun:
            logger.info("Saving epsilon_hat to {}".format(FILEPATH))
            _save_netcdf(epsilon_hat, FILEPATH)
        y_star = _get_y_star(y_hat, epsilon_hat, years).copy()

    else:
        # no arima for ntds
        y_star = y_hat
        y_star.name = "value"

    # intercept shift and bias
    if intercept_shift:
        y_star = _intercept_shift(acause, y_star, years, measure, gbd_round_id,
                                  draws=draws, no_arima=no_arima,
                                  past_version=past_version)
    if not no_correction:
        y_star = xr.ufuncs.log(bias_exp(y_star))

    if not dryrun:
        logger.info("Saving y_star to {}".format(FILEPATH))
        _save_netcdf(y_star, FILEPATH)


def _intercept_shift(acause, y_star, years, measure, gbd_round_id, draws=100,
                     no_arima=False, past_version="best"):
    """Incorporates past uncertainty by intercept-shifting future draws by the
    draw-level residual for the last year of the past. For non-ntd causes,
    the shift is the residual of the last-past-year draws to the last-past-year
    mean (data only, not modeled draws or means), while for ntds, the shift is
    the draw-level residual between the modeled past (draws) and the past data
    (draws).

    :param str acause: name of the acause for which the intercept shift is
        desired
    :param xarray.DataArray y_star: the draw-level arima-ed estimates for the
        log of the measure and acause in question
    :param list[int] years: [past_start, forecast_start, forecast_end]
    """
    past_draws = _get_y_past(acause, years, measure, gbd_round_id, draws=draws,
                             draw_level=True, last_year_only=True,
                             past_version=past_version)
    modeled_draws = y_star.loc[{"year_id": years.forecast_start - 1,
                                "scenario": 0}].drop("scenario")

    # make sure the number of draws matches up
    num_draws = len(modeled_draws.draw)
    # get the shift values from the past data if there was an arima, but
    # from the modeled past if there wasn't
    if not no_arima:
        past_mean = past_draws.mean("draw")
        sampled_past_draws = resample(past_draws, num_draws)
        shift = -1. * (sampled_past_draws - past_mean).drop("year_id")
    else:
        sampled_modeled_draws = resample(modeled_draws, num_draws)
        sampled_past = resample(past_draws, num_draws)
        sampled_past = sampled_past.transpose(
                *sampled_modeled_draws.coords.dims)
        shift = (sampled_modeled_draws - sampled_past).drop("year_id")

    # get the residuals and shift
    return y_star - shift


def _save_netcdf(data_array, path_out):
    """Saves the dataarray as a netcdf.

    :param xarray.DataArray data_array: data to save.
    :param Path path_out: save target.
    """
    path_out.PFN().parent.mkdir(parents=True, exist_ok=True)
    data_array.to_netcdf(str(path_out))


def _get_y_star(y_hat, epsilon_hat, years):
    """Returns draws of mortality or yld rates with estimated uncertainty.

    :param xarray.DataArray y_hat: expected value of mortality or yld rates.
    :param xarray.DataArray epsilon_hat: expected value of error.
    :param fbd_core.argparse.YearRange years: a container for the three years
        which define our forecast.
    :return xarray.DataArray: draws of mortality or yld rates with estimated
        uncertainty.
    """
    logger.info("Creating y_star by adding y_hat with epsilon_hat.")
    logger.debug("Make sure y_hat has the right number of draws.")
    draws = len(epsilon_hat.coords["draw"])
    y_hat_resampled = resample(y_hat, draws)
    # make sure the two dimensions are ordered in the same way
    y_hat_resampled = y_hat_resampled.transpose(
            *(list(epsilon_hat.coords.dims) + ["scenario"]))
    # correlate the time series draws with modeled estimates for uncertainty
    epsilon_correlated = ar1_utils.correlate_draws(epsilon_hat.copy(),
                                                   y_hat_resampled.copy(),
                                                   years)
    return y_hat_resampled + epsilon_correlated


def _draw_epsilons(epsilon_past, draws, smoothing, years, acause, decay,
                   gbd_round_id):
    """Draws forecasts for epsilons. For all-cause, this is done by running an
    attenuated drift model first to remove some of the time trend from the
    epsilons. Then for all causes (including all-cause) except NTD's, a Pooled
    random walk model is used to forecast the remaining residuals and generate
    expanding uncertainty.

    :param xarray.DataArray epsilon_past: past epsilons (error of predictions
        based on data)
    :param int draws: number of draws to grab.
    :param list[str] smoothing: what dimensions to smooth over in ARIMA
        modeling.
    :param fbd_core.argparse.YearRange years: a container for the three years
        which define our forecast.
    :param str acause: the cause to forecast epsilons for
    """
    logger.info((
        "Sampling epsilon_hat from ARIMA with cross sections by {}"
        "{}-{}").format(smoothing, years.forecast_start, years.forecast_end))
    # for all-cause, remove drift first, then run random walk on the remainder
    if acause == "_all":
        drift_component = remove_drift.get_decayed_drift_preds(epsilon_past,
                                                               years,
                                                               decay)
        remainder = epsilon_past - drift_component.sel(year_id=years.past_years)
        ds = xr.Dataset(dict(y=remainder.copy()))

    # if not all-cause, directly model epsilons with random walk
    else:
        ds = xr.Dataset(dict(y=epsilon_past.copy()))

    regdf = db.get_modeled_locations(gbd_round_id)[["location_id",
                                                    "region_id",
                                                    "super_region_id"]]

    regdf.set_index("location_id", inplace=True)
    ds.update(xr.Dataset(regdf))

    if acause == "_all":
        rw_model = rw.RandomWalk(ds, years.past_start,
                                 years.forecast_start, years.forecast_end,
                                 draws)
        rw_model.fit()
        rw_preds = rw_model.predict()
        epsilon_hat = drift_component + rw_preds
    else:
        # if not all-cause, use a pooled random walk whose location pooling dimension is based on cause levelß
        pooled_rw = PooledRandomWalk(ds, draws=draws, holdout=years.forecast_start,
                                     forecast=years.forecast_end, dims=smoothing)
        pooled_rw.fit()
        rw_preds = pooled_rw.predict()
        epsilon_hat = rw_preds
    return epsilon_hat



def _get_y_past(acause, years, measure, gbd_round_id, draw_level=False,
                draws=None, last_year_only=False,
                past_version=PAST_VERSION):
    """Gets expected value of cause specific mortality rates.

    Past data is saved in normal rate space. The past data is returned in log
    rate space.

    :param str acause: name of the target acause to aggregate to.
    :param fbd_core.argparse.YearRange years: a container for the three years
        which define our forecast.
    :param int year_start: start of the past.
    :param int year_end: end of the past, inclusive.
    :param bool draw_level: whether the past should be retrieved at the draw
        level (default is mean)
    :param bool last_year_only: whether to only get data from the last past
        year
    :return xarray.DataArray: The expected value of the cause specific
        mortality or yld rate.
    """
    logger.info("Getting past data from {} for years {}-{}.".format(
        FILEPATH, years.past_start, years.forecast_start-1))
    y_past = xr.open_dataarray(str(FILEPATH))

    # select the years of interest and convert to log space
    if last_year_only:
        past_years = years.forecast_start - 1
    else:
        past_years = years.past_years
    if draw_level:
        return resample(y_past.loc[dict(year_id=past_years)], draws)
    else:
        return y_past.loc[dict(year_id=past_years)].mean("draw")


def _get_y_hat(acause, input_version, agg_version, measure, period,
               draws, gbd_round_id):
    """Gets expected value of cause specific mortality or yld rates.

    For modeled causes, if the data is split by sex, then it is assumed that it
    is in log rate space. If the data is not split by sex, then it is assumed
    that it is in normal rate space.

    For aggregate causes, it is assumed that the data is not split by sex and
    is saved in log rate space.

    The resulting y_hat is in log rate space.

    :param str acause: name of the target acause to aggregate to.
    :param str mort_version: name of the mortality or yld version the aggregate
    is based on.
    :param str agg_version: name of the aggregate version.
    :return xarray.DataArray: The expected value of the cause specific
        mortality or yld rate.
    """
    # read GK modeled-level (most-detailed) causes from database
    engine = db.db_engine(NAME, database=DATABASE)
    session = sessionmaker(bind=engine)()

    gk_causes = get_strategy_set(session, FATAL_GK_STRATEGY_ID,
            CAUSE_HIERARCHY_ID)["acause"].values

    if acause in gk_causes:
        logger.info("{} is a modeled cause.".format(acause))
        y_hat = _get_modeled_y_hat(acause, input_version, measure,
                                   period, gbd_round_id, draws)

    else:
        logger.info("{} is an aggregated cause.".format(acause))
        y_hat = _get_aggregated_y_hat(acause, agg_version, measure,
                                      period, gbd_round_id)

    if isinstance(y_hat, xr.Dataset):
        if len(y_hat.data_vars) == 1:
            y_hat.rename({list(y_hat.data_vars.keys())[0]: "value"},
                         inplace=True)
            return y_hat["value"]
        logger.info(
            "Using __xarray_dataarray_variable__, "
            "but other data_vars are present! (probably just acause)")
        y_hat.rename({"__xarray_dataarray_variable__": "value"}, inplace=True)
    else:
        y_hat.name = "value"
    return y_hat


def _get_aggregated_y_hat(acause, version, measure, period, gbd_round_id):
    """Gets expected value of cause specific mortality rates.

    For aggregate causes, it is assumed that the data is not split by sex and
    is saved in log rate space.

    When the children are added to form the aggregated acause result, the
    summation happens in normal space. Therefore, we must exponentiate the
    children's rates, add them up, and log them to get an aggregated
    y_hat in log rate space.

    The resulting y_hat is in log rate space.

    :param str acause: name of the target acause to aggregate to.
    :param str version: name of the aggregation version.
    :return xarray.DataArray: The expected value of the cause specific
        mortality rate.
    """
    # connect to db and read in cause hierarchy
    engine = db.db_engine(NAME, database=DATABASE)
    session = sessionmaker(bind=engine)()
    all_causes = get_hierarchy(session, "cause", CAUSE_HIERARCHY_ID)[
            ["acause", "cause_id", "parent_id"]]
    # subset to just fatal causes
    cause_strategy_set = get_strategy_set(
                session, FATAL_GK_STRATEGY_ID, CAUSE_HIERARCHY_ID)
    cause_hierarchy = get_hierarchy(session, "cause", CAUSE_HIERARCHY_ID)
    cause_tree, node_map = subset_fatal.make_hierarchy_tree(cause_hierarchy, 294, "cause_id")
    fatal_subset = subset_fatal.include_up_hierarchy(cause_tree, node_map,
                                                     cause_strategy_set["cause_id"].values)
    fatal_causes = all_causes[all_causes.cause_id.isin(fatal_subset)]

    cause_id = fatal_causes[fatal_causes.acause == acause].cause_id.values[0]
    children = fatal_causes.query(
            "parent_id == {}".format(cause_id))["acause"].values
    logger.info("y_hat is a sum of children: {}".format(children))

    # Create a list of child acause files which are not external causes and
    # check to make sure all the ones we want to sum up are actually present.
    potential_child_files = [
        FBDPath("/{gri}/{p}/{m}/{v}/{c}_hat.nc".format(
                gri=gbd_round_id, p=period, m=measure, v=version, c=child),
                root_dir="scratch")
        for child in children if child not in ("_all", "_none")]
    child_files = [str(child_file)
                   for child_file in potential_child_files
                   if child_file.exists()]
    if len(potential_child_files) != len(child_files):
        logger.error("You are missing files, bud. {} vs {}".format(
            potential_child_files, child_files))
        raise Exception("Missing y_hat files!")
    logger.debug("Summing these files: {}".format(child_files))

    exp_y_hat_sum = None
    for child_file in child_files:
        logger.info("Adding {}".format(child_file))
        exp_y_hat = xr.ufuncs.exp(
            xr.open_dataarray(child_file, drop_variables=["measure", "cov"]))
        if exp_y_hat_sum is None:
            exp_y_hat_sum = exp_y_hat
        else:
            exp_y_hat_broadcasted = xr.broadcast(exp_y_hat_sum, exp_y_hat)
            exp_y_hat_broadcasted = [data.fillna(0.) for data in
                                     exp_y_hat_broadcasted]
            exp_y_hat_sum = sum(exp_y_hat_broadcasted)
    y_hat = xr.ufuncs.log(exp_y_hat_sum)
    y_hat.coords["acause"] = acause
    return y_hat


def _get_modeled_y_hat(acause, version, measure, period, gbd_round_id, draws):
    """Gets mortality data for a modeled acause.

    For modeled causes, if the data is split by sex, then it is assumed that it
    is in log rate space. If the data is not split by sex, then it is assumed
    that it is in normal rate space.

    :param str acause: acause for a modeled acause.
    :param str version: name of the mortality or yld version which modeled this
        acauaArray: the mortality or yld data for acause.
    """
    if period == "past":
        input_file = FILEPATH/ "{}.nc".format(acause)
        y_hat_exp = xr.open_dataset(str(input_file))["value"] + FLOOR
        y_hat_exp = resample(y_hat_exp, draws)
        y_hat = xr.ufuncs.log(y_hat_exp)
        y_hat.coords["acause"] = acause
    else:
        try:
            logger.info("No children. y_hat is from mort/yld file {}".format(
                FILEPATH))
            # Because the data is modeled and not split by sex, it is saved in
            # normal rate space. Log it.
            y_hat_exp = xr.open_dataarray(str(FILEPATH))
            y_hat_exp = resample(y_hat_exp, draws)
            y_hat = xr.ufuncs.log(y_hat_exp + FLOOR)
            # some of the yld files are missing acause, so add that info
            y_hat.coords["acause"] = acause

        except IOError:  # Modeled data is split by sex.
            input_files = [FILES for FILES in POTENTIAL_FILES
                           if FILES.exists()]
            logger.info("Input results are split by sex. Files are {}".format(
                input_files))

            if len(input_files) == 1:
                logger.info("This is a sex specific cause. Gotta give it a "
                            "real coordinate on sex.")
                if "female" in input_files[0].as_posix():
                    sex_id = 2
                else:
                    sex_id = 1
                dataarray_one_sex = xr.open_dataarray(
                    str(input_files[0]),
                    drop_variables=["measure", "cov"])
                dataarray_one_sex = resample(dataarray_one_sex, draws)
                new_vals = np.expand_dims(dataarray_one_sex.values, 0)
                new_dims = ["sex_id"] + list(dataarray_one_sex.dims)
                logger.info("New dimensions: {}".format(new_dims))
                new_coords = ([[sex_id]] + [
                    coord.values
                    for coord in list(dataarray_one_sex.coords.indexes.values())]
                )
                y_hat = xr.DataArray(new_vals, dims=new_dims,
                                     coords=new_coords
                                     ).to_dataset(name="value")
                y_hat.coords["acause"] = acause

            elif len(input_files) == 2:
                y_hat = xr.open_mfdataset(
                    [str(input_file) for input_file in input_files],
                    concat_dim="sex_id",
                    drop_variables=["measure", "cov"])
                y_hat = resample(y_hat[list(y_hat.data_vars.keys())[0]], draws)

            else:
                logger.error((
                    "{} has no modeled mortality/ylds for version {}. ruh-roh."
                ).format(acause, version))
                raise Exception("Modeled acause has no saved results.")
            # if data are split by sex, they are in log space. convert back to
            # regular space to add the floor
            y_hat = xr.ufuncs.log(xr.ufuncs.exp(y_hat) + FLOOR)
    return y_hat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_arg_years()
    parser.add_arg_draws()
    parser.add_arg_dryrun()
    parser.add_argument(
            "--acause", type=str, required=True,
            help="The acause to compute y_hat, epsilon_hat, and y_star for.")
    parser.add_argument(
            "--input-version", type=str, required=True,
            help="The mortality or yld version that the results are based on.")
    parser.add_argument(
            "--agg-version", type=str, required=True,
            help="The version the aggregations are saved under.")
    parser.add_argument(
            "--arima-version", type=str, required=True,
            help="The version the arima results are saved under.")
    parser.add_argument(
            "--measure", type=str, required=True, choices=["yld", "death"],
            help="yld or death.")
    parser.add_argument(
            "--smoothing", nargs="+",
            choices=["location_id", "region_id", "super_region_id", "sex_id",
                     "age_group_id"],
            help="The location level to smooth over")
    parser.add_argument(
            "--gbd-round-id", type=int, required=True,
            help="The gbd round id the estimates are based on,"
            "e.g., 3 for 2015")
    parser.add_argument(
            "--no-bias-correction", action="store_true",
            help="Don't change the mean of the values to be distributed exp(E[X]).")
    parser.add_argument(
            "--period", type=str, required=False,
            default="future", choices=["past", "future"],
            help="Whether to aggregate the past or the future")
    parser.add_argument(
            "--make-yhat", action="store_true",
            help="ETL yhat (aggregating if necessary).")
    parser.add_argument(
            "--make-ystar", action="store_true",
            help="Run the ARIMA model on latent trends and save ystar/epsilon")
    parser.add_argument(
            "--past-version", type=str, required=False,
            default="best",
            help="The version of past aggregates to use in arima.")
    parser.add_argument(
            "--intercept-shift", action="store_true",
            help="Intercept shift the draws to incorporate past uncertainty")
    parser.add_argument(
            "--no-arima", action="store_true",
            help="Create y-star without running an random walk on the residuals")
    parser.add_argument(
            "--decay-rate", type=float, required=False, default=0.1,
            help="Rate at which drift on all-cause epsilons decay in future")


    args = parser.parse_args()
    logger.info("Log level: {}".format(logger.level))

    if args.make_yhat:
        # No need to specify draws here.
        aggregate_yhats(
            args.acause, args.input_version, args.agg_version, args.measure,
            args.period, args.draws, args.gbd_round_id, dryrun=args.dryrun)

    if args.make_ystar:
        # Draws matter here because we are sampling from an ARIMA.
        arima_and_ystar(
            args.acause, args.agg_version, args.arima_version,
            args.smoothing, args.years, args.measure, args.intercept_shift,
            args.gbd_round_id, draws=args.draws, decay=args.decay_rate,
            dryrun=args.dryrun, no_correction=args.no_bias_correction,
            past_version=args.past_version, no_arima=args.no_arima)

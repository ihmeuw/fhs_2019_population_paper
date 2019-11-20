"""Module for squeezing results into an envelope. The envelope starts with
_all_star.nc in the hat_star_dot directory and is generated by squeezing
children causes into their parent envelopes.

By running `python squeeze.py --parent-cause <acause> --squeeze-version
<squeeze-version> --star-version <star-version>`, the parent acause will act as
the envelope to squeeze its children causes.

The things being squeezed are the *_star.nc files from the hat_star_dot
directory.

The number of draws of the *_star.nc files must match the number of draws of
the _all_star.nc file, and subsequently the resulting squeezed mortality must
have the same number of draws.
"""
import matplotlib
matplotlib.use("Agg")

import logging

import pandas as pd
import numpy as np
import seaborn as sns
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fbd_core import db, argparse
from fbd_core.file_interface import FBDPath
from fbd_research.lex import model as lex
from fbd_scenarios.subset_fatal import get_fatal_causes

logger = logging.getLogger("fbd_scenarios.squeeze")

def squeeze(parent_cause, star_version, squeeze_version, measure, gbd_round_id,
            dryrun=False):
    '''
    Main method for squeezing. Makes sure the sum of the parent_cause's
    children is equal to the parent cause's estimates, then saves the
    squeezed child cause results.

    Args:
        parent_cause (str): the acause whose estimates used as the envelope
            which its child causes are squeezed into.
        star_version (str): the version of mortality with ARIMAed residuals
            being squeezed.
        squeeze_version (str): the mortality or yld version the squeezed
            results will be saved as. The parent_cause envelope will
            be of this squeeze_version.
        measure (str): either death or yld.
        dryrun (bool): dryrun flag. If True, no results are saved.
    '''
    children_sum = None
    children_acauses = get_children_acauses(parent_cause, measure, gbd_round_id)
    logger.debug("Children acauses are: {}".format(children_acauses))
    for child_acause in children_acauses:
        child_data = get_y_stars([child_acause], star_version, measure,
                                 gbd_round_id)
        # For most causes with coord `acause`, drop `acause`
        if "acause" in child_data.coords:
            child_data = child_data.drop("acause").squeeze()
        # For some ntd causes, the dataset do not have coord `acause`
        else:
            child_data = child_data.squeeze()

        if children_sum is None:
            children_sum = child_data
        else:
            children_broadcast = xr.broadcast(children_sum, child_data)
            children_broadcast = [data.fillna(0.) for data in
                                     children_broadcast]
            children_sum = sum(children_broadcast)
            children_sum.load()
    parent_data = get_y_squeezed(parent_cause, squeeze_version, measure,
                                 gbd_round_id)
    parent_data.load()
    if parent_data.coords["draw"].shape != children_sum.coords["draw"].shape:
        raise ValueError(
            "The parent and child don't have the same number of draws.")
    ratio = parent_data / children_sum

    # loop through and multiply all children by ratio, saving results
    for child in children_acauses:
        logger.info("Squeezing {}".format(child))
        child_data = get_y_stars([child], star_version, measure, gbd_round_id)
        if "acause" in child_data.coords:
            child_data = child_data.drop("acause").squeeze()
        else:
            child_data = child_data.squeeze()

        squeezed_child = child_data * ratio

        if not dryrun:
            save_netcdf(squeezed_child, FILEPATH)


def save_netcdf(data_array, path_out):
    '''
    Saves the dataarray as a netcdf.

    :param xarray.DataArray data_array: data to save.
    :param Path path_out: save target.
    '''
    path_out.parent.mkdir(parents=True, exist_ok=True)
    data_array.to_netcdf(str(path_out))


def get_y_squeezed(acause, squeeze_version, measure, gbd_round_id):
    '''
    Gets squeezed y estimates (i.e. from the parent) for a given acause
    and version.

    Both loads and returns in regular rate space

    Args:
        acause (str): acause for the cause being used as the envelope.
        squeeze_version (str): version to take the envelope from.

    Returns:
        xarray.DataArray: squeezed estimates envelope for the given
        acause in regular space.
    '''
    squeezed_data = xr.open_dataarray(str(FILEPATH))
    return squeezed_data


def _drop_measure(data):
    '''
    Helper function to call as a preprocessing step when calling xarray's
    open_mfdataset function in order to make sure all the dimensions are
    compatible across dataarrays.

    Args:
        data (xarray.DataArray): the dataarray to preprocess
    Returns:
        xarray.DataArray: the original data with the measure dimension dropped
            if it originally existed.
    '''
    try:
        data = data.drop("measure")
    except ValueError:
        pass
    try:
        data.rename({"__xarray_dataarray_variable__": "value"}, inplace=True)
    except ValueError:
        pass
    return data


def get_y_stars(acause_list, star_version, measure, gbd_round_id, draws=True):
    '''Gets estimates of means and modeled residuals (*_star.nc files) for a
    given list of causes.

    y_stars are saved in log rate space, but this function returns them in
    regular rate space.

    Args:
        acause_list (list(str)): a list of acauses to get data for.
        star_version (str): the version whose data is being getted.
        draws (bool): if False, the mean over the draws dimension is used.

    Returns:
        xarray.DataArray: data in regular rate space which contains a dimension
            for acause.
    '''

    log_data = xr.open_mfdataset(
                   [str(FILEPATH)
                    for acause in acause_list],
                   concat_dim='acause',
                   preprocess=_drop_measure)
    if not draws:
        log_data = log_data.mean('draw')  # take mean to get rid of draws
    y_star = xr.ufuncs.exp(log_data)  # exponentiate into normal space
    return y_star.to_array()


def get_children_acauses(acause, measure, gbd_round_id):
    '''
    Gets the children acauses for a given acause. Does not include any that are
    in the CAUSES_TO_EXCLUDE list

    Args:
        acause (str): the acause of the cause to find children of.
        measure (str): either "death" or "yld"

    Return:
        list(str): the children acauses of the input acause.
    '''
    fatal_causes = get_fatal_causes(gbd_round_id)[["acause",
                                                   "cause_id",
                                                   "parent_id"]]
    cause_id = fatal_causes[fatal_causes.acause == acause].cause_id.values[0]
    all_children = fatal_causes.query(
            'parent_id == {}'.format(cause_id))['acause'].values
    children = [child for child in all_children
                if child not in ('_all', "_none")]
    return children


def _copy_all_star(star_version, squeeze_version, measure, gbd_round_id,
                   years, make_lex=False, past_version=None, dryrun=False):
    '''Copies _all_star.nc into _all.nc.

    The data is converted from log rate space into regular rate space before
    being saved.

    Args:
        star_version (str): the version of mortality with ARIMAed residuals
        squeeze_version (str): the version of mortality with squeezed results.
        measure (str): the measure being squeezed (death or yld)
        gbd_round_id (int): the gbd round id for the results being squeezed
        years (fbd_core.argparse.YearRange): the years over which the results
            were modeled
        make_lex (bool): whether or not to create and save a shock-subtracted
            version of life expectancy based on the _all_star.nc mortality
        past_version (str): past mortality (should also be shock-subtracted) to
            use to create the full lex series
    '''
    log_all = xr.open_dataarray(str(FILEPATH)).sel(year_id=years.forecast_years)
    all_data = xr.ufuncs.exp(log_all)
    if not dryrun:
        save_netcdf(all_data, FILEPATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
            "--parent-cause", type=str, required=True,
            help="The cause to squeeze the children for")
    parser.add_argument(
            "--star-version", type=str, required=True,
            help=("The version of mortality with ARIMAed residuals in the "
                  "hat_star_dot directory (the *_star.nc files).")
            )
    parser.add_argument(
            "--squeeze-version", type=str, required=True,
            help="The mortality version to save the squeezed mortality as.")
    parser.add_argument(
            "--measure", type=str, required=True, choices=["yld", "death"],
            help="yld or death.")
    parser.add_argument(
            "--gbd-round-id", type=int, required=True,
            help="The gbd round to base estimates off of,"
            "e.g., 3 for 2015")
    parser.add_argument(
            "--past-version", type=str, required=False,
            help="The version of past mortality to use in lex timeseries")

    parser.add_arg_dryrun()
    parser.add_arg_years()

    args = parser.parse_args()
    if args.parent_cause == '_all':
        _copy_all_star(args.star_version, args.squeeze_version, args.measure,
                       args.gbd_round_id, years=args.years,
                       make_lex=args.make_lex, past_version=args.past_version,
                       dryrun=args.dryrun)
    squeeze(args.parent_cause, args.star_version, args.squeeze_version,
            args.measure, args.gbd_round_id, dryrun=args.dryrun)

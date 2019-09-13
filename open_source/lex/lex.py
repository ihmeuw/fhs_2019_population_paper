"""
Calculates Life Expectancy.

Works on the fbd35-2 environment.  For 1000 draws, it's safe to assume it
needs 38 slots.

This script does two things. First is to do the lex:

   $ python -m fbd_research.lex.lex -v \
     --input_death 20170629_haq_front_run_squeezed \
     --lexcsv . --version fitting

The second would visualize from those CSVs:

   $ python -m fbd_research.lex.pmplot -v --version=fitting --outdir=.
"""
import datetime
import difflib
import gc
import logging
from argparse import RawTextHelpFormatter
from enum import Enum
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

import xarray as xr

from fbd_core.argparse import ArgumentParser
from fbd_core.db import get_countries
from fbd_core.file_interface import FBDPath
from fbd_research.lex import model
from fbd_research.lex.nlx import nlx_to_file


class Exit(Enum):
    success = 0  # This task succeeded.
    failure = 1  # This failure means all tasks fail.
    recoverable = 2  # This task failed, but others might succeed.


LOGGER = logging.getLogger("fbd_research.lex.lex")


SCENARIOS = {
    -1: "Worse",
    0: "Reference",
    1: "Better"
}
POSSIBLE_YOUNGEST = [2, 28, 1]  # Age groups that start at 0.
MODELS = [mxx for mxx in dir(model)
          if mxx.startswith("gbd")]


def convert_dimension_names_to_strings(ds_xr):
    """Unicode names not mixing well with non-unicode. Rename to regular str.
    On Python 2.7, str and unicode differ. On Python 3, str is unicode.
    This is an XArray conversion problem.
    """
    conversion = dict()
    for var in ds_xr.dims:
        if not isinstance(var, str):
            conversion[var] = str(var)
    if conversion:
        return ds_xr.rename(conversion)
    else:
        return ds_xr


def ordered_truncation(coordinate, limiting):
    """
    Ensure none of the entries in limiting are in the coordinate.
    Remove them from the coordinate without changing its ordering.

    :param coordinate (XArray.DataArray): XArray.Dataset which is a coord.
    :param limiting: Iterable of values
    :return: reduced list or DataArray
    """
    years_to_cut = set(coordinate.values) - set(limiting)
    if years_to_cut:
        keep_years = list(coordinate.values)
        for year_cut in years_to_cut:
            keep_years.remove(year_cut)
        LOGGER.debug("Removing {} from array".format(years_to_cut))
    else:
        keep_years = coordinate.values
        LOGGER.debug("Not removing from array")
    return keep_years


def life_expectancy_to_wide(life_exp, years_limit=None):
    """Converts xarray dataset into a dataframe wide on draws.

    Args:
        life_exp (pd.Dataframe): Wide dataframe of life expectancy
        years_limit (list): [past_start, forecast_end]

    Returns:
        pandas.DataFrame: dataframe wide on draws.
    """
    years_limit = years_limit or [1990, 2040]
    life_exp = convert_dimension_names_to_strings(life_exp)
    if "variable" in life_exp.dims and life_exp.dims["variable"] > 1:
        lex_ds = life_exp.sel(variable="value", drop=True)
    else:
        lex_ds = life_exp

    # Grab the subset of life expectancy we want to plot
    country_ids = get_countries().location_id.values
    youngest_age_group = [young_idx for young_idx in POSSIBLE_YOUNGEST
                          if young_idx in lex_ds.age_group_id.values]
    if not youngest_age_group or len(youngest_age_group) > 1:
        raise Exception("Could not find a suitable youngest age group in lex")
    LOGGER.debug("Youngest age group is {}".format(youngest_age_group))
    plot_years = list(range(years_limit[0], years_limit[1]+1))
    life_exp_subset = lex_ds.sel(age_group_id=youngest_age_group[0]) \
        .sel(year_id=ordered_truncation(lex_ds.year_id, plot_years)) \
        .sel(location_id=ordered_truncation(lex_ds.location_id, country_ids)) \
        .drop("age_group_id")
    if "scenario" not in life_exp_subset.dims:
        LOGGER.warning("No scenarios in life expectancy")
        life_exp_subset = life_exp_subset.expand_dims("scenario")
        life_exp_subset = life_exp_subset.assign_coords(scenario=[0])
        low = life_exp_subset.copy()
        low["scenario"] = [-1]
        high = life_exp_subset.copy()
        high["scenario"] = [1]
        life_exp_subset = xr.concat([low, life_exp_subset, high],
                                    dim="scenario")
    if "draw" not in life_exp_subset.dims:
        LOGGER.warning("No draws in life expectancy")
        life_exp_subset = life_exp_subset.expand_dims("draw")
        life_exp_subset["draw"] = [0]
    # life_exp_subset.name = "value"

    # Reindex the dataarray so the multiindex in the dataframe is in a decent
    # order.
    LOGGER.debug("dimensions of dataset {}".format(lex_ds.dims))
    LOGGER.debug("dataset summary {}".format(life_exp_subset))
    transpose_order = ["scenario", "year_id", "location_id", "sex_id", "draw"]
    # Because only dataset has a "variable".
    if "variable" in life_exp_subset.dims:
        LOGGER.debug("inserting variable into dimensions")
        transpose_order.insert(0, "variable")
    try:
        life_exp_subset_reindexed = life_exp_subset.transpose(*transpose_order)
    except ValueError as ve:
        # This is usually "axes don't match array"
        LOGGER.error("ValueError transposing life expectancy to make csv "
                     "transpose_order {}\nex {}\nerror: {}\n".
                     format(transpose_order, life_exp_subset.coords, ve))
        exit(37)
    LOGGER.debug("dimensions of transposed dataset {}".format(
        life_exp_subset_reindexed.dims))
    life_exp_df = life_exp_subset_reindexed.to_dataframe()
    LOGGER.debug("dataframe levels {}".format(life_exp_df.index.names))

    # Make the dataframe wide on draws because that's how a ton of our old data
    # is formatted.
    unstack_idx = life_exp_df.index.names.index("draw")
    life_exp_df_wide = life_exp_df.unstack(level=unstack_idx)
    LOGGER.debug("life_exp_df_wide cols {}".format(life_exp_df_wide.columns))
    life_exp_df_wide_data = life_exp_df_wide
    num_of_draws = len(life_exp_df_wide_data.columns)
    life_exp_df_wide_data.columns = [
            "ex_{}".format(i) for i in range(num_of_draws)]

    return life_exp_df_wide_data


def life_expectancy_dataframe_to_scenario_csvs(
        life_expectancy_dataframe, outpath):
    """
    Converts a dataframe containing all scenarios worth of life expectancy into
    three CSVs.

    Args:
        life_expectancy_dataframe (pd.Dataframe): Just the :math:`e_x` values.
        outpath (Path): Directory into which to store files.
        version (str): Unique dataset identifier

    Returns:
        List of files written.
    """
    wrote_path = list()
    found_a_scenario = False
    for scenario in SCENARIOS:
        try:
            scenario_life_expectancy = life_expectancy_dataframe.xs(
                    scenario, level="scenario")
            scenario_csv_path = outpath.with_name(
                "{}.csv".format(SCENARIOS[scenario]))
            wrote_path.append(scenario_csv_path)
            scenario_life_expectancy.to_csv(str(scenario_csv_path))
            LOGGER.info("Wrote {}".format(scenario_csv_path))
            found_a_scenario = True
        except KeyError:
            LOGGER.error("Can't find scenario {}".format(scenario))

    if not found_a_scenario:
        exit(Exit.failure)
    return wrote_path


def xarray_to_csvs(lex_xr, output_path):
    """
    Saves an XArray with scenarios as a set of CSVs.

    Args:
        lex_xr: Life expectancy as a data array
        output_path: Path to a directory to write in.

    Returns:
        list(Path): The paths that were written
    """
    pandas_wide = life_expectancy_to_wide(lex_xr)
    written = life_expectancy_dataframe_to_scenario_csvs(
        pandas_wide, output_path)
    return written


def lex_calculation(mortality_rate, function_name):
    try:
        modeling_function = getattr(model, function_name)
    except AttributeError:
        LOGGER.exception("Should have been able to load the model, but no.")
        exit(Exit.failure)
    return modeling_function(mortality_rate)


def read_calc_write(function_name, path):
    """
    Entry point for lex for outside calls. The path contains
    the GBD Round, past or future, and version. The function_name
    should be one of the models, all of which are model.gbd3_*
    or model.gbd4_* in model.py. It assumes the input file
    is called ``_all.nc`` in the ``path`` directory.

    This version pulls out the "mean" variable from any dataset,
    but it could keep draws.

    Args:
        function_name (str): Which function to call to calculate
                             life expectancy. If the function is called
                             model.single_gbd2015, then this is gbd2015.

        path (FBDPath): Path in which to save all files.
    """
    input_mx = path.set_stage("death").with_name("_all.nc")
    try:
        LOGGER.info("Reading {}".format(input_mx))
        try:
            mx = xr.open_dataarray(str(input_mx))
        except ValueError:
            ds = xr.open_dataset(str(input_mx))
            LOGGER.info("input data {}".format(ds))
            mx = ds["value"]  # NOTE the "value" variable preserves draws
        if "cause_id" in mx.coords:
            mx = mx.loc[dict(cause_id=int(mx.cause_id[0]))]
        # Most mx values (not all) are well under one, so, in log space,
        # they will be negative. Some zeros are a little negative for numbers
        # in normal space. Hence we check for mx < -0.01.
        if float(mx.min()) < -0.01:
            LOGGER.info("Converting mx from log metric")
            mx_future_xr = xr.ufuncs.exp(mx)
        else:
            LOGGER.info("Assuming mx in normal metric")
            mx_future_xr = mx
    except IOError as ioe:
        if not input_mx.exists():
            LOGGER.error("File {} doesn't exist. Exiting.".format(input_mx))
            exit(3)
        else:
            raise ioe

    mx_no_point, point_coords = model.without_point_coordinates(mx_future_xr)

    # Because we operate across age group id in all the calculations so
    # this will be much faster.
    reordered = list(mx_no_point.dims)
    reordered.remove("age_group_id")
    reordered.append("age_group_id")
    mx_no_point = mx_no_point.transpose(*reordered)

    chunk_size = 100  # number of draws in each chunk
    n_draws = len(mx_no_point["draw"].values)

    for i, start_indx in enumerate(range(0, n_draws, chunk_size)):
        end_indx = (start_indx + chunk_size
                    if start_indx + chunk_size <= n_draws else n_draws)
        mx_small = mx_no_point.sel(draw=mx_no_point["draw"].
                                   values[start_indx:end_indx])
        ds_small = lex_calculation(mx_small, function_name)
        if i == 0:
            ds = ds_small
        else:
            ds = xr.concat([ds, ds_small], dim="draw")
            del mx_small, ds_small
            gc.collect()

    del mx_no_point
    gc.collect()

    nc_file = path.set_stage("life_expectancy").with_name("lex.nc")
    nc_file.parent.mkdir(parents=True, exist_ok=True)

    xarray_to_csvs(ds.ex, nc_file.with_name("lex.csv"))

    # The CSV writer can't handle extra point coordinates.
    ds_point = ds.assign_coords(**point_coords)

    del ds
    gc.collect()

    ds_point.ex.to_netcdf(str(nc_file))

    dataset_file = path.set_stage("life_expectancy").with_name(
        "lifetable_ds.nc")
    ds_point.to_netcdf(str(dataset_file))
    LOGGER.info("wrote {}".format(dataset_file))

    return ds_point


def available_versions(search_dir, pattern="*/_all.nc"):
    """
    This prints version strings with dates and creators.

    Args:
        search_dir(libpath.Path): directory where versions are located.

    Returns:
        versions, ordered oldest to newest
    """
    LOGGER.debug("input dir: {}".format(search_dir))
    files = search_dir.glob(pattern)

    def version_from_glob(globf):
        return globf.parts[-2]

    no_scenario = dict()
    for lf in files:
        version = version_from_glob(lf)
        LOGGER.debug("version {} file lf {}".format(version, lf))
        no_scenario[version] = lf
    ordered = [(f.stat().st_mtime, vers, f)
               for (vers, f) in no_scenario.items()]
    ordered.sort()
    rows = list()
    for when, pvers, path in ordered:
        cols = list()
        cols.append(pvers)
        cols.append(path.owner())
        cols.append(datetime.datetime.fromtimestamp(when).strftime("%x %X"))
        rows.append(cols)
    max_cols = [0] * len(rows[0])
    for rsize in rows:
        for idx, csize in enumerate(rsize):
            max_cols[idx] = max(max_cols[idx], len(csize))
    print("Looking in {}".format(search_dir))
    for show_row in rows:
        print(("{:{vers}} {:{wh}} {:{ow}}").format(*show_row,
              vers=max_cols[0], wh=max_cols[1], ow=max_cols[2]))
    return [rvers for (rwhen, rvers, rpath) in ordered]


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument("--version", type=str, required=False,
                        help=("The version of input to use. "
                              "Output will have same version string"))
    parser.add_argument("--past", action="store_true", default=False)
    parser.add_argument("--testfile", type=bool, default=False,
                        help="Run on a small test file")
    parser.add_argument("--gbd-round-id", type=int, required=False, default=4)
    parser.add_argument("--model", type=str, choices=MODELS,
                        default=model.BEST_LEX_MODEL,
                        help="which model to run")
    parser.add_argument("--nlx", type=bool, default=False,
                        help="Write nLx to output directory")
    parser.add_argument(
            "--available", action="store_true", required=False, default=False,
            help="Show what versions exist")
    args, _ = parser.parse_known_args()

    if args.available:
        available_versions(args.mxdir)
        exit(0)

    try:
        if args.model in MODELS:
            model_name = args.model
        else:
            near_matches = difflib.get_close_matches(args.model, MODELS)
            if len(near_matches) is 1:
                model_name = near_matches[0]
            else:
                LOGGER.error(
                    "There is no model named {}. Please try one of {}.".format(
                        args.model, MODELS
                    ))
                exit(Exit.failure)
    except AttributeError:
        LOGGER.exception("Should have been able to load the model, but no.")
        exit(Exit.failure)

    if not args.version:
        print("Please give a version of mx to read.")
        exit(2)
    path = FBDPath("/{}/{}/death/{}/{}".format(
        args.gbd_round_id, "past" if args.past else "future",
        args.version, "_all.nc"))
    main_ds = read_calc_write(model_name, path)
    if args.nlx:
        nlx_to_file(main_ds, path)

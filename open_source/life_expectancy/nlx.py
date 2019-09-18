"""
Calculates person-years lived in the future, which is an output.
"""
import csv
import logging
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

import numpy as np
import xarray as xr

import fbd_core.demog
from fbd_core.argparse import ArgumentParser

LOGGER = logging.getLogger("fbd_research.lex.nlx")

ROOT_DIR = ""

def to_int(x):
    """int("1.0") doesn't work, so this does."""
    # The 0.1 is to deal with 3.9999 because int truncates.
    return int(round(float(x)) + 0.1)


def _population_blank_storage(filenames):
    """
    Make an empty xarray.Dataarray that can hold all data from future
    populations. This looks at files in a directory and figures out
    what size data array would hold everything.

    Args:
        filenames: An iterable of Path objects where each name is
                   pop_<year>_<location>.csv and rows are wide by
                   draw with pop_<draw> titles.

    Returns:
        (xarray.Dataarray) Containing nothing. Empty.
    """
    year_loc = [fn.name[:-4].split("_")[1:] for fn in filenames]
    years, locations = [{to_int(ys[idx]) for ys in year_loc} for idx in [0, 1]]
    with filenames[0].open() as infile:
        r = csv.reader(infile)
        headers = r.next()
        age_group_idx = headers.index("age_group_id")
        age_group_id = {to_int(rx[age_group_idx]) for rx in r}
    # The order of columns is chosen to agree with the CSV read order for speed.
    return xr.DataArray(
        np.empty((len(years), len(locations), 2, len(age_group_id)),
                 dtype=np.float),
        coords=dict(year_id=sorted(years), sex_id=[1, 2],
                    age_group_id=sorted(age_group_id),
                    location_id=sorted(locations)),
        dims=["year_id", "location_id", "sex_id", "age_group_id"]
    )


def _fill_population(filename, destination):
    """
    Fills destination with contents of filename for populations.
    Takes the mean of all draws in a file.

    Args:
        filename (pathlib.Path): A single year and location's draws for
                                 population.
        destination (xarray.Dataarray): Modified in place.

    Returns:
        None
    """
    with filename.open() as infile:
        r = csv.reader(infile)
        headers = r.next()
        pop_idx = [pidx for (pidx, pval) in enumerate(headers)
                   if pval.startswith("pop_")]
        h_idx = {hn: headers.index(hn) for hn
                 in ["age_group_id", "location_id", "sex_id", "year_id"]}
        for line in r:
            dims = {dn: to_int(line[h_idx[dn]])
                    for dn in h_idx}
            val = np.mean([float(line[vx]) for vx in pop_idx])
            destination.loc[dims] = val


def population_future_mean_csv():
    """
    Population estimates for the future are stored in 4700 CSV files.
    This reads those files and returns a DataArray with the mean population
    estimate over all draws for all locations, ages, sexes, and years.

    Returns:
        (xarray.Dataarray) Mean population over location, age, sex, and year.
    """
    pop_dir = Path("{}/fbd_pop/bayes_fertility/pieces").format(ROOT_DIR)
    all = pop_dir / "all.nc"
    if all.exists():
        LOGGER.info("Using {} for population".format(all))
        return xr.open_dataarray(str(all))

    LOGGER.info("Using directory {} for population".format(pop_dir))
    filenames = list(pop_dir.glob("pop*.csv"))
    xr_da = _population_blank_storage(filenames)
    for read_file in filenames:
        _fill_population(read_file, xr_da)
    xr_da.to_netcdf(str(all))

    return xr_da


def population_future_mean():
    # These are populations that start at 100,000 people.
    # Not the right ones.
    best = ("{}/life_expectancy/"
            "20170407_reg_sdi_sr_secular_wquant-lxgk.nc").format(ROOT_DIR)
    pop = xr.open_dataarray(best)


def nLx_from_real_populations(ds):
    nx = fbd_core.demog.nx_from_age_group_ids(ds.mx.age_group_id)
    mx = ds.mx.mean("draw").loc[dict(scenario=0)].drop("scenario")
    ax = ds.ax.mean("draw").loc[dict(scenario=0)].drop("scenario")
    nUx = fbd_core.demog.fm_unit_time_lived(mx, ax, nx)
    import pdb; pdb.set_trace()
    lx_n2 = population_future_mean_csv()

    # Convert youngest age groups from ids (2, 3, 4) to (28).
    # We want mid-year population. Taking midyear of age_group
    # 4 is close.
    not_youngest = lx_n2.age_group_id.values[2:]
    pop_slice = lx_n2.loc[dict(age_group_id=not_youngest)]
    not_youngest[0] = 28
    pop_slice.coords["age_group_id"] = not_youngest

    # Because it's mid-year, we need to adjust it to get approximate nLx.
    px_sqrt = xr.ufuncs.sqrt(fbd_core.demog.fm_survival(mx, ax, nx))
    return nUx * pop_slice / px_sqrt


def nlx_to_file(csv_path, ds):
    nLx = nLx_from_real_populations(ds)
    nLx_df = nLx.to_dataframe(name="nLx")
    nLx_file = csv_path.set_stage("life_expectancy").with_name("nLx.csv")
    nLx_df.to_csv(str(nLx_file))
    LOGGER.info("wrote {}".format(nLx_file))


def le_to_nlx(csv_path, version):
    ex = xr.open_dataset(str(Path(csv_path) / "{}_ds.nc".format(version)))
    nlx_to_file(csv_path, ex, version)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=str, required=False,
                        help=("The version of input to use. "
                              "Output will have same version string"))
    parser.add_argument("--outdir", type=Path,
                        help="Where to save lex CSV and XArray")
    main_args, _ = parser.parse_known_args()

    le_to_nlx(main_args.outdir, main_args.version)

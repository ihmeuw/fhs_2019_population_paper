"""
convert csv predictions to xarray file and make epsilon
"""
import pandas as pd
import xarray as xr
from pathlib import Path

from fbd_core.file_interface import FBDPath, save_xr
from fbd_core import argparse

def make_eps(mig_version, model_version, model_name, gbd_round_id, years):
    df = pd.read_csv(model_path)
    # add all-sex and all-age id columns
    df["sex_id"] = 3
    df["age_group_id"] = 22
    # select the columns we need
    df = df[["location_id", "year_id", "age_group_id", "sex_id", "predictions",
             "migration_rate"]]
    # set index columns
    index_cols = ["location_id", "year_id", "age_group_id", "sex_id"]

    dataset = df.set_index(index_cols).to_xarray()
    dataset["eps"] = dataset["migration_rate"] - dataset["predictions"]

    save_xr(dataset["eps"].sel(year_id=years.past_years), eps_path, metric = "rate", space = "identity")

    pred_path = mig_dir / "mig_hat.nc"
    save_xr(dataset["predictions"].sel(year_id=years.years), pred_path, metric = "rate", space = "identity")

    mig_path = mig_dir / "wpp_hat.nc"
    save_xr(dataset["migration_rate"].sel(year_id=years.years), mig_path, metric = "rate", space = "identity")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_arg_years(required=True)
    parser.add_argument(
        "--mig-version", type=str, required=True,
        help="The version the migration data are saved under.")
    parser.add_argument(
        "--model-version", type=str, required=True,
        help="The version the model results are saved under.")
    parser.add_argument(
        "--model-name", type=str, required=True,
        help="The name of the model.")
    parser.add_argument("--gbd-round-id", type=int, required=True,
        help="Which gbd round id to use")
    args = parser.parse_args()

    make_eps(args.mig_version, args.model_version, args.model_name,
        args.gbd_round_id, args.years)

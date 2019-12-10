"""
apply Random Walk on every-5-year migration data without draws
"""
import xarray as xr
from pathlib import Path

from fbd_core import argparse
from fbd_core.etl.transformation import bias_exp
from fbd_core.file_interface import save_xr, open_xr, FBDPath
from fbd_core.etl import expand_dimensions
from fbd_model.model import RandomWalk as rw
from fbd_scenarios import ar1_utils, remove_drift

def save_y_star(eps_version, arima_version, years, measure,
               draws, decay, gbd_round_id):
    """
    apply random walk and save the output
    """

    ds = open_xr(eps_path).data
    try:
        eps_preds = open_xr(f"{mig_dir}/eps_star.nc").data
    except Exception:
        eps_preds = arima_migration(ds, years, draws, decay)
        epsilon_hat_out = mig_dir / "eps_star.nc"
        save_xr(eps_preds, epsilon_hat_out, metric = "rate",
                space = "identity")

    # cap residuals between 10 and -10
    # the population forecasts to 2100 is decreasing to 0 with current
    # forecasts from migration for Syria, Latvia and Jamaica, the capping
    # method helps to make things more reasonable
    eps_past = eps_preds.sel(year_id=years.past_years)
    eps_preds = eps_preds.sel(year_id=years.forecast_years)
    eps_preds = eps_preds.clip(min=-10, max=10)
    eps_preds = xr.concat([eps_past, eps_preds], dim = "year_id")

    pred_path = mig_dir / "mig_hat.nc"
    preds = open_xr(pred_path).data
    preds = preds.sel(year_id = years.years)
    preds = expand_dimensions(preds, draw=range(0,draws))
    y_star = preds + eps_preds

    save_xr(y_star, ystar_out, metric = "rate", space = "identity")

def arima_migration(epsilon_past, years, draws, decay):
    """
    apply drift attenuation and fit random walk on the dataset
    """
    drift_component = remove_drift.get_decayed_drift_preds(epsilon_past,
                                                           years,
                                                           decay)
    remainder = epsilon_past - drift_component.sel(year_id=years.past_years)
    ds = xr.Dataset(dict(y=remainder.copy()))

    rw_model = rw.RandomWalk(ds, years.past_start, years.forecast_start,
                             years.forecast_end, draws)
    rw_model.fit()
    rw_preds = rw_model.predict()
    return drift_component + rw_preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_arg_years(required=True)
    parser.add_arg_draws(required=True)
    parser.add_argument(
        "--eps-version", type=str, required=True,
        help="The version the eps are saved under.")
    parser.add_argument(
        "--arima-version", type=str, required=True,
        help="The version the arima results are saved under.")
    parser.add_argument(
        "--measure", type=str, required=True, choices=["migration", "death"])
    parser.add_argument(
        "--decay-rate", type=float, required=False, default=0.1,
        help="Rate at which drift on all-cause epsilons decay in future")
    parser.add_argument("--gbd-round-id", type=int, required=True,
        help="Which gbd round id to use")
    args = parser.parse_args()

    save_y_star(args.eps_version, args.arima_version, args.years, args.measure,
                args.draws, args.decay_rate, args.gbd_round_id)

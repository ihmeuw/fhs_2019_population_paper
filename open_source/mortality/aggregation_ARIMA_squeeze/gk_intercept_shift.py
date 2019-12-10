"""
gk intercept shift at draw level for mortality
"""
import xarray as xr
import itertools as it

from fbd_core import argparse, db
from fbd_core.file_interface import FBDPath
from fbd_core.etl.transformation import resample

def intercept_shift_at_draw(preds, acause, past_version, gbd_round_id, years,
                            draws):
    """
    intercept shift at draw level for GK results used in mortality
    """
    input_past = FILEPATH / f"{acause}_hat.nc"
    past = xr.open_dataset(str(input_past)).sel(year_id = years.past_end)["value"]
    past = resample(past, draws)
    preds = resample(preds, draws)
    modeled_last = preds.sel(
        year_id = years.forecast_end, scenario = 0).drop("scenario")
    modeled_first = preds.sel(
        year_id = years.past_end, scenario = 0).drop("scenario")
    shifted = shift_draws(preds, modeled_last, modeled_first, past)

    return shifted

def shift_draws(preds, modeled_last, modeled_first, past):
    """
    Shift forecasts at draw level
    shifted = sorted_forecasts + sorted_past (last past year) - sorted_forecasts (last
    past year)

    Args:
        preds (xarray.DataArray):
            the draw-level predictions. Has at least dimensions `draw`, `year_id`.
        modeled_last (xarray.DataArray):
            the last-forecast-year model estimates at draw level.
        modeled_first (xarray.DataArray):
            the last-past-year model estimates at draw level.
        past (xarray.DataArray):
            the real past data of the last past year at draw level.
    Returns:
            xarray.DataArray: Shifted data
    """
    modeled_last_cp = modeled_last.copy()
    modeled_first_cp = modeled_first.copy()
    preds_cp = preds.copy()
    past_cp = past.copy()
    # save the original draw orders
    draw_n = modeled_last_cp.draw.values
    # get dimensions to loop through except for dimension draw
    temp = modeled_last_cp.drop("draw")
    coords = list(temp.coords.indexes.values())
    dims = list(temp.coords.indexes.keys())

    for coord in it.product(*coords):
        loc_dict = {dims[i]: coord[i] for i in range(len(coord))}
        sub_modeled_last = modeled_last_cp.sel(**loc_dict)
        sub_modeled_last_draw_index = sub_modeled_last.coords.dims.index("draw")
        rank_y = sub_modeled_last.argsort(sub_modeled_last_draw_index
                                         ).argsort(sub_modeled_last_draw_index).values
        sub_modeled_first = modeled_first_cp.sel(**loc_dict)
        sub_modeled_first.draw.values = rank_y
        sub_past = past_cp.sel(**loc_dict)
        sub_past_draw_index = sub_past.coords.dims.index("draw")
        rank_p = sub_past.argsort(sub_past_draw_index
                                 ).argsort(sub_past_draw_index).values
        sub_past.draw.values = rank_p
        shift = sub_past - sub_modeled_first

        sub_preds = preds_cp.sel(**loc_dict)
        sub_preds.draw.values = rank_y
        sub_preds = sub_preds + shift
        sub_preds.draw.values = draw_n
        preds_cp.loc[loc_dict] = sub_preds

    return preds_cp

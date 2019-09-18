"""
This is for capping maternal_hiv, maternal_indirect and drug causes in GK"
"""
from collections import namedtuple
import pdb

import numpy as np
import pandas as pd
import xarray as xr

from fbd_core import argparse, db
from fbd_core.file_interface import FBDPath, open_xr, save_xr

from fbd_cod import settings as ss

Quantiles = namedtuple("Quantiles", "lower, upper")

def cap_acause_sex(preds, acause, sex_id, years, past_version, quantiles=(0.01, 0.99)):
    """
    Caps a given acause-sex pair at input percentiles across
    past location-years.
    """
    # load in data
    past_path = FBDPath("/5/past/death/{}/{}.nc".format(past_version, acause))
    past_mean = open_xr(past_path).data["mean"].sel(sex_id=sex_id,
                                                    year_id=years.past_years)
    forecast = xr.ufuncs.exp(preds)
    last_year = forecast.sel(year_id=years.past_end)

    # find the caps
    caps = _find_limits(past_mean, last_year, quantiles)

    # reshape the bounds and cap the forecasts
    lower_bound = _reshape_bound(forecast, caps.lower)
    upper_bound = _reshape_bound(forecast, caps.upper)
    capped_forecast = forecast.clip(min=lower_bound,
                                    max=upper_bound).fillna(0)

    return xr.ufuncs.log(capped_forecast)

def _find_limits(mean, last_year, cap_quantiles):
    cap_quantiles = Quantiles(*sorted(cap_quantiles))

    quantile_values = mean.quantile(
        cap_quantiles, dim=["location_id", "year_id"])
    upper = quantile_values.sel(quantile=cap_quantiles.upper, drop=True)
    lower = quantile_values.sel(quantile=cap_quantiles.lower, drop=True)

    last_year_gt_upper = last_year.where(last_year > upper)
    last_year_lt_lower = last_year.where(last_year < lower)

    upper_cap_lims = last_year_gt_upper.fillna(upper).rename("upper")
    lower_cap_lims = last_year_lt_lower.fillna(lower).rename("lower")

    cap_lims = xr.merge([upper_cap_lims, lower_cap_lims])
    return cap_lims

def _reshape_bound(data, bound):
    """Broadcast and align the dims of `bound` so that they match `data`"""
    expanded_bound, _ = xr.broadcast(bound, data)
    return expanded_bound.transpose(*data.coords.dims)

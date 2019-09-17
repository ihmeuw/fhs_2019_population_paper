"""This module provides the function to help correlate high epsilon draws with
high modeled draws. It is called by aggregate_mortality_with_arima_error.py.
"""
import numpy as np
import xarray as xr
import itertools as it
from collections import namedtuple

from fbd_core.etl import resample


def correlate_draws(epsilon_preds, modeled_draws, years, first=False):
    """
    Correlates high epsilon draws with high modeled draws (currently
    correlates based on first forecasted year) to create a confidence
    interval that captures both past uncertainty and model uncertainty.
    Returns the correctly ordered epsilons for future years. Time series of
    epsilons are ordered by draw in the first predicted year and made to
    align in rank with the modeled rates for each predicted year.

    :param xarray.DataArray epsilon_preds: unordered draws of ARIMAed
        epsilons
    :param xarray.DataArray modeled_draws: modeled log mortality rate to
        correlate with
    :param fbd_core.argparse.ourgparse.YearRange: years being modeled

    :return xarray.DataArray: correlated epsilons for all years
    """
    # get both dataarrays lined up in terms of coords
    coords_order = ("location_id", "age_group_id", "sex_id", "draw",
                    "year_id")
    epsilon_preds = epsilon_preds.transpose(*coords_order)
    modeled_draws = modeled_draws.sel(scenario=0)

    if "acause" in modeled_draws.dims:
        modeled_draws = modeled_draws.sel(acause=modeled_draws["acause"].values[0]).drop(["acause"])
    elif "acause" in modeled_draws.coords:
        modeled_draws = modeled_draws.drop(["acause"])
    else:
        pass

    if "sex_id" in modeled_draws.dims:
        pass
    elif "sex_id" in modeled_draws.coords:
        modeled_draws = modeled_draws.expand_dims("sex_id")
    else:
        pass

    modeled_draws = modeled_draws.transpose(*coords_order)
    # we only need the first year of predictions for ordering
    first_year_draws = modeled_draws.sel(
            year_id=years.forecast_start).drop("year_id")

    # for each demographic, line up the appropriate draws
    AgeSexLocation = namedtuple("AgeSexLocation", "age sex location")
    dem_combos = it.starmap(AgeSexLocation,
                            it.product(epsilon_preds.age_group_id.values,
                                       epsilon_preds.sex_id.values,
                                       epsilon_preds.location_id.values))
    if first:
        order_year_index = years.forecast_start - years.past_start
    else:
        order_year_index = years.forecast_end - years.past_start
    for dem_combo in dem_combos:
        d = dict(location_id=dem_combo.location,
                 age_group_id=dem_combo.age,
                 sex_id=dem_combo.sex)
        modeled_sub = first_year_draws.loc[d]
        # argsort twice to get ranks
        modeled_order = modeled_sub.argsort(modeled_sub.dims.index('draw'))
        modeled_ranks = modeled_order.argsort(modeled_order.dims.index('draw'))
        # order by draw, then just take the order from the first predicted year
        eps_sub = epsilon_preds.loc[d]
        eps_order = eps_sub.argsort(eps_sub.dims.index('draw'))
        eps_order = [eps_order.values[i, order_year_index]
                     for i in range(eps_sub.shape[0])]
        # apply modeled ranks to the ordered epsilons to correctly place them
        eps_ordered = eps_sub.values[eps_order]
        eps_ordered = eps_ordered[modeled_ranks.values]
        epsilon_preds.loc[d] = eps_ordered
    return epsilon_preds

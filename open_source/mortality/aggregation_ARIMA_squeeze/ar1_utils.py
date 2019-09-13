"""This module provides functions to help create the all-cause ARIMA ensemble.
It is called by aggregate_mortality_with_arima_error.py on the epsilons (error)
between the modeled past and the past data to create a time series into the
future. The intent is to capture any time trends that remain after the
explanatory variables have been incorporated into the model.
"""
import numpy as np
import xarray as xr
import itertools as it
from collections import namedtuple

from fbd_model.model.PooledAR1 import PooledAR1
from fbd_core.etl import resample

NUM_HOLDOUT_YEARS = 3


def run_ar1(dat, locations, start, holdout, forecast, draws, model):
    '''
    Creates and runs an AR1 model for the specified years and parameters on
    an input data set

    :param xarray.Dataset dat: dataset to model and forecast, with the variable
        of interest labeled as "y"
    :param int p: order (number of time lags)
    :param int d: degree of differencing
    :param int q: order of moving average
    :param bool constant: whether or not to include a drift parameter
    :param list[int] locations: a list of locations to subset to (optional)
    :return xarray.DataArray: modeled future values based on the input data
    '''
    diff = model[0]
    constant = model[1]

    print('Fitting AR1, d={}, with{} constant'.format(
        diff, '' if constant else 'out'))
    # fit the AR1
    dd = dat.copy()
    if locations is not None:
        dd = dd.sel(location_id=locations)
    ar = PooledAR1(dd, diff=diff, drift=constant, y='y', start=start,
                   holdout=holdout, forecast=forecast, draws=draws,
                   dims=["location_id", "sex_id", "age_group_id"])
    ar.fit(parallelize=True)

    # make predictions
    parallelize = (draws <= 100)  # 1000 draws is too large for pickling
    pred = ar.predict(parallelize=parallelize)
    pred.coords['model'] = '{}{}'.format(diff, 'c' if constant else 'nc')

    return pred


def run_all(dat, models=((0,False), (1,False), (0,True), (1,True)),
            start=1990, holdout=2017, forecast=2040, locations=None,
            draws=100):
    """
    Runs AR1 models on the input dataset for all sets of input parameters
    specified in the 'models' input

    :param xarray.Dataset dat: dataset to model and forecast, with the variable
        of interest labeled as 'y'
    :param list[tuple(int, int, int, bool)] models: the sets of parameters to
        use in the AR1s

    :return xarray.DataArray: predictions for each specified model
    """
    eps_hat_list = []
    for model in models:
        eps_hat = run_ar1(dat, locations, start, holdout, forecast, draws,
                          model)
        eps_hat_list += [eps_hat]
    # return all the models as one big dataarray
    return xr.concat(eps_hat_list, dim='model')


def calculate_weights(eps_hat, eps_obs, years):
    """
    Weights the submodels based on how well they perform during the eval_years
    out of sample. Performance is evaluated based on how many locations the
    model produces the lowest RMSE out of sample, and weights sum to 1. All
    weights are age-specific.

    :param xarray.DataArray eps_hat: the predicted/modeled epsilons
    :param xarray.DataArray eps_obs: the 'true' epsilons
    """

    # find the years to evaluate oos performance during
    eval_years = list(range(years.forecast_start - NUM_HOLDOUT_YEARS,
                       years.forecast_start))

    # find the error
    err = (eps_hat.sel(year_id=eval_years) -
           eps_obs.sel(location_id=eps_hat.coords['location_id'])
          ).mean("draw")

    # find average error across year/sex
    avg_err = xr.ufuncs.square(err).mean(['year_id','sex_id'])

    # convert to ranks
    ranks = avg_err.argsort(avg_err.dims.index('model')
                           ).argsort(avg_err.dims.index('model'))

    # count number of first places
    first_place = (ranks == 0).sum('location_id').astype('float')

    # generate weights
    weights = first_place / first_place.sum('model')
    return weights



def combine_draws(pred_array, weights):
    """
    Given a list of predictions by model and the associated weights for each
    model, return a DataArray whose draws for each age are taken from each model
    proportional to its age-specific weight.

    :param xarray.DataArray pred_array: an array with prediction (and their
        draws) corresponding to each model
    :param xarray.DataArray weights: age-specific weights for each submodel

    :return xarray.DataArray: predictions with the number of draws weighted
        according to each submodel's weight
    """
    # make into a list for easier draw extraction
    list_of_preds = [pred_array.sel(model=model) for model in
                     pred_array.model.values]
    # get slices by age and combine them
    preds = xr.concat([_combine_age_specific_draws(list_of_preds,
                                                   age_group_id,
                                                   weights)
                       for age_group_id in weights.age_group_id.values],
                      dim="age_group_id")
    return preds


def combine_draws_simple(pred_array):
    """
    Combines predictions from two or four models by taking an equal number of
    draws from each model

    :param xarray.DataArray pred_array: an array with predictions (and
    their draws) corresponding to each model

    :return xarray.DataArray: predictions with an equal number of draws from
    each model"""
    models = pred_array.model.values
    num_draws = len(pred_array.draw.values)
    if len(models) == 2:
        preds = xr.concat([pred_array.sel(model=models[0],
                                          draw=np.arange(num_draws / 2)),
                           pred_array.sel(model=models[1],
                                          draw=np.arange(num_draws / 2,
                                                         num_draws))],
                          dim="draw").drop("model")
    else:
        preds = xr.concat([pred_array.sel(model=models[0],
                                          draw=np.arange(num_draws / 4)),
                           pred_array.sel(model=models[1],
                                          draw=np.arange(num_draws / 4,
                                                         num_draws / 2)),
                           pred_array.sel(model=models[2],
                                          draw=np.arange(num_draws / 2,
                                                         3 * num_draws / 4)),
                           pred_array.sel(model=models[3],
                                          draw=np.arange(3 * num_draws / 4,
                                                         num_draws))],
                          dim="draw").drop("model")
    return preds


def _combine_age_specific_draws(list_of_preds, age_group_id, weights):
    """
    Samples each model for the appropriate number of draws (by weight)
    for a given age_group_id and returns the concatenated array result

    :param list(xarray.DataArray) list_of_preds: list with each entry
        corresponding to the predictions from one submodel
    :param int age_group_id: the age group id for which the draws should be
        combined
    :param xarray.DataArray weights: age-specific weights for each submodel

    :return xarray.DataArray: draws weighted proportionally to their OOS
        performance for the specified age_group_id
    """
    age_specific_preds = [preds.sel(age_group_id=age_group_id).drop("model")
                          for preds in list_of_preds]
    age_specific_weights = [weight.sel(age_group_id=age_group_id).values
                            for weight in weights]
    assert len(age_specific_preds) == len(age_specific_weights)
    # get number of draws for each model - base on weights, then round down
    # and use the mean for the remainder
    total_draws = len(age_specific_preds[0].draw)
    num_draws_list = [int(weight * total_draws) for weight in
                      age_specific_weights]

    # sample from each model
    sampled = xr.concat([resample(age_specific_preds[i], num_draws_list[i]) for
                         i in range(len(age_specific_preds))], dim="draw")
    # fill in the missing draws
    num_missing = total_draws - sum(num_draws_list)
    missing_draws = (sampled.mean("draw") *
                     xr.DataArray([1 for x in range(num_missing)],
                                   coords=[np.arange(num_missing)],
                                   dims=["draw"]))
    sampled = xr.concat([sampled, missing_draws], dim="draw")

    # relabel coordinates
    assert len(sampled.draw) == total_draws
    sampled.coords["draw"] = np.arange(total_draws)

    return sampled


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

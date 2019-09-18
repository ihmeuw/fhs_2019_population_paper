"""Module with functions for making forecast scenarios."""
from __future__ import division
from collections import namedtuple
import logging

from frozendict import frozendict
import numpy as np
import xarray as xr

from fbd_core.etl.computation import weighted_mean, weighted_quantile
from fbd_core.etl.transformation import expand_dimensions
from fbd_core import YearRange

LOGGER = logging.getLogger(__name__)


def weighted_mean_with_extra_dim(data, stat_dims, weights, extra_dim=None):
    """Calculates the weighted-mean. If `extra_dim` is a dimension of `data`
    then loop through the `extra_dim` coordinates and calculate coord-specific
    ARCs using that coord's specific weights. Otherwise one ARC for all coords.

    Args:
        data (xarray.DataArray):
            Data to compute a weighted mean for.
        stat_dims (str, list[str]):
            dimension(s) of the dataarray to reduce over
        weights (xarray.DataArray):
            a 1-D dataarray the same length as the weighted dim, with dimension
            name equal to that of the weighted dim. Must be nonnegative.
        extra_dim (str):
            Extra dimension that exists in `weights` and `data`. It should not
            be in `stat_dims`.
    Returns:
        (xarray.DataArray):
            The mean over the given dimension. So it will contain all
            dimensions of the input that are not in ``stat_dims``.
    Raises:
        (ValueError):

            * If `weights` has more than 1 dimension while `extra_dim` is None.
            * If `extra_dim` is in `stat_dims`.
            * If `extra_dim` is not in a dimension of `weights`.
            * If `extra_dim` is not in a dimension of `data`.
            * If `extra_dim` does must have the same coordinates for `weights`
              and `data`.
    """
    LOGGER.debug("Entering the `weighted_mean_with_extra_dim` function")

    LOGGER.debug("extra_dim:{}".format(extra_dim))

    if len(weights.dims) > 1 and not extra_dim:
        dim_err_msg = ("`weights` cannot have more than 1 dim if `extra_dim` "
                       "is None")
        LOGGER.error(dim_err_msg)
        raise ValueError(dim_err_msg)
    elif extra_dim and extra_dim in stat_dims:
        dim_err_msg = "{} must cannot be in `stat_dims`".format(extra_dim)
        LOGGER.error(dim_err_msg)
        raise ValueError(dim_err_msg)
    elif extra_dim and extra_dim not in weights.dims:
        dim_err_msg = "{} must a dimension of `weights`".format(extra_dim)
        LOGGER.error(dim_err_msg)
        raise ValueError(dim_err_msg)
    elif extra_dim and extra_dim in weights.dims:
        if extra_dim and extra_dim not in data.dims:
            data = expand_dimensions(data, draw=weights["draw"].values)
        elif extra_dim and not data[extra_dim].equals(weights[extra_dim]):
            dim_err_msg = ("The {} dimension must have the same coordinates "
                           "for `weights` and `data`".format(extra_dim))
            LOGGER.error(dim_err_msg)
            raise ValueError(dim_err_msg)
        else:
            pass  # `data` already has "draw" dim with same coords as `weights`

        mean = []
        for coord in weights[extra_dim].values:
            LOGGER.debug("coord: {}".format(coord))
            coord_specific_data = data.loc[{extra_dim: coord}]
            coord_specific_weights = weights.loc[{extra_dim: coord}]
            coord_specific_mean = weighted_mean(coord_specific_data, stat_dims,
                                                coord_specific_weights)
            mean.append(coord_specific_mean)
        mean = xr.concat(mean, dim=extra_dim)
    else:
        mean = weighted_mean(data, stat_dims, weights)
    LOGGER.debug("Leaving the `weighted_mean_with_extra_dim` function")
    return mean


def weighted_quantile_with_extra_dim(data, quantiles, stat_dims, weights,
                                     extra_dim=None):
    """Calculates the weighted-mean. If `extra_dim` is a dimension of `data`
    then loop through the `extra_dim` coordinates and calculate coord-specific
    ARCs using that coord's specific weights. Otherwise one ARC for all coords.

    Args:
        data (xarray.DataArray):
            Data to compute a weighted mean for.
        quantiles (float or list of float):
            quantile(s) to evaluate.  Must be <= 1.
        stat_dims (str, list[str]):
            dimension(s) of the dataarray to reduce over
        weights (xarray.DataArray):
            a 1-D dataarray the same length as the weighted dim, with dimension
            name equal to that of the weighted dim. Must be nonnegative.
        extra_dim (str):
            Extra dimension that exists in `weights` and `data`. It should not
            be in `stat_dims`.
    Returns:
        (xarray.DataArray):
            The mean over the given dimension. So it will contain all
            dimensions of the input that are not in ``stat_dims``.
    Raises:
        (ValueError):
            * If `weights` has more than 1 dimension while `extra_dim` is None.
            * If `extra_dim` is in `stat_dims`.
            * If `extra_dim` is not in a dimension of `weights`.
            * If `extra_dim` is not in a dimension of `data`.
            * If `extra_dim` does must have the same coordinates for `weights`
              and `data`.
    """
    LOGGER.debug("Entering the `weighted_quantile_with_extra_dim` function")

    LOGGER.debug("extra_dim:{}".format(extra_dim))

    if len(weights.dims) > 1 and not extra_dim:
        dim_err_msg = ("`weights` cannot have more than 1 dim if `extra_dim` "
                       "is None")
        LOGGER.error(dim_err_msg)
        raise ValueError(dim_err_msg)
    elif extra_dim and extra_dim in stat_dims:
        dim_err_msg = "{} must cannot be in `stat_dims`".format(extra_dim)
        LOGGER.error(dim_err_msg)
        raise ValueError(dim_err_msg)
    elif extra_dim and extra_dim not in weights.dims:
        dim_err_msg = "{} must a dimension of `weights`".format(extra_dim)
        LOGGER.error(dim_err_msg)
        raise ValueError(dim_err_msg)
    elif extra_dim and extra_dim in weights.dims:
        if extra_dim and extra_dim not in data.dims:
            data = expand_dimensions(data, draw=weights["draw"].values)
        elif extra_dim and not data[extra_dim].equals(weights[extra_dim]):
            dim_err_msg = ("The {} dimension must have the same coordinates "
                           "for `weights` and `data`".format(extra_dim))
            LOGGER.error(dim_err_msg)
            raise ValueError(dim_err_msg)
        else:
            pass  # `data` already has "draw" dim with same coords as `weights`
        quantile_values = []
        for coord in weights[extra_dim].values:
            LOGGER.debug("coord: {}".format(coord))
            coord_specific_data = data.loc[{extra_dim: coord}]
            coord_specific_weights = weights.loc[{extra_dim: coord}]
            coord_specific_quantile_values = (
                weighted_quantile(da=coord_specific_data, q=quantiles,
                                  dim=stat_dims, ws=coord_specific_weights))
            quantile_values.append(coord_specific_quantile_values)
        quantile_values = xr.concat(quantile_values, dim=extra_dim)
    else:
        quantile_values = weighted_quantile(da=data, q=quantiles,
                                            dim=stat_dims, ws=weights)
    LOGGER.debug("Leaving the `weighted_quantile_with_extra_dim` function")
    return quantile_values


def truncate_dataarray(dataarray, quantile_dims, replace_with_mean=False,
                       mean_dims=None, weights=None, quantiles=None,
                       extra_dim=None):
    r"""Truncates the dataarray over the given dimensions, meaning that data
    outside the upper and lower quantiles, which are taken across the
    dimensions ``quantile_dims``, are replaced either with:
    1. the upper and lower quantiles themselves.
    2. or with the mean of the in-lier data, which is taken across the
       dimensions given by ``mean_dims``.

    **Note**: If weights are given, then weighted-quantiles and weighted-means
    are taken, otherwise the quantiles and means are unweighted.

    Args:
        dataarray (xarray.DataArray):
            dataarray that has at least the dimensions given by ``dims``, and
            if ``replace_with_mean`` is True, then also ``mean_dims``.
        replace_with_mean (bool, optional):
            If True, then replace values outside of the upper and lower
            quantiles and with the mean across the dimensions given by
            `mean_dims`, if False, then replace with the upper and lower bounds
            themselves.
        mean_dims (list[str], optional):
            dimensions to take mean within the bounds over
        quantile_dims (list[str]):
            dimensions to take quantiles over -- the quantiles are
            used to make the bounds.
        weights (xarray.DataArray, optional):
            Must have one dimension and can have up two dimensions.
        quantiles (tuple[float, float] | list[float, float], optional):
            The tuple of two floats representing the quantiles to take.
        extra_dim (str):
            Extra dimension that exists in `weights` and `data`. It should not
            be in `stat_dims`.
    Returns:
        (xarray.DataArray):
            Same shape as the original array, but with truncated values.
    Raises:
        (ValueError):
            If `replace_with_mean` is True, and `mean_dims` is not list of
            strings.
    """
    LOGGER.debug("Entering the `truncate_dataarray` function")

    LOGGER.debug("quantile_dims:{}".format(quantile_dims))
    LOGGER.debug("replace_with_mean:{}".format(replace_with_mean))
    LOGGER.debug("mean_dims:{}".format(mean_dims))
    LOGGER.debug("weights:{}".format(weights))
    LOGGER.debug("quantiles:{}".format(quantiles))
    LOGGER.debug("extra_dim:{}".format(extra_dim))

    if replace_with_mean and not mean_dims:
        mean_dims_err_msg = (
            "If `replace_with_mean` is True, then `mean_dims` "
            "must be a list of strings")
        LOGGER.error(mean_dims_err_msg)
        raise ValueError(mean_dims_err_msg)
    else:
        pass  # `mean_dims` doesn't can be None

    quantiles = (
        Quantiles(*sorted(quantiles))
        if quantiles else Quantiles(0.05, 0.95))

    if weights is not None:
        quantile_values = weighted_quantile_with_extra_dim(
            dataarray, quantiles, list(quantile_dims), weights, extra_dim)
    else:
        quantile_values = dataarray.quantile(
            quantiles, dim=list(quantile_dims))
    lower_da = quantile_values.sel(quantile=quantiles.lower)
    upper_da = quantile_values.sel(quantile=quantiles.upper)

    if replace_with_mean:
        good_indexes = (dataarray >= lower_da) & (dataarray <= upper_da)
        inside_da = dataarray.where(good_indexes)
        outside_da = dataarray.where(~good_indexes)

        if weights is not None:
            inside_mean_da = weighted_mean_with_extra_dim(
                inside_da, mean_dims, weights, extra_dim)
        else:
            inside_mean_da = inside_da.mean(mean_dims)

        truncated_da = (
            inside_da.combine_first(xr.ones_like(outside_da) * inside_mean_da))
    else:
        expanded_lower_da, _ = xr.broadcast(lower_da, dataarray)
        expanded_lower_da = expanded_lower_da.transpose(*dataarray.coords.dims)

        expanded_upper_da, _ = xr.broadcast(upper_da, dataarray)
        expanded_upper_da = expanded_upper_da.transpose(*dataarray.coords.dims)

        truncated_da = dataarray.clip(
            min=expanded_lower_da, max=expanded_upper_da)
    LOGGER.debug("Leaving the `truncate_dataarray` function")
    return truncated_da


def arc(past_data_da, years, weight_exp, stat_dims, statistic, quantiles=None,
        diff_over_mean=False, truncate=False, truncate_dims=None,
        truncate_quantiles=None, replace_with_mean=False, extra_dim=None):
    r"""Makes rate forecasts by forecasting the Annualized Rates-of-Change
    (ARC) using either weighted means or weighted quantiles .

    The steps for forecasting logged or logitted rates with ARCs are:

    (1) Annualized rate differentials (or annualized rates-of-change if data is
        in log or logit space) are calculated.

        .. Math::

            \vec{D_{p}} =
            [x_{1991} - x_{1990}, x_{1992} - x_{1991}, ... x_{2016} - x_{2015}]

        where :math:`x` are values from ``past_data_da`` for each year and
        :math:`\vec{D_p}` is the vector of differentials in the past.

    (2) Year weights are used to weight recent years more heavily. Year weights
        are made by taking the interval

        .. math::

            \vec{W} = [1, ..., n]^w

        where :math:`n` is the number of past years, :math:`\vec{w}` is the
        value given by ``weight_exp``, and :math:`\vec{W}` is the vector of
        year weights.

    (3) Weighted quantiles or the weighted mean of the annualized
        rates-of-change are taken over the dimensions.

        .. math::

            s = \text{weighted-statistic}(\vec{W}, \vec{D})

        where :math:`s` is the weighted quantile or weighted mean.

    (4) Future rates-of-change are simulated by taking the interval

        .. math::

            \vec{D_{f}} = [1, ..., m] * s

        where :math:`\vec{D_f}` is the vector of differentials in the future
        and :math:`m` is the number of future years to forecast and

    (5) Lastly, these future differentials are added to the rate of the last
        observed year.

        .. math::

            \vec{X_{f}} = \vec{D_{f}} + x_{2016} = [x_{2017}, ..., x_{2040}]

        where :math:`X_{f}` is the vector of forecasted rates.

    Args:
        past_data_da (xarray.DataArray):
            Past data with a year-id dimension. Must be in log or logit space
            in order for this function to actually calculate ARCs, otherwise
            it's just calculating weighted statistic of the first differences.
        years (YearRange):
            past and future year-ids
        weight_exp (float | int | xarray.DataArray):
            power to raise the increasing year weights -- must be nonnegative.
            It can be dataarray, but must have only one dimension, "draw", it
            must have the same coordinates on that dimension as
            ``past_data_da``.
        stat_dims (list[str]):
            list of dimensions to take quantiles over
        statistic (str):
            The statistic to use for calculating the ARC of past years. Can
            either be "mean" or "quantile".
        quantiles (object, optional):
            The quantile or quantiles to take on ``past_data``. Defaults to
            None, but must be a float, or a iterable of floats if
            statistic="quantile".
        diff_over_mean (bool, optional):
            If True, then take annual differences for means-of-draws, instead
            of draws. Defaults to False.
        truncate (bool, optional):
            If True, then truncates the dataarray over the given dimensions.
            Defaults to False.
        truncate_dims (list[str], optional):
            A list of strings representing the dimensions to truncate over.
        truncate_quantiles (object, optional):
            The iterable of two floats representing the quantiles to take.
        replace_with_mean (bool, optional):
            If True and `truncate` is True, then replace values outside of the
            upper and lower quantiles taken across "location_id" and "year_id"
            and with the mean across "year_id", if False, then replace with the
            upper and lower bounds themselves.
        extra_dim (str):
            Extra dimension that exists in `weights` and `data`. It should not
            be in `stat_dims`.
    Returns:
        (xarray.DataArray):
            Forecasts made using the ARC method.
    Raises:
        ValueError:
            If ``statistic`` is not equal to one of the strings "mean" or
            "quantile"
        ValueError:
            If ``weight_exp`` is a negative number
        ValueError:
            If `truncate` is True, then `truncate_quantiles` must be a list of
            floats.
    """
    LOGGER.debug("Entering the `arc` function")

    LOGGER.debug("years:{}".format(years))
    LOGGER.debug("weight_exp:{}".format(weight_exp))
    LOGGER.debug("statistic:{}".format(statistic))
    LOGGER.debug("stat_dims:{}".format(stat_dims))
    LOGGER.debug("quantiles:{}".format(quantiles))
    LOGGER.debug("diff_over_mean:{}".format(diff_over_mean))
    LOGGER.debug("truncate:{}".format(truncate))
    LOGGER.debug("replace_with_mean:{}".format(replace_with_mean))
    LOGGER.debug("truncate_quantiles:{}".format(truncate_quantiles))
    LOGGER.debug("extra_dim:{}".format(extra_dim))

    quantile_is_valid = (
        all([isinstance(quantile, float) for quantile in quantiles])
        if hasattr(quantiles, "__iter__") else isinstance(quantiles, float))

    if truncate and not truncate_dims:
        truncate_dims = ["location_id", "year_id"]

    if statistic not in ("mean", "quantile"):
        stat_arg_err_msg = (
            "`statistic` must be one of ('mean', 'quantile'), {} is not valid"
        ).format(statistic)
        LOGGER.error(stat_arg_err_msg)
        raise ValueError(stat_arg_err_msg)
    elif statistic == "quantile" and not quantile_is_valid:
        qnt_arg_err_msg = (
            "If `statistic='quantile'`, then `quantiles` must be of type float"
            " or a list of floats."
        ).format(statistic)
        LOGGER.error(qnt_arg_err_msg)
        raise ValueError(qnt_arg_err_msg)
    else:
        pass  # valid input given for `statistic` arg

    stat_dims = list(stat_dims)

    trunc_quantile_is_valid = (
        all([isinstance(trunc_quantile, float)
             for trunc_quantile in truncate_quantiles])
        if hasattr(truncate_quantiles, "__iter__") else False)

    if truncate and not trunc_quantile_is_valid:
        truncate_err_msg = (
            "If `truncate` is True, then "
            "`truncate_quantiles` must be a list of floats."
        )
        LOGGER.error(truncate_err_msg)
        raise ValueError(truncate_err_msg)
    elif truncate and trunc_quantile_is_valid:
        truncate_quantiles = Quantiles(*sorted(truncate_quantiles))
    else:
        pass  # `truncate_quantiles` can be None

    # Calculate the annual differentials.
    if diff_over_mean and "draw" in past_data_da.dims:
        annual_diff = past_data_da.mean("draw").sel(
            year_id=years.past_years).diff("year_id", n=1)
    else:
        annual_diff = past_data_da.sel(
            year_id=years.past_years).diff("year_id", n=1)

    if isinstance(weight_exp, xr.DataArray) and "draw" in weight_exp.dims:
        weight_exp = expand_dimensions(
            weight_exp, year_id=annual_diff["year_id"].values)
    elif isinstance(weight_exp, float) or isinstance(weight_exp, int):
        pass  # weight_exp can be a float or an integer
    else:
        weight_exp_err_msg = (
            "`weight_exp` must be a float, an int, or an xarray.DataArray "
            "with a 'draw' dimension")
        LOGGER.error(weight_exp_err_msg)
        raise ValueError(weight_exp_err_msg)

    year_weights = xr.DataArray(
        (np.arange(len(years.past_years) - 1) + 1),
        dims="year_id", coords={"year_id": years.past_years[1:]}) ** weight_exp

    # If annual-differences were taken over means (`annual_diff` doesn't have
    # a "draw" dimension), but `year_weights` does have a "draw" dimension,
    # then the draw dimension needs to be expanded for `annual_diff` such that
    # the mean is replicated for each draw.
    if "draw" in year_weights.dims and "draw" not in annual_diff.dims:
        annual_diff = expand_dimensions(annual_diff,
                                        draw=year_weights["draw"].values)
    else:
        pass  # `annual_diff` already has a draw dim, or `year_weights` doesn't

    if truncate:
        annual_diff = truncate_dataarray(
            annual_diff, truncate_dims, replace_with_mean=replace_with_mean,
            mean_dims=["year_id"], weights=year_weights,
            quantiles=truncate_quantiles, extra_dim=extra_dim)
    else:
        pass  # Annual differences are not truncated

    if (xr.DataArray(weight_exp) > 0).any():
        if statistic == "mean":
            arc_da = weighted_mean_with_extra_dim(
                annual_diff, stat_dims, year_weights, extra_dim)
        else:
            arc_da = weighted_quantile_with_extra_dim(
                annual_diff, quantiles, stat_dims, year_weights, extra_dim)
    elif (xr.DataArray(weight_exp) == 0).all():
        # If ``weight_exp`` is zero, then just take the unweighted mean or
        # quantile.
        if statistic == "mean":
            arc_da = annual_diff.mean(stat_dims)
        else:
            arc_da = annual_diff.quantile(q=quantiles, dim=stat_dims)
    else:
        err_msg = "weight_exp must be nonnegative."
        LOGGER.error(err_msg)
        raise ValueError(err_msg)

    # Find future change by multiplying an array that counts the future
    # years, by the quantiles, which is weighted if `weight_exp` > 0. We want
    # the multipliers to start at 1, for the first year of forecasts, and count
    # to one more than the number of years to forecast.
    forecast_year_multipliers = xr.DataArray(
        np.arange(len(years.forecast_years)) + 1,
        dims=["year_id"],
        coords={"year_id": years.forecast_years})
    future_change = arc_da * forecast_year_multipliers

    forecast_data_da = past_data_da.sel(year_id=years.past_end) + future_change
    LOGGER.debug("Leaving the `arc` function")
    return forecast_data_da


def arc_method(past_data_da, years=None, weight_exp=1.,
               reference_scenario="median", reverse_scenarios=False,
               quantiles=DEFAULT_SCENARIO_QUANTILES, diff_over_mean=False,
               reference_arc_dims=None, scenario_arc_dims=None, truncate=False,
               truncate_dims=None, truncate_quantiles=False,
               replace_with_mean=False, extra_dim=None):
    r"""Makes rate forecasts using the Annualized Rate-of-Change (ARC) method.

    Forecasts rates by taking a weighted quantile or weighted mean of
    annualized rates-of-change from past data, then walking that weighted
    quantile or weighted mean out into future years.

    A reference scenarios is made using the weighted median or mean of past
    annualized rate-of-change across all past years.

    Better and worse scenarios are made using weighted 15th and 85th quantiles
    of past annualized rates-of-change across all locations and all past years.

    The minimum and maximum are taken across the scenarios (values are
    granular, e.g. age/sex/location/year specific) and the minimum is taken as
    the better scenario and the maximum is taken as the worse scenario. If
    scenarios are reversed (``reverse_scenario = True``) then do the opposite.

    Args:
        past_data_da (xarray.DataArray):
            A dataarray of past data that must at least of the dimensions
            ``year_id`` and ``location_id``. The ``year_id`` dimension must
            have coordinates for all the years in ``years.past_years``.
        years (tuple[int] | list[int] | YearRange, optional):
            years to include in the past when calculating ARC.
        weight_exp (float | int | xarray.DataArray):
            power to raise the increasing year weights -- must be nonnegative.
            It can be dataarray, but must have only one dimension, "draw", it
            must have the same coordinates on that dimension as
            ``past_data_da``.
        reference_scenario (str, optional):
            If "median" then the reference scenarios is made using the
            weighted median of past annualized rate-of-change across all past
            years, "mean" then it is made using the weighted mean of past
            annualized rate-of-change across all past years. Defaults to
            "median".
        reverse_scenarios (bool, optional):
            If True, reverse the usual assumption that high=bad and low=good.
            For example, we set to True for vaccine coverage, because higher
            coverage is better. Defaults to False.
        quantiles (iterable[float, float], optional):
            The quantiles to use for better and worse scenarios. Defaults to
            ``0.15`` and ``0.85`` quantiles.
        diff_over_mean (bool, optional):
            If True, then take annual differences for means-of-draws, instead
            of draws. Defaults to False.
        reference_arc_dims (list[str], optional):
            To calculate the reference ARC, take weighted mean or median over
            these dimensions. Defaults to ["year_id"]
        scenario_arc_dims (list[str], optional):
            To calculate the scenario ARCs, take weighted quantiles over these
            dimensions.Defaults to ["location_id", "year_id"]
        truncate (bool, optional):
            If True, then truncates the dataarray over the given dimensions.
            Defaults to False.
        truncate_dims (list[str], optional):
            A list of strings representing the dimensions to truncate over.
        truncate_quantiles (object, optional):
            The tuple of two floats representing the quantiles to take.
        replace_with_mean (bool, optional):
            If True and `truncate` is True, then replace values outside of the
            upper and lower quantiles taken across "location_id" and "year_id"
            and with the mean across "year_id", if False, then replace with the
            upper and lower bounds themselves.
        extra_dim (str):
            Extra dimension that exists in `weights` and `data`. It should not
            be in `stat_dims`.
    Returns:
        (xarray.DataArray):
            Past and future data with reference, better, and worse scenarios.
            It will include all the dimensions and coordinates of the input
            dataarray and a ``scenario`` dimension with the coordinates 0 for
            reference, -1 for worse, and 1 for better. The ``year_id``
            dimension will have coordinates for all of the years from
            ``years.years``.
    Raises:
        (ValueError):
            If ``weight_exp`` is a negative number or if ``reference_scenario``
            is not "median" or "mean".
    """
    LOGGER.debug("Entering the `arc_method` function")

    LOGGER.debug("years:{}".format(years))
    LOGGER.debug("weight_exp:{}".format(weight_exp))
    LOGGER.debug("reference_scenario:{}".format(reference_scenario))
    LOGGER.debug("reverse_scenarios:{}".format(reverse_scenarios))
    LOGGER.debug("quantiles:{}".format(quantiles))
    LOGGER.debug("diff_over_mean:{}".format(diff_over_mean))
    LOGGER.debug("truncate:{}".format(truncate))
    LOGGER.debug("replace_with_mean:{}".format(replace_with_mean))
    LOGGER.debug("truncate_quantiles:{}".format(truncate_quantiles))
    LOGGER.debug("extra_dim:{}".format(extra_dim))

    years = YearRange(*years) if years else YearRange(*DEFAULT_YEAR_RANGE)

    past_data_da = past_data_da.sel(year_id=years.past_years)

    # Create baseline forecasts. Take weighted median or mean only across
    # years, so values will be as granular as the inputs (e.g. age/sex/location
    # specific)
    if reference_scenario == "median":
        reference_statistic = "quantile"
        reference_quantile = 0.5
    elif reference_scenario == "mean":
        reference_statistic = "mean"
        reference_quantile = None
    else:
        err_msg = "reference_scenario must be either 'median' or 'mean'"
        LOGGER.error(err_msg)
        raise ValueError(err_msg)

    if truncate and not truncate_dims:
        truncate_dims = ["location_id", "year_id"]

    truncate_quantiles = (
        Quantiles(*sorted(truncate_quantiles))
        if truncate_quantiles else Quantiles(0.025, 0.975))

    reference_arc_dims = reference_arc_dims or ["year_id"]
    reference_da = arc(
        past_data_da, years, weight_exp, reference_arc_dims,
        reference_statistic, reference_quantile, diff_over_mean=diff_over_mean,
        truncate=truncate, truncate_dims=truncate_dims,
        truncate_quantiles=truncate_quantiles,
        replace_with_mean=replace_with_mean, extra_dim=extra_dim)

    forecast_data_da = past_data_da.combine_first(reference_da)

    try:
        forecast_data_da = forecast_data_da.rename({"quantile": "scenario"})
    except ValueError:
        pass  # There is no "quantile" point coordinate.

    forecast_data_da["scenario"] = SCENARIOS["reference"]

    # Create better and worse scenario forecasts. Take weighted 85th and 15th
    # quantiles across year and location, so values will not be location
    # specific (e.g. just age/sex specific).
    scenario_arc_dims = scenario_arc_dims or ["location_id", "year_id"]
    scenarios_da = arc(
        past_data_da, years, weight_exp, scenario_arc_dims, "quantile",
        list(quantiles), diff_over_mean=diff_over_mean, truncate=False,
        replace_with_mean=replace_with_mean, extra_dim=extra_dim)
    scenarios_da = scenarios_da.rename({"quantile": "scenario"})
    scenarios_da.coords["scenario"] = [SCENARIOS["better"], SCENARIOS["worse"]]

    forecast_data_da = xr.concat([forecast_data_da, scenarios_da],
                                 dim="scenario")

    # Get the minimums and maximums across the scenario dimension, and set
    # worse scenarios to the worst (max if normal or min if reversed), and set
    # better scenarios to the best (min if normal or max if reversed).
    low_values = forecast_data_da.min("scenario")
    high_values = forecast_data_da.max("scenario")
    if reverse_scenarios:
        forecast_data_da.loc[{"scenario": SCENARIOS["worse"]}] = low_values
        forecast_data_da.loc[{"scenario": SCENARIOS["better"]}] = high_values
    else:
        forecast_data_da.loc[{"scenario": SCENARIOS["better"]}] = low_values
        forecast_data_da.loc[{"scenario": SCENARIOS["worse"]}] = high_values

    forecast_data_da = past_data_da.combine_first(forecast_data_da)

    forecast_data_da = forecast_data_da.loc[
        {"scenario": sorted(forecast_data_da["scenario"])}]
    LOGGER.debug("Leaving the `arc_method` function")
    return forecast_data_da



def approach_value_by_year(past_data, years, target_year, target_value,
                           method='linear'):
    r"""Method to handle use-cases where the desired level and the year by
    which it has to be achieved are known.
    For e.g., the Rockefeller project for min-risk diet scenarios, wanted to
    see the effect of eradicating diet related risks by 2030 on mortality. For
    this we need to reach 0 SEV for all diet related risks by 2030 and keep
    the level constant at 0 for further years. Here the target_year is 2030
    and target_value is 0.

    Args:
        past_data (xarray.DataArray):
            The past data with all past years.
        years (YearRange):
            past and future year-ids
        target_year (int):
            The year at which the target value will be reached.
        target_value (int):
            The target value that needs to be achieved during the target year.
        method (str):
            The extrapolation method to be used to calculate the values for
            intermediate years (years between years.past_end and target_year).
            The method currently supported is: `linear`.
    Returns:
        (xarray.DataArray):
            The forecasted results.
    """
    if method == 'linear':
        forecast = _linear_then_constant_arc(
            past_data, years, target_year, target_value)
    else:
        err_msg = ("Method {} not recognized. Please see the documentation for"
                   " the list of supported methods.").format(method)
        LOGGER.error(err_msg)
        raise ValueError(err_msg)

    return forecast


def _linear_then_constant_arc(past_data, years, target_year, target_value):
    r"""Makes rate forecasts by linearly extrapolating the point ARC from the
    last past year till the target year to reach the target value.

    The steps for extrapolating the point ARCs are:

    (1) Calculate the rate of change between the last year of the past
        data (eg.2017) and ``target_year`` (eg. 2030).

        .. Math::

            R =
             \frac{target\_value - past\_last\_year_value}
            {target\_year- past\_last\_year}

        where :math:`R` is the slope of the desired linear trend.

    (2) Calculate the rates of change between the  last year of the past and
        each future year by multiplying R with future year weights till
        ``target_year``.

        .. math::

            \vec{W} = [1, ..., m]

            \vec{F_r} = \vec{W} * R

        where :math:`m` is the number of years between the ``target_year`` and
        the last year of the past, and :math:`\vec{W}` forms the vector of
        year weights.
        :math:`\vec{F_r}` contains the linearly extrapolated ARCs for each
        future year till the ``target_year``.

    (3) Add the future rates :math: `\vec{F_r}` to last year of the past
        (eg. 2017) to get the forecasted results.

    (4) Extend the forecasted results till the ``forecast_end`` year by
        filling the ``target_value`` for all the remaining future years.

    Args:
        past_data (xarray.DataArray):
            The past data with all past years. The data is assumed to be in
            normal space.
        years (YearRange):
            past and future year-ids
        target_year (int):
            The year at which the target value will be reached.
        target_value (int):
            The value that needs to be achieved by the `target_year`.
    Returns:
        (xarray.DataArray):
            The forecasted results.
    """
    LOGGER.info("Entered `linear_then_constant_arc` function.")
    pre_target_years = np.arange(years.forecast_start, target_year+1)
    post_target_years = np.arange(target_year+1, years.forecast_end+1)

    past_last_year = past_data.sel(year_id=years.past_end)
    target_yr_arc = (
        target_value - past_last_year) / (target_year - years.past_end)

    forecast_year_multipliers = xr.DataArray(
        np.arange(len(pre_target_years)) + 1,
        dims=["year_id"],
        coords={"year_id": pre_target_years})

    LOGGER.info("Calculating future rates of change.")
    future_change = target_yr_arc * forecast_year_multipliers
    forecast_bfr_target_year = past_last_year + future_change

    forecast = expand_dimensions(
        forecast_bfr_target_year, fill_value=target_value,
        year_id=post_target_years)

    LOGGER.info("Leaving `linear_then_constant_arc`.")
    return forecast

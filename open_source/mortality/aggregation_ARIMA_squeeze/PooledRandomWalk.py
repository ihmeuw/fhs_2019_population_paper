"""
Random walk model for cause specific
"""
from contextlib import contextmanager
from functools import partial
import itertools as it
from multiprocessing import Pool
import os

import numpy as np
import xarray as xr


DEFAULT_WORKERS = 1


def fit_work(dims_, year_dict, y, l_):
    """
    Calculate sigma based on past time series variation
    """
    N = len(dims_)
    sub_dict = {dims_[i]: l_[i] for i in range(N)}
    sub_dict.update(year_dict)
    X = y.loc[sub_dict]
    sub_dict.pop("year_id")
    if len(X.coords.indexes) > 1:
        if "location_id" in list(X.coords.indexes.keys()):
            X.location_id.values = np.arange(len(X.location_id.values))
        dimsub = [k for k in X.coords.indexes.keys() if k != "year_id"]
        coordssub = [X.coords.indexes[k] for k in dimsub]
        diff_l = []
        n = len(dimsub)
        for l_ in it.product(*coordssub):
            temp_dict = {dimsub[i]: l_[i] for i in range(n)}
            X_ = X.loc[temp_dict]
            diff_l.append(X_.diff("year_id").values)
        sigma = np.std(diff_l)
    else:
        sigma = np.std(X.values)
    res = [sub_dict, sigma]
    return res


def predict_work(pardims, obs, yhat, sigma, forecast_years, holdout, l_):
    """
    Generate predictions for a specific age, sex, location
    """
    sub_dict = {pardims[i]: l_[i] for i in range(len(l_))}
    yhatsub = yhat.loc[sub_dict]
    yhatsub_forecast = yhatsub.sel(year_id = forecast_years)
    yhatsub_past = yhatsub.sel(year_id = obs.year_id.values)
    ysub = obs.loc[sub_dict]
    sigma_ = sigma.loc[sub_dict].values

    ran = np.random.normal(loc=0, scale=sigma_, size=yhatsub_forecast.shape)
    yhatsub_forecast.values = ysub.values[-1] + np.cumsum(ran, axis = 1)
    for i in yhatsub_past.draw.values:
        yhatsub_past.values[i] = ysub.values
    yhatsub = xr.concat([yhatsub_past, yhatsub_forecast], dim = "year_id")
    return (sub_dict, yhatsub)


@contextmanager
def terminating(thing):
    try:
        yield thing
    finally:
        thing.terminate()


class PooledRandomWalk:
    """
    Runs an random walk model on the data while pulling strength across multiple time
    series. Within the model structure argument define a key dims which defines
    which dimensions to use specific ar parameter values for as a list. An
    empty list, which is the default will use the same parameters for all time
    series. The time dimension should always be "year_id". If you specify
    "region_id" or "super_region_id" the dataset parameter must be an
    xr.Dataset with the modeled value as "y" and another key with "region_id"
    or "super_region_id" which has a location_id dimension and a value for
    each location.
    """

    def __init__(self, dataset, start=1990, holdout=2016, forecast=2040, y="y",
                 draws=100, dims=[], *args, **kwargs):
        """
        :param dataset: Dataset
            Dataset with a y key which you wish to forecast.
        :param constant: bool
            Whether to include a constant in the model
        :param start: int
            Year to start for the analysis
        :param holdout: int
            Year where data likelihood ends and forecast begins must be > start
        :param forecast: int
            Year to forecast to must be >= holdout
        :param y: str
            If using a Dataset the variable to apply analysis to.
        :param draws: int
            Number of draws to make for predictions.
        :param dims: list of str
            List of demographics to pool random walk estimates over.
        """
        if isinstance(dataset, xr.DataArray):
            dataset = xr.Dataset({"y": dataset})
        assert y in list(dataset.data_vars.keys()), "y was not found in the data"
        self.dataset = dataset.copy()
        self.y = y
        assert start < holdout <= forecast, "Year specifications not valid."
        self.start = start
        self.holdout = holdout
        self.forecast = forecast
        self.draws = draws
        self.dims = dims
        self.sigma = None
        self.yhat = None

    def fit(self, parallelize=False):
        """
        Fit the pooled random walk model.

        :param parallelize: bool
            Whether or not to run the 'fit' portion of the random walk in parallel
            by demographic specification.
        """
        # This is the most important copy in the whole world, never delete it.
        y = self.dataset[self.y].copy()
        dims_ = [i for i in self.dims]
        if "region_id" in dims_:
            stored_locs = self.dataset.location_id.values
            y.location_id.values = self.dataset.region_id.values
            dims_ = [i if i != "region_id" else "location_id" for i in dims_]
        if "super_region_id" in dims_:
            stored_locs = self.dataset.location_id.values
            y.location_id.values = self.dataset.super_region_id.values
            dims_ = [i if i != "super_region_id" else "location_id"
                     for i in dims_]
        coords_ = [y[x].values for x in dims_]
        datur = np.ones([len(x) for x in coords_]) if len(dims_) >= 1 else 1.
        self.sigma =\
            xr.DataArray(datur.copy(), dims=dims_, coords=coords_, name="sigma")

        year_dict = {"year_id": list(range(self.start, self.holdout))}
        coords_ = [np.unique(x) for x in coords_]

        if parallelize:
            num_workers = int(os.environ.get("SGE_HGR_fthread", DEFAULT_WORKERS))
            with terminating(Pool(num_workers)) as pool:
                results = pool.map(partial(fit_work, dims_, year_dict, y),
                                   it.product(*coords_))
        else:
            results = []
            for l_ in it.product(*coords_):
                res = fit_work(dims_, year_dict, y, l_)
                results.append(res)
        for result in results:
            sub_dict = result[0]
            self.sigma.loc[sub_dict] = result[1]

        if "region_id" in self.dims:
            self.sigma.location_id.values = stored_locs
            self.dataset.location_id.values = stored_locs
        if "super_region_id" in self.dims:
            self.sigma.location_id.values = stored_locs
            self.dataset.location_id.values = stored_locs

    def predict(self, parallelize=False):
        """
        Generate predictions based on model fits.
        :param parallelize: bool
            Whether or not to run the 'predict' portion of the random walk in parallel
            by demographic specification. Should not be used for models with
            1000 draws
        """
        N = self.draws
        d = xr.DataArray(np.ones(N), coords={"draw": list(range(N))},
                         dims=["draw"])
        years = list(range(self.start, self.forecast + 1))
        forecast_years = list(range(self.holdout, self.forecast + 1))
        year_dict = {"year_id": years}
        y = xr.DataArray(np.ones(len(years)),
                         coords=year_dict,
                         dims=["year_id"])
        self.yhat = (self.dataset[self.y] *
                     0. * d).mean("year_id") * y
        self.yhat = self.yhat.loc[year_dict]
        self.yhat.load()
        obs = self.dataset[self.y]
        if len(self.dims) > 0:
            pardims = list(self.sigma.coords.indexes.keys())
            parcoords = list(self.sigma.coords.indexes.values())
            if parallelize:
                num_workers = int(os.environ.get("SGE_HGR_fthread", DEFAULT_WORKERS))
                with terminating(Pool(num_workers)) as pool:
                    results = pool.map(partial(predict_work, pardims, obs,
                                               self.yhat, self.sigma,
                                               forecast_years, self.holdout),
                                       it.product(*parcoords))
            else:
                results = []
                for l_ in it.product(*parcoords):
                    res = predict_work(pardims, obs, self.yhat, self.sigma,
                                       forecast_years, self.holdout, l_)
                    results.append(res)
            for result in results:
                sub_dict = result[0]
                self.yhat.loc[sub_dict] = result[1]
        else:
            ran = np.random.normal(scale=self.sigma.values,
                                                size=self.yhat.shape)
            self.yhat.values = obs.values[-1] + np.cumsum(ran, axis = 1)

        return self.yhat

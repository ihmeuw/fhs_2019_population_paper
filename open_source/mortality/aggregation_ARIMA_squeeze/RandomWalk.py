import numpy as np
import xarray as xr
import itertools as it
from fbd_core.etl import expand_dimensions

class RandomWalk:
    """
    RandonWalk model which runs an independent random walk model for every time
    series in the dataset.
    """

    def __init__(self, dataset, start, holdout, forecast, draws, y="y", *args,
                 **kwargs):
        """
		:param dataset: Dataset
       		Dataset with a y key which you wish to forecast.
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
        """

        if isinstance(dataset, xr.DataArray):
            dataset = xr.Dataset({y: dataset})

        assert y in list(dataset.data_vars.keys()), "y was not found in the data"
        self.dataset = dataset.loc[{"year_id": np.arange(start, holdout)}]
        self.y = y
        assert start < holdout <= forecast, "Year specifications not valid."
        self.start = start
        self.holdout = holdout
        self.forecast = forecast
        self.draws = draws

    def fit(self):
        """
        Fit the std of the normal distribution used to generate the random walk
        """
        diff = self.dataset[self.y].diff("year_id")
        self.sigma = diff.std("year_id")

    def predict(self):
        """
        Generate predictions based on model fit.
        """
        locations = self.dataset[self.y].location_id.values
        ages = self.dataset[self.y].age_group_id.values
        sexs = self.dataset[self.y].sex_id.values
        location_data_list = []
        for location_id in locations:
            age_data_list = []
            for age_group_id in ages:
                sex_data_list = []
                for sex_id in sexs:
                    forecast = self.predict_single_ts(location_id,
                                                      age_group_id,
                                                      sex_id)
                    sex_data_list.append(forecast)
                age_data_list.append(xr.concat(sex_data_list, dim = "sex_id"))
            location_data_list.append(xr.concat(age_data_list, dim = "age_group_id"))
        all_preds = xr.concat(location_data_list, dim = "location_id")
        past = self.dataset.y
        try:
            past = past.drop(["acause", "scenario"])
        except ValueError:
            pass
        past = expand_dimensions(past, draw=range(0,self.draws))
        all_preds = xr.concat([past, all_preds], dim = "year_id")
        return all_preds



    def predict_single_ts(self, location_id, age_group_id, sex_id):
        """
        Generate predictions based on model fit on a single time series.
        """
        loc_dict = {"location_id":location_id, "age_group_id":age_group_id,
                    "sex_id":sex_id}
        past_ts = self.dataset.loc[loc_dict]
        assert ("year_id" in past_ts.dims) & (len(past_ts.dims) == 1), "past_ts is not one-dimensional"
        past_ts = past_ts[self.y].values
        sigma = self.sigma.loc[loc_dict].values
        ran = np.random.normal(loc=0, scale=sigma, size=(self.draws,
                                                         self.forecast-self.holdout
                                                        + 1))
        forecast = past_ts[-1] + np.cumsum(ran, axis = 1)
        years = np.arange(self.holdout, self.forecast + 1)
        forecast_xr = xr.DataArray(forecast,
                                   coords=[list(range(self.draws)),years],
                                   dims=["draw", "year_id"])
        forecast_xr.coords["location_id"] = location_id
        forecast_xr.coords["age_group_id"] = age_group_id
        forecast_xr.coords["sex_id"] = sex_id

        return forecast_xr

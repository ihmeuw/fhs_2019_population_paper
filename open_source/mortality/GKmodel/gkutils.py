import xarray as xr
import numpy as np
import pandas as pd
import gc
from scipy.special import gammaln
from scipy.optimize import brentq
import itertools as it
import scipy.stats as stats
from scipy.special import expit


def check_str(str_maybe):
    return isinstance(str_maybe, str)


def np2xr(arr, dem_coords, covs):
    """
    Convert a numpy array from TMB to an xarray for use in forecasting.


    """
    arr_rs = arr.copy()
    coords_copy = dem_coords.copy()
    coords_copy["cov"] = covs
    dims = ["location_id", "age_group_id", "year_id", "cov", "draw"]
    drop_dims = [x for x in range(3) if arr.shape[x] == 1]
    drop_names = [dims[x] for x in range(3) if arr.shape[x] == 1]
    dims = [dims[i] for i in range(len(dims)) if i not in drop_dims]
    for k in drop_names:
        coords_copy.pop(k)
    coords_copy["draw"] = np.arange(arr.shape[-1])
    if len(drop_dims) > 0:
        arr_rs = arr_rs.mean(tuple(drop_dims))
    return xr.DataArray(arr_rs, dims=dims, coords=coords_copy)


def check_constraints(constraints):
    """
    Make sure that all constraints are either -1, 0, or 1

    :param constraints: list like object of integers
        list of integers to use as GK TMB constraints
    """
    bad_constraints = np.setdiff1d(constraints, [1, -1, 0])
    assert len(bad_constraints) == 0, "Constraints must be either 1, -1, or 0."


def parse_covariates(cov_list):
    """
    Parse covariates from a list of tuples or strings

    :param cov_list: list of tuples or str
        list of either strings or tuples to parse out for modeling
    :return: tuple of lists
        tuple of list with the covariate names and list with constraints
    """
    if len(cov_list) == 0:
        return [], np.array([])
    if check_str(cov_list[0]):
        covariates = [l for l in cov_list]
        constraints = np.zeros(len(cov_list))
    else:
        covariates = [x for subl in cov_list for x in subl if check_str(x)]
        constraints =\
            [x for subl in cov_list for x in subl if not check_str(x)]
    check_constraints(constraints)
    return list(covariates), np.array(constraints)


def optimize_rate(x, m, v):
    """
    The function used in smoothing param that optimizes the  1/rate value of
    the gamma distribution in order to find an optimal smoothing parameter for
    the GK model.

    :param x: float > 0
        value of 1/rate to test
    :param m: float
        float representing the weighted average of an array
    :param v: float
        float representing the variance
    :return: float
        the return value of rate testing to find teh root of
    """
    opr = (np.sqrt((x/2.)-1.) * np.exp(gammaln((x/2.)-0.5) - gammaln(x/2.)) -
           (m / np.sqrt(m**2. + v)))
    return opr


def smoothing_param(array):
    """
    Function to define a GK smoothing parameter given its mean and sd. A
    reference for this code can be found here
    https://github.com/IQSS/YourCast/blob/955a88043fa97d71b922585b5bcd28cc5738d75c/R/compute.sigma.R#L194-L270

    :param array: array like of floats
        one dimensional array like object to compute smoothing parameter over
    :return: float
        appropriate smoothing value given an array
    """
    ma = np.ma.array(array, mask=np.isnan(array))
    weights = np.ones_like(array).cumsum(axis=2)**2.
    m = np.ma.average(ma, weights=weights)
    std = np.sqrt(np.ma.average((ma-m)**2., weights=weights))
    v = std**2.
    d = brentq(optimize_rate, 2.0000001, 1000, args=(m, v))
    e = (d - 2.) * (v + m**2.)
    return d / e


def demographic_dimensions(df):
    """
    Get the demographic dimensions of the data.

    :param df: DataFrame
        data frame to extract information from
    :return: tuple of integers
        integers representing the dimensions of each demographic variable.
    """
    demo_vars = ["location_id", "age_group_id", "year_id"]
    l, a, t = [len(df.index.get_level_values(d).unique()) for d in demo_vars]
    return l, a, t


def region_mapping(df):
    """
    Maps unique locations in a data frame to their respective regions and super
    regions

    :param df: DataFrame
        data frame from which to pull the unique locations and their respective
        regions and super regions.
    :return: tuple of arrays
        2 arrays corresponding to region ids and super region ids.
    """
    region_list =\
        df.groupby(level='location_id').aggregate({'region_id': np.mean,
                                                   'super_region_id': np.mean})
    region_factors = pd.factorize(region_list['region_id'])
    regions = region_factors[0]
    super_regions = pd.factorize(region_list['super_region_id'])[0]
    return regions, super_regions


def omega_priors(response_array, holdout_index, level, amp):
    """
    Get the vectors for omega priors to be calculated for a response variable
    as per the GK model process.

    :param response_array: array
        3D response variable indexed on location, age, time
    :param holdout_index: int
        the index by which the hold outs are started
    :param level: str
        level name to apply to the end of the dict key
    :param amp: dict
        how much to scale the omega priors by
    :return: dict
        dictionary of vectors to get the optimal smoothing from
    """
    l, a, t = response_array.shape
    response_array = np.copy(response_array)
    response_array[:, :, holdout_index:] = np.nan
    response_array_adj =\
        response_array - np.nanmean(response_array, axis=2).reshape((l, a, 1))
    response_array_adj = response_array_adj[:, :, :holdout_index]
    omega = dict()

    omega["omega_loc_" + level] =\
        (np.sqrt(np.abs(np.diff(response_array_adj, axis=0))) *
         amp["omega_location"])
    omega["omega_age_" + level] =\
        np.sqrt(np.abs(np.diff(response_array_adj, axis=1))) * amp["omega_age"]
    omega["omega_loc_time_" + level] = \
        np.sqrt(
            np.abs(
                np.diff(
                    np.diff(
                        response_array_adj, axis=2), axis=0))) *\
        amp["omega_location_time"]
    omega["omega_age_time_" + level] =\
        np.sqrt(
            np.abs(
                np.diff(
                    np.diff(
                        response_array_adj, axis=2), axis=1))) * \
        amp["omega_age_time"]

    # return 0. if only one or two ages for smoothing params that act over age
    sm = {k: smoothing_param(v) if (a > 2 or 'age' not in k)
          else 0. for k, v in omega.items()}
    return sm


def name_to_shape(cov_name, dim_sizes, super_region_size=0, region_size=0):
    """
    Converts a covariate name to a proper shape.

    :param cov_name: str
        a string representing a covariate type
    :param dim_sizes: list like
        list like of len of each dimension
    :return: tuple
        tuple of proper dimensions
    """
    l_, a_, _, k = dim_sizes
    t = 1
    if "age" in cov_name:
        a = a_
    else:
        a = 1
    if "super_region" in cov_name:
        l = super_region_size
    elif "region" in cov_name:
        l = region_size
    elif "location" in cov_name:
        l = l_
    else:
        l = 1
    return l, a, t, k


def transform_parameter_array(arr, constraints):
    """
    Transforms an array based on the constraints that are passed in

    :param arr: array
        5 dimensional(LxAxTxKxD) array to transform
    :param constraints: list-like
        vector of transforms to apply
    :return: array
        transformed 5-D array
    """
    arr_trans = np.copy(arr)
    for i in range(len(constraints)):
        if constraints[i] == -1:
            arr_trans[:, :, :, i, :] = -1 * np.exp(arr_trans[:, :, :, i, :])
        if constraints[i] == 1:
            arr_trans[:, :, :, i, :] = np.exp(arr_trans[:, :, :, i, :])
    return arr_trans


def region_transform(arr, regions):
    """
    Given an array of values  and regions transforms the array from
    (RxAxTxKxD) to (LxAxTxKxD)

    :param arr: 5D array
        5d array with regions to transform
    :param regions: list-like
        a region index for every location
    :return: array
        5d array (LxAxTxKxD)
    """
    _, a, t, k, d = arr.shape
    l = len(regions)
    arr_trans = np.zeros((l, a, t, k, d))
    for i in range(len(regions)):
        arr_trans[i, :, :, :, :] = arr[regions[i], :, :, :, :]
    return arr_trans


def omega_translate(omega_amp):
    """
    Take what is given for omega scale and transform into an appropriate dict

    :param omega_amp: int, float or dict
        original input for omega_amp in GKModel class
    :return: dict
        dictionary of corresponding values
    """
    omega_dict_keys = ["omega_age", "omega_location", "omega_age_time",
                       "omega_location_time"]
    if not isinstance(omega_amp, dict):
        omega_dict = {o: omega_amp for o in omega_dict_keys}
    else:
        omega_dict = dict()
        for o in omega_dict_keys:
            if o not in list(omega_amp.keys()):
                omega_dict[o] = 1.
            else:
                omega_dict[o] = omega_amp[o]
    return omega_dict


def modcolarray(xarray):
    """
    Collapse an array to its modelabe dimensions collapsing on the draw
    dimnesion and only using the zero scenario dimension if they exit.
    """
    save_locs = ["location_id", "age_group_id", "year_id", "cov"]
    if "scenario" in xarray.dims:
        xarray = xarray.sel(scenario = 0)
    for k in xarray.dims:
        if k not in save_locs:
            xarray = xarray.mean(k)
    reorder = [k for k in save_locs if k in xarray.dims]
    xarray = xarray.transpose(*reorder)
    return xarray


def xrmerge(das, collapse=False):
    """
    """
    final_das = []
    col_vars = [k for k in das.data_vars.keys() if k not in
                ["region_id", "super_region_id"]]

    for e in col_vars:
        ex_da = das[e].copy()
        das_to_expand_by = list(das.data_vars.keys())
        das_to_expand_by.remove(e)
        for oda_key in das_to_expand_by:
            ex_da, _ = xr.broadcast(ex_da, das[oda_key].copy())
        ex_da = ex_da * xr.DataArray([1], coords=[[ex_da.name]], dims=["cov"])
        final_das.append(ex_da)

    del ex_da
    gc.collect()

    final_array = xr.concat(final_das, dim="cov")
    del final_das
    gc.collect()

    if collapse:
        final_array = modcolarray(final_array)

    return final_array


def recursively_concat(big_list, concat_dims, concat_coords):
    """Concats multiple coords.

    :param list[xarray] big_list: the biggets list of xarray things.
    :param list[str] concat_coords: list of coords to concat over.
    :return xarray: concated thing.
    """
    concat_coords = [x for x in concat_coords]
    concat_dims = [x for x in concat_dims]
    if len(concat_coords) == 1:
        return xr.concat(big_list, concat_dims[0])
    else:
        smaller_list = []
        concat_dim = concat_dims.pop()

        list_of_coord_vals = it.product(*concat_coords)
        for coord_vals in list_of_coord_vals:
            really_small_list = []
            for da in big_list:
                for i, k in enumerate(concat_dims):
                    if da.coords[k] == coord_vals[i]:
                        really_small_list.append(da)
            concatted_da = xr.concat(really_small_list, dim=concat_dim)
            smaller_list.append(concatted_da)
        return recursively_concat(smaller_list, concat_dims, concat_coords)


class ParameterEffects:
    """
    Creates the initial random and fixed effects for PYMB model
    """
    def __init__(self, fixed_effects, random_effects, constant):
        """
        Takes a dictionary of random and fixed effects and makes them readable
        for a GKModel object

        :param fixed_effects: dict
            dictionary of list of tuples or str to parse out for modeling
        :param random_effects: dict
            dictionary of list of tuples or str to parse out for modeling

        """
        ty = ["age", "location_age", "location", "global", "region",
              "super_region", "region_age", "super_region_age"]
        self.fixed_names = ["{p}_{t}".format(p="beta", t=t) for t in ty]
        self.random_names = ["{p}_{t}".format(p="gamma", t=t) for t in ty]
        missing_fixed = np.setdiff1d(list(fixed_effects.keys()), self.fixed_names)
        missing_random = np.setdiff1d(list(random_effects.keys()), self.random_names)
        self.init = dict()
        self.data = dict()
        fixerr = "The fixed effects have improper names:" + str(missing_fixed)
        ranerr =\
            "The random effects have improper names:" + str(missing_random)
        assert len(missing_fixed) == 0, fixerr
        assert len(missing_random) == 0, ranerr
        self.fixed_effects = {k: (fixed_effects[k] if k in fixed_effects.keys()
                                  else []) for k in self.fixed_names}
        self.random_effects = {k: (random_effects[k] if k in
                                   list(random_effects.keys()) else [])
                               for k in self.random_names}
        self.constant = constant

    def extract_param_data(self, ds, final_array, final_array_col):
        """
        Parse out the covariate information from the appropriate dictionaries.

        :param df: data set
            data set to extract parameter information
        """
        if "region_id" in list(ds.data_vars.keys()):
            region_size = len(np.unique(ds.region_id.values))
        else:
            region_size = 0
        if "super_region_id" in list(ds.data_vars.keys()):
            super_region_size = len(np.unique(ds.super_region_id.values))
        else:
            super_region_size = 0
        for cov in self.fixed_names:
            cov_type = "_".join(cov.split("_")[1:])
            beta_const_name = "beta_{}_constraint".format(cov_type)
            beta_raw_name = "beta_{}_raw".format(cov_type)
            beta_mean_name = "beta_{}_mean".format(cov_type)
            X = "X_{}".format(cov_type)
            X_draw = "X_{}_draw".format(cov_type)
            self.data[cov], self.data[beta_const_name] =\
                parse_covariates(self.fixed_effects[cov])
            self.data[X] = final_array_col.sel(cov=self.data[cov]).values
            self.data[X_draw] = final_array.sel(cov=self.data[cov])
            param_shape = name_to_shape(cov_type, self.data[X].shape,
                                        super_region_size, region_size)
            self.init[beta_raw_name] = np.zeros(param_shape)
            self.init[beta_mean_name] = np.ones(param_shape)
        for re in self.random_names:
            re_type = "_".join(re.split("_")[1:])
            self.data[re], _ = parse_covariates(self.random_effects[re])
            gamma_name = "gamma_{}".format(re_type)
            Z = "Z_{}".format(re_type)
            Z_draw = "Z_{}_draw".format(re_type)
            tau_name = "log_tau_{}".format(re_type)
            self.data[Z] = final_array_col.sel(cov=self.data[re]).values
            self.data[Z_draw] = final_array.sel(cov=self.data[re])
            param_shape = name_to_shape(re_type, self.data[Z].shape,
                                        super_region_size, region_size)
            self.init[gamma_name] = np.zeros(param_shape)
            self.init[tau_name] = np.zeros((1, 1, 1, self.data[Z].shape[-1]))
        self.data["constant"] =\
            final_array_col.sel(cov=self.constant).values
        self.data["constant_draw"] = final_array.sel(cov=self.constant)
        self.data["constant_mult"] =\
            np.ones((1, 1, 1, self.data["constant"].shape[3]))

def ar1nll(x0, x):
    """
    Evaluate the likelihood of ar1 parameters in x0 for the time series x

    :param x0: 1D array of size 2 (if no drift) or 3 (if drift)
        the transformed AR1 coefficients
    :param x: 1D array
        time series of data
    :return: float
        nll of paramters for observed data
    """
    rho = expit(x0[0])
    sigma = np.exp(x0[1])
    try:
        const = x0[2]
    except IndexError:
        const = 0
    xhat = np.zeros(len(x))
    for i in range(1, len(x)):
        xhat[i] = rho*x[i-1] + const
    dist = stats.norm(x[2:], sigma)
    return np.sum(-1 * dist.logpdf(xhat[2:]))


def ar1arraynll(x0, X):
    """
    Evaluate the likelihood of ar1 parameters for multiple time series.

    :param x0: 1D array of size 2
        the transformed AR1 coefficients
    :param X: 2D array
        M 1D time series data of length N that form an (M,N) matrix
    :return: float
        nll of paramters for the M time series
    """
    if len(X.coords.indexes) > 1:
        if "location_id" in list(X.coords.indexes.keys()):
            X.location_id.values = np.arange(len(X.location_id.values))
        dimsub = [k for k in X.coords.indexes.keys() if k != "year_id"]
        coordssub = [X.coords.indexes[k] for k in dimsub]
        nlllist = []
        N = len(dimsub)
        for l_ in it.product(*coordssub):
            sub_dict = {dimsub[i]: l_[i] for i in range(N)}
            X_ = X.loc[sub_dict]
            nlllist.append(ar1nll(x0, X_.values))
        nll = np.sum(nlllist)
    else:
        nll = ar1nll(x0, X.values)
    return nll

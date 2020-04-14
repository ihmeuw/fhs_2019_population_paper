import numpy as np
import PyMB
import pandas as pd
import site
import os
import xarray as xr
from fbd_model.exc import ConvergenceError
from fbd_model.gkutils import ParameterEffects, xrmerge, omega_priors, \
    omega_translate, parse_covariates, np2xr, modcolarray, region_transform, \
    transform_parameter_array


class GKModel(PyMB.model):
    """
    GKModel is a subclass of the PyMB model class. It allows for ease of data
    prep as long as the expected variables are included in a DataFrame which
    is passed in at the initialization step of the class.
    """

    def __init__(self, dataset, fixed_effects, random_effects, constant=[],
                 y="y", omega_amp=0., weight_decay=0., draws=100, start=1990,
                 holdout=2017, *args, **kwargs):
        """
        :param dataset: Dataset
            Dataset with a y key which you wish to forecast.
        :param fixed_effects: dict of list of tuples or str
            dictionary of lists of either strings or tuples to parse for model
        :param random_effects: dict of list of tuples or str
            dictionary of lists of either strings or tuples to parse for model
        :param constant: list of str
            list of strings referring to contsnats to be added to the model
        :param y: str
            the name of the response variable
        :param omega_amp: float or dict
            how much to amplify the omega priors by
        :param weight_decay: float
            how much to decay the weight of the likelihood on predictions as
            they get farther away from the last year in sample
        :param start: int
            Year to start for the analysis
        :param holdout: int
            Year where data likelihood ends and forecast begins must be > start
        :param draws: int
            Number of draws to make for predictions.
        """
        PyMB.model.__init__(self, name="GKModel")
        for k in site.getsitepackages():
            f_ = os.path.join(k, "GKModel.so")
            if os.path.isfile(f_):
                model = f_
        model = MODELPATH / "GKModel.so"
        self.load_model(model)
        final_array = xrmerge(dataset)
        final_array_col = xrmerge(dataset, collapse=True)
        self.resid = None
        self.conv = False
        self.sigma_global = None
        assert start < holdout, "Year specifications not valid."
        self.start = start
        self.holdout = holdout
        self.draw_num = draws
        self.y = y
        self.data = dict()
        self.init = dict()
        self.dataset = dataset
        self.data["mean_adjust"] = int(0)  # depreceated
        self.data["testing_prints"] = int(0)  # depreceated
        self.data["weight_decay"] = weight_decay
        rset = random_effects
        self.random_effects_ = [g for g in rset.keys() if len(rset[g]) != 0]
        omega_dict = omega_translate(omega_amp)
        self.data["location"] = self.dataset.location_id.values
        self.data["age"] = self.dataset.age_group_id.values
        self.data["year"] = self.dataset.year_id.values
        self.data['holdout_start'] = \
            np.where(self.data["year"] == self.holdout)[0][0]
        # depreceated
        self.data["covariates2"], self.data["beta2_constraint"] = \
            parse_covariates([])

        param_effects = ParameterEffects(fixed_effects, random_effects,
                                         constant)
        param_effects.extract_param_data(self.dataset, final_array,
                                         final_array_col)
        self.data["X2"] = np.zeros((0, 0, 0, 0))  # depreceated
        self.data["X2_draw"] = np.zeros((0, 0, 0, 0))  # depreceated
        self.data.update(param_effects.data)
        self.init.update(param_effects.init)
        self.fixed_names = param_effects.fixed_names
        self.random_names = param_effects.random_names
        self.data["y_T"] = final_array_col.loc[dict(cov=self.y)].values
        self.data["y_U"] = final_array_col.loc[dict(cov=self.y)].values
        self.data["y_T_draws"] = final_array.loc[dict(cov=self.y)]

        # get region mappings
        self.data["region"], self.data["super_region"] = \
            (np.repeat(0, len(self.data["location"])),
             np.repeat(0, len(self.data["location"])))

        if "region_id" in list(self.dataset.data_vars.keys()):
            self.data["region"] = \
                pd.factorize(self.dataset.region_id.values)[0]

        if "super_region_id" in list(self.dataset.data_vars.keys()):
            self.data["super_region"] = \
                pd.factorize(self.dataset.super_region_id.values)[0]

        # omega priors
        if omega_amp != 0:
            self.data.update(omega_priors(self.data["y_U"],
                                          self.data['holdout_start'],
                                          level="U", amp=omega_dict))
            self.data.update(omega_priors(self.data["y_T"],
                                          self.data['holdout_start'],
                                          level="T", amp=omega_dict))
        else:
            okeys = ['omega_age_T', 'omega_age_U', 'omega_age_time_T',
                     'omega_age_time_U', 'omega_loc_T', 'omega_loc_U',
                     'omega_loc_time_T', 'omega_loc_time_U']
            self.data.update({k: 0. for k in okeys})

        # depreceated two level modeling
        self.data["has_risks"] = np.zeros_like(self.data["age"], dtype=int)

        # depreceated ar effects process
        self.init.update({'log_age_sigma_pi': np.array([]),
                          'log_location_sigma_pi': np.array([]),
                          'logit_rho': np.array([]),
                          'pi': np.zeros((0, 0, 0))})

        self.data_draw = dict()
        for k in list(self.data.keys()):
            if k.endswith("draw") or k.endswith("draws"):
                self.data_draw[k] = self.data.pop(k)

        _, _, _, k2 = self.data['X2'].shape
        r = np.sum(self.data['has_risks'])
        self.init['log_sigma_U'] = np.zeros(0 if r == 0 else 1)
        self.init['log_sigma_T'] = 0.
        self.init['log_zeta_D'] = np.zeros(r)
        self.init['beta2_raw'] = np.zeros((1, r, 1, k2))


    def optimize(self, opt_fun='nlminb', method='L-BFGS-B', verbose=False,
                 converror=True, **kwargs):

        """
        Run the pymb parent class optimize method with the draw number set

        :param opt_fun: str, default 'nlminb'
            the R optimization function to use (e.g. 'nlminb' or 'optim')
        :param method: str, default 'L-BGFS-B'
            method to use for optimization
        :param verbose: boolean, default False
            whether to print detailed optimization state
        :param converror: boolean, default True
            Should the code break on non-convergence
        :**kwargs: additional arguments to be passed to the R optim function
        """
        PyMB.model.optimize(self, opt_fun=opt_fun, method=method,
                            draws=self.draw_num, verbose=verbose,
                            random=self.random_effects_, **kwargs)
        if self.convergence != 0 and converror:
            print ("Covergence flag: " + str(self.convergence))
            raise ConvergenceError("Model failed to converge.")


    def fit(self, *args, **kwargs):
        """
        Optimize model and return fit parameters.

        Returns:
            xarray.Dataset: a dataset containing the coefficients for the
                fixed and random effects fit in the model. There is one
                variable per effect type (e.g. ``gamma_location_age``), and
                a covariate dimension specifying which covariate is associated
                with which coefficients. All covariate/effect-type pairs that
                were not fit (e.g., `gamma_age` and `sdi` if `sdi` was included
                as a global fixed effect) contain NaN values.
        """
        self.optimize(*args, **kwargs)

        ## get fit parameters into xarray format
        # use to keep track of parameter order
        dem_coords = dict(location_id=self.data["location"],
                          year_id=self.data["year"],
                          age_group_id=self.data["age"])

        # get the names of the fixed effects (e.g. "beta_global_raw")
        fixed_effects = [k + "_raw" for k in self.fixed_names if k +
                         "_raw" in list(self.parameters.keys())]
        # get the names of the random effects
        rand_effects = [k for k in self.random_names if
                        k in list(self.parameters.keys())]

        # get the draws of the FE and RE params - does not involve cov values
        fixed_arrays = [transform_parameter_array(self.draws(k),
                    self.data[k.replace('raw', 'constraint')])
                for k in fixed_effects]
        rand_arrays = [self.draws(k) for k in rand_effects]

        # get names of fixed covs (e.g. "intercept" and "sdi" for beta_global)
        fixed_cov_names = [self.data[k.replace("_raw", "")]
                           for k in fixed_effects]
        # same for random effects covariate names
        rand_cov_names = [self.data[k] for k in rand_effects]

        # if region or super region is part of the effects structure,
        # copy it out so the param exists for every location
        fixed_arrays = \
            [region_transform(fixed_arrays[i], self.data["region"])
             if "beta_region" in fixed_effects[i]
             else fixed_arrays[i] for i in range(len(fixed_arrays))]
        fixed_arrays = \
            [region_transform(fixed_arrays[i], self.data["super_region"])
             if "super_region" in fixed_effects[i]
             else fixed_arrays[i] for i in range(len(fixed_arrays))]
        rand_arrays = \
            [region_transform(rand_arrays[i], self.data["region"])
             if "gamma_region" in rand_effects[i]
             else rand_arrays[i] for i in range(len(rand_arrays))]
        rand_arrays = \
            [region_transform(rand_arrays[i], self.data["super_region"])
             if "super_region" in rand_effects[i]
             else rand_arrays[i] for i in range(len(rand_arrays))]

        # convert to xarray
        fixed_arrays = [np2xr(r, dem_coords, fixed_cov_names[i])
                        for i, r in enumerate(fixed_arrays)]
        rand_arrays = [np2xr(r, dem_coords, rand_cov_names[i])
                       for i, r in enumerate(rand_arrays)]

        # put the random and fixed effect parameters into one xarray.Dataset
        self.additive_params = dict()
        for i, k in enumerate(fixed_effects):
            self.additive_params[k.replace("_raw", "")] = fixed_arrays[i]
        for i, k in enumerate(rand_effects):
            self.additive_params[k] = rand_arrays[i]
        self.additive_params = xr.Dataset(self.additive_params)

        return self.additive_params


    @staticmethod
    def predict(ds, fit_params, start, end, constant_vars=[]):
        """
        Generate predictions from the year `start` up through the year `end`
        for input covariates using the fit_params. Variables with a coefficient
        defined to be 1 (like scalars), specified in constant_vars, are added
        on. Currently there is no intercept_shift option

        Args:
            ds (xarray.Dataset): dataset containing all relevant covariates,
                including the scalar if applicable, as data_variables.
            fit_params (xarray.Dataset): dataset containing the fit parameters
                for all relevant covariates. `covariate` is listed as a
                dimension, while each type/level of covariate (e.g.,
                `gamma_location_age`) is a separate variable
            start (int): first year of predictions
            end (int): last year of predictions
            constant_vars (list[str]): a list of all the variables to add on
                with an implied coefficient of 1. Example: `["ln_risk_scalar"]`
        Returns:
            xarray.DataArray: data array containing the predictions from the
                input covariates and fit parameters
        """
        # convert fit_params to an array in order to sum more easily
        fit_params = fit_params.to_array().fillna(0.)

        # loop through covariates and add on their contribution to the total
        pred_ds = None
        for cov in fit_params.cov.values:
            contribution = (ds[cov] * fit_params.sel(cov=cov)).sum("variable")
            if pred_ds is None:
                pred_ds = contribution
            else:
                pred_ds = pred_ds + contribution

        # add on the data from the constant_vars
        for var in constant_vars:
            pred_ds = pred_ds + ds[var]

        return pred_ds.sel(year_id=list(range(start, end+1)))



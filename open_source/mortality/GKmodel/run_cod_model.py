#!/usr/bin/env python
import sys

import numpy as np
from sksparse.cholmod import CholmodNotPositiveDefiniteError
import xarray as xr

from fbd_core import argparse
from fbd_core.etl.post_hoc_relu import post_hoc_relu
from fbd_model.exc import ConvergenceError
from fbd_model.model.GKModel import GKModel
from fbd_model.model.RepeatLastYear import RepeatLastYear

from fbd_cod.capping_mt import cap_acause_sex
from fbd_cod import settings as ss
from fbd_cod.downloaders import load_cod_dataset, read_paf_covs
from fbd_cod.qa import write_cod_forecast, write_cod_betas
from fbd_core.etl import expand_dimension


CUTOFF = 1000
NLOCS_CUTOFF = 150

def _repeat_last_year(ds, years, draws):
    """
    Run a model for mortality using the RepeatLastYear model

    Args:
        ds (xarray.Dataset): Dataset containing y as the response variable
            and time_var as the time variable
        years (YearRange): past and forecasted years (e.g. 1990:2017:2040)
        draws (int): number of draws to return (will all be identical)

    Returns:
        DataArray: the past and projected values (held constant)
    """
    print("running repeat_last_year")
    rly = RepeatLastYear(ds["y"], forecast=years.forecast_end,
                         holdout=years.forecast_start,
                         start=years.past_start)
    rly.fit()
    d = xr.DataArray(np.ones(draws), coords={"draw": np.arange(draws)},
                     dims=["draw"])
    s = xr.DataArray([1, 1, 1], dims=["scenario"],
                     coords=[[-1, 0, 1]])
    preds = expand_dimension(rly.predict(), "draw", np.arange(draws))
    return expand_dimension(preds, "scenario", ss.SCENARIOS)


def _get_addcovs(acause):
    """
    Returns a list of the additional covariates associated with a given acause

    Args:
        acause (str): Cause to find covariates for

    Returns:
        list(str): The non-sev covariates associated with the input acause
    """
    addcovs = []
    if acause.startswith("inj_trans_road"):
        addcovs.append("vehicles_2_plus_4wheels_pc")
    if acause.startswith("maternal"):
        addcovs.append("asfr")
    if acause == "maternal_hiv":
        addcovs.append("hiv")
    print("Additional covariates: {}".format(addcovs))
    return addcovs


def _get_sex_name(sex_id):
    """
    Gets the sex name associated with a given sex_id

    Args:
        sex_id (int): 1 or 2

    Returns:
        str: '_male' for 1 or '_female' for 2
    """
    assert sex_id in [1, 2], "sex_id must be 1 or 2"
    return "_male" if sex_id == 1 else "_female"


def main(acause, version, sex_id, sdi_time_interaction, oos, years, draws,
         subnat, spline=True):
    """
    Run a cause-of-death model using the appropriate strategy for the input
    acause and save the results. This file must be run from the `scripts`
    directory of fbd_cod.

    Args:
        acause (str): the acause being forecasted.
        version (str): name of the model (i.e. version).
        sex_id (int): sex_id for which the acause is being modeled.
        sdi_time_interaction (bool): whether or not to include an interaction
            term on SDI*time
        oos (bool): whether or not to use inputs that have been run on time
            series holdouts
        years (YearRange): first year of the past, the first year of the
            forecast, and the last year of the forecast.
        draws (int): number of draws to run the model on
        subnat (bool): whether or not to include the subnational
            locations
        spline (bool): whether or not ot include a spline on SDI to separate
            out the different effects SDI has in high-income vs low-income
            countries
    """
    versions = ss.VERSIONS_OOS if oos else ss.VERSIONS
    sex_name = _get_sex_name(sex_id)
    addcovs = _get_addcovs(acause)
    ds = load_cod_dataset(acause, sex_id, years=years, draws=draws, oos=oos,
                          addcovs=addcovs, subnat=subnat)
    if spline:
        ds["sdi_part1"] = xr.ufuncs.minimum(ds["sdi"], ss.SDI_KNOT)
        ds["sdi_part2"] = xr.ufuncs.maximum(ds["sdi"] - ss.SDI_KNOT, 0.)

    if sdi_time_interaction:
        ds["sdi_time"] = ds["sdi"] * ds["time_var"]


    # ntd_nema isn't stable enough to have a regular gk model run on it
    if acause in ["ntd_nema", "ntd_dengue"]:
        print("Running repeat_last_year model for " + acause + ": " + sex_name)
        preds = _repeat_last_year(ds, years, draws)
        preds.coords["sex_id"] = sex_id
        write_cod_forecast(preds, acause, version, gbd_round_id=ss.GBD_ROUND_ID,
                           addendum=sex_name)
        sys.exit()

    ds["y"] = ds["y"].mean("draw")

    # Don't put a spline on SDI for causes that aren't modeled for very many
    # locations.
    nlocs = len(ds.location_id.values)
    if (nlocs > NLOCS_CUTOFF) and spline:
        fxeff = {"beta_global": [("intercept", 0), ("sdi_part1", 0),
                                 ("sdi_part2", 0), ("time_var", 0)]}
    else:
        fxeff = {"beta_global": [("intercept", 0), ("sdi", 0),
                                 ("time_var", 0)]}

    if sdi_time_interaction:
        fxeff["beta_global"] += [("sdi_time", 0)]

    raneff = {"gamma_location_age": ["intercept"],
              "gamma_age": ["time_var"]}

    reis = read_paf_covs(acause, sex_id, versions["sev"][0],
                         versions["sev"][1], draws=draws,
                         listonly=True)
    for r in reis:
        if acause == "nutrition_iron" and sex_id == 1:
            continue
        fxeff["beta_global"] += [(r, 1)]
        ds[r].values[np.isnan(ds[r].values)] = 0.

    if acause.startswith("inj_trans_road"):
        fxeff["beta_age"] = [("vehicles_2_plus_4wheels_pc", 0)]
        if not acause == "inj_trans_road_pedest":
            # drop global SDI and time trend
            for cov in [("sdi_part1", 0),
                        ("sdi_part2", 0),
                        ("sdi", 0),
                        ("time_var", 0)]:
                try:
                    fxeff["beta_global"].remove(cov)
                except ValueError:
                    pass
            # drop time if present in gamma_age
            raneff = {"gamma_location_age": ["intercept"]}
    if acause.startswith("maternal_hiv"):
        fxeff["beta_global"] += [("hiv", 1)]
#        fxeff["beta_age"] = [("hiv", 1)]

    if acause in ss.VACCINE_CAUSES:
        try:
            fxeff["beta_global"].remove(("sdi_part1", 0))
            fxeff["beta_global"].remove(("sdi_part2", 0))
        except ValueError:
            fxeff["beta_global"].remove(("sdi", 0))

    if acause.startswith("maternal"):
        fxeff["beta_global"].append(("asfr", 1))

    if acause in ["maternal_hiv", "maternal_other", "maternal_indirect"]:
        fxeff["beta_global"].remove(("time_var", 0))
        raneff["gamma_age"].remove("time_var")

    weight_decay = 0.9 if acause.startswith('inj_homicide') else 0

    gkmodel = GKModel(ds, fixed_effects=fxeff, random_effects=raneff,
                      draws=draws, constant=["ln_risk_scalar"], y="y",
                      omega_amp=0., weight_decay=weight_decay,
                      start=years.past_start, holdout=years.forecast_start)

    # Try fitting with all the specified fixed effects. If a convergence
    # error is given, drop all but the location-age intercepts, and set time
    # to be a global variable. If the model technically converges but any of
    # the variables have coefficients with unreasonably high standard
    # deviations, drop said variables and try again.
    try:
        params = gkmodel.fit()
        refit = False
        cov_list = ["sdi", "sdi_part1", "sdi_part2"] + reis
        if acause.startswith("maternal"):
            cov_list += ["asfr"]
        for covariate in cov_list:
            try:
                sd = params["beta_global"].sel(cov=covariate).std()
                median_coeff = params["beta_global"].sel(cov=covariate).median()
            except:
                continue
            # if it doesn't have reasonable sd, drop the cov
            if np.isnan(sd) or (abs(sd / median_coeff) > CUTOFF):
            #if np.isnan(sd) or sd > CUTOFF:
                if covariate in ["sdi", "sdi_part1", "sdi_part2"]:
                    fxeff["beta_global"].remove((covariate, 0))
                else:
                    fxeff["beta_global"].remove((covariate, 1))
                refit = True
        if refit:
            raneff = {"gamma_location_age": ["intercept"]}
            print("refitting after dropping covariates")
            gkmodel = GKModel(ds, fixed_effects=fxeff, draws=draws,
                              omega_amp=0., random_effects=raneff,
                              constant=["ln_risk_scalar"],
                              forecast=years.forecast_end,
                              holdout=years.forecast_start,
                              start=years.past_start,
                              weight_decay=weight_decay, y="y")
            params = gkmodel.fit()
    except(ConvergenceError, CholmodNotPositiveDefiniteError):
        print("refitting after dropping sdi and all covs")
        # drop sdi and covariates if model still doesn't converge
        raneff = {"gamma_location_age": ["intercept"]}
        if acause.startswith("inj_trans_road_pedest"):
            fxeff["beta_global"] = [("intercept", 0)]
        else:
            fxeff["beta_global"] = [("time_var", 0), ("intercept", 0)]
        gkmodel = GKModel(ds, fixed_effects=fxeff, draws=draws,
                          omega_amp=0., random_effects=raneff,
                          constant=["ln_risk_scalar"],
                          forecast=years.forecast_end,
                          holdout=years.forecast_start,
                          start=years.past_start, weight_decay=weight_decay,
                          y="y")
        params = gkmodel.fit()

    # apply post-hoc RELU if the dictionary defined in settings is non-empty
    if ss.RELU:
        params = post_hoc_relu(params, ss.RELU)

    preds = GKModel.predict(ds, params, years.past_start,
                                years.forecast_end,
                                constant_vars=["ln_risk_scalar"])
    # cap maternal_hiv and drug causes
    if acause in ["maternal_hiv", "maternal_indirect", "mental_drug_opioids", "mental_drug_other",
                  "mental_drug_cocaine", "mental_drug_amphet"]:
        preds = cap_acause_sex(preds, acause, sex_id, years = years,
                               past_version = versions["past_mortality"][0])
    write_cod_forecast(preds, acause, version, gbd_round_id=ss.GBD_ROUND_ID,
                       addendum=sex_name)
    params = xr.DataArray([1], dims=["sex_id"], coords=[[sex_id]]) * params
    write_cod_betas(params, acause, version, gbd_round_id=ss.GBD_ROUND_ID,
                    addendum=sex_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Run last year constant model')
    parser.add_arg_years()
    parser.add_arg_draws()
    parser.add_arg_dryrun()
    parser.add_argument('-c', '--acause', type=str, required=True,
                        help='acause (e.g. "_all", "cvd_ihd", etc)')
    parser.add_argument('-s', '--sex_id', type=int, required=True,
                        help='sex_id either 1 or 2 as integer')
    parser.add_argument('--version', type=str, required=True,
                        help='run name/version id')
    parser.add_argument('--oos', action="store_true",
                        help=('Whether or not to use the out-ouf-sample '
                              'versions of inputs'))
    parser.add_argument('--sdi-interaction', action="store_true",
                        help=('Whether or not to include an interaction '
                              'term on time and sdi'))
    parser.add_argument('--subnational', action="store_true",
                        help=('Whether or not to run the model using '
                              'the subnational locations'))
    parser.add_argument('--spline', action="store_true",
                        help=('Whether to put a spline on SDI'))

    args = parser.parse_args()
    if args.dryrun:
        sys.exit()
    main(args.acause, args.version, args.sex_id, args.sdi_interaction, args.oos,
         args.years, args.draws, args.subnational, args.spline)

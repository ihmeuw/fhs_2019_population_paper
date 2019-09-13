import numpy as np
import pandas as pd
import xarray as xr

from sqlalchemy.orm import sessionmaker

from fbd_core import db
from fbd_core.etl.transformation import resample
from fbd_core.file_interface import FBDPath
from fbd_cod import settings as ss
from fbd_core.strategy_set.strategy import get_strategy_set
from fbd_core.etl import expand_dimensions

LOC_DF = db.get_modeled_locations()
ENGINE = db.db_engine(ENGINE_NAME, DATABASE_NAME)
DIRECT_MODELED_PAFS = ["drugs_illicit_direct", "unsafe_sex"]

def validate_cause(acause):
    """
    Check whether a cause is in the cod cause list. Throws an assertion error
    if not.

    Args:
        acause (str): Cause to check whether it is a modeled cod acause.
    """
    session = sessionmaker(bind=ENGINE)()
    causes = get_strategy_set(session, FATAL_GK_STRATEGY_ID,
                              CAUSE_HIERARCHY_ID).acause.values
    assert acause in causes, "acause must be a valid modeled cod acause."


def validate_version(version, variable, gbd_round_id):
    """
    Check whether or not the version to load from exists. Throws an assertion
    error if not.

    Args:
        version (str): the name of the version to check for existence
        variable (str): the metric to check the version under (e.g. scalar)
        gbd_round_id (int): the gbd_round under which the version is stored
    """
    assert FILEPATH.exists(), "version {} is not valid".format(version)


def get_parent_id(location_id):
    """Gets the parent id for a subnational location_id

    Args:
        location_id (int): the location_id to get the parent_id for
    """
    parent_id = ALL_LOCS.query("location_id==@location_id").parent_id.values[0]
    return parent_id


def empty_dem_xarray(locs, val=0, draws=100, start=1990, end=2040):
    """
    Build an empty xarray which has all dimensions required for modeling i.e.
    location_id, age_group_id, year_id, sex_id, draw, & scenario.

    Args:
        locs (list[int]): list or array of locations to use as location_id
            coordinates
        val (float): value to fill the array with, default 0.
        draws (int): what the length of the draw dimension should be.
        start (int): the beginning of the year_id dimension, default 1990
        end (int): the end of the year_id dimension, default 2040

    Returns:
        DataArray: Six-dimensional data array.
    """
    demog = dict(age_group_id=db.get_ages(gbd_round_id=4).age_group_id.values,
                 year_id=np.arange(start, end+1),
                 location_id=locs,
                 scenario=ss.SCENARIOS, draw=np.arange(draws), sex_id=[1, 2])
    dims = ["location_id", "age_group_id", "year_id",
            "sex_id", "draw", "scenario"]
    size = [len(demog[x]) for x in dims]
    vals = np.ones(size) * val
    dem_array = xr.DataArray(vals, coords=demog, dims=dims)
    return dem_array


def load_scalar(acause, sex_id, model, gbd_round_id, years,
                draws, log=True, subnat=False):
    """
    Load scalar scenario data as an xarray.

    Params:
        acause (str): valid acause for scalar
        sex_id (int): the sex for the scalar to read; 1 for males, 2 for females
        model (str): scalar version name to load
        gbd_round_id (int): gbd_round under which scalar data is stored
        model (str): scalar version name to load
        gbd_round_id (int): gbd_round under which scalar data is stored
        years (YearRange): the years being used for the model
        draws (int): Number of draws to pull from the scalar data.
        log (bool): Whether to take the natural log of the values, default True
        subnat (bool): Whether to append the 93 CSU subnational locations

    Returns:
        DataArray: array with scalar data for acause
    """
    all_locs = LOC_DF.location_id.values.tolist()
    validate_cause(acause)

    session = sessionmaker(bind=ENGINE)()
    causes_with_scalars = get_strategy_set(session, SCALAR_PAIR_STRATEGY_ID,
                                           CAUSE_RISK_HIERARCHY_ID)
    acauses_with_scalars = causes_with_scalars.cause_id.map(
            db.get_acause).unique()

    if acause in acauses_with_scalars.tolist():
        future = xr.open_dataarray(str(FUTURE_FILEPATH))
        past = xr.open_dataarray(str(PAST_FILEPATH))
        da = xr.concat([past, future], dim="year_id")
    else:
        locs = all_locs if subnat else LOC_DF.location_id.values.tolist()
        da = empty_dem_xarray(locs, 1., draws, years.past_start,
                              years.forecast_end)


    da = resample(da, draws).loc[dict(sex_id=sex_id, year_id=years.years)]
    da.name = "risk_scalar"
    da = da.where(da <= ss.SCALAR_CAP).fillna(ss.SCALAR_CAP)
    if log:
        da = xr.ufuncs.log(da)
        da.name = "ln_risk_scalar"
    return da


def load_sdi(model, gbd_round_id, years, log=False, draws=100, subnat=False):
    """
    Loads and returns sociodemographic index

    Args:
        model (str): sdi version to load
        gbd_round_id (int): gbd_round that the sdi estimates are under
        years (YearRange): the years being modeled (e.g. 1990:2017:2040)
        log (bool): whether to take the log of SDI, default False
        draws (int): Number of draws to pull from the sdi data.
        subnat (bool): whether to append the 93 CSU subnational locations

    Returns:
        DataArray: array with sdi information.
    """
    da = xr.open_dataset(str(FILEPATH))["sdi"].sel(year_id=years.years)
    if subnat:
        subnat_model = ss.VERSIONS_SUBNAT["sdi"][0]
        subnat_gri = ss.VERSIONS_SUBNAT["sdi"][1]
        sdi_sub = xr.open_dataset(str(FILEPATH))["sdi"]
        sdi_sub = sdi_sub.sel(location_id=SUBNAT_DF.location_id.values)
        da = xr.concat([da, sdi_sub], dim="location_id")
    da = resample(da, draws)
    da.name = "sdi"
    if log:
        da = xr.ufuncs.log(da)
        da.name = "ln_sdi"
    return da


def load_cod_data(acause, sex_id, past_version, gbd_round_id, years, log=True,
                  draws=100):
    """
    Load in cause-sex specific mortality rate.

    Args:
        acause (str): valid acause for cod outcome.
        sex_id (int): sex for cod outcome; 1 for males, 2 for females
        past_version (str): the version of past death to pull from
        gbd_round_id (int): gbd round the mortality data are stored under
        years (YearRange): the years to load and model (e.g. 1990:2017:2040)
        log (bool): Whether to take the natural log of the mortality values.
        draws (int): Number of draws to pull from the mortality data.

    Returns:
        DataArray: array with acause-sex specific death rate information.
    """
    validate_cause(acause)

    da = xr.open_dataset(str(FILEPATH))
    d = xr.DataArray(np.ones(draws), coords={"draw": np.arange(draws)},
                     dims=["draw"])
    mean_array = da["mean"].loc[{
            "year_id": years.past_years[:-1]  # leave the last off for draws
        }] * d
    draw_array = da["value"].loc[{"year_id": [years.past_end]}]
    draw_array = resample(draw_array, draws)
    da_draw = xr.concat([mean_array, draw_array], dim="year_id")
    locdiv = ("draw", "age_group_id", "year_id", "sex_id")
    locidx = np.where(~(da_draw == 0).all(locdiv))[0]
    locs = da_draw.location_id.values[locidx]
    agediv = ("draw", "location_id", "year_id", "sex_id")
    ageidx = np.where(~(da_draw == 0).all(agediv))[0]
    ages = da_draw.age_group_id.values[ageidx]
    demdict = dict(location_id=locs, age_group_id=ages, sex_id=sex_id)
    if acause == "ntd_nema":
        da_draw += ss.FLOOR
    if log:
        da_draw = xr.ufuncs.log(da_draw)
    da_draw_sub = da_draw.copy().loc[demdict]
    return da_draw_sub


def read_cov(cov, sex_id, model, gbd_round_id, years, draws, subnat=False):
    """
    Read a covariate and format for cod modeling. Fills in the country-level
    covariates for subnational locations if applicable.

    Args:
        cov (str): covariate name to load
        sex_id (int): sex for covariate; 1 for male, 2 for female
        model (str): version name to load the covariate from
        gbd_round_id (int): gbd round under which the cov is stored
        years (YearRange): the years being loaded and modeled (e.g.
            1990:2017:2040)
        draws (int): Number of draws to pull from the covariate data.
        subnat (bool): Whether or not to include the 93 CSU subnational
            locations

    Returns:
        DataArrray: array with covariate information loaded and formatted
    """
    versions = ss.VERSIONS
    # append past data to asfr forecasts
    if cov == "asfr":
        past_cov = xr.open_dataarray(str(FILEPATH)).sel(year_id=years.past_years)
        past_cov = expand_dimensions(past_cov, scenario=[-1, 0, 1])
        future_cov = xr.open_dataarray(str(FILEPATH)).sel(year_id=years.forecast_years)
        raw = xr.concat([past_cov, future_cov], dim = "year_id")
    else:
        raw = xr.open_dataarray(str(FILEPATH)).sel(year_id=years.years)
    raw = resample(raw, draws)
    badkeys = np.setdiff1d(list(raw.coords.keys()), list(raw.coords.indexes.keys()))
    for k in badkeys:
        raw = raw.drop(k)

    raw_coords_keys = list(raw.coords.keys())
    # the return data shouldn't include any aggregate age groups
    agg_ages = [22, 27]
    if "age_group_id" in raw_coords_keys:
        # drop point coordinate
        if len(raw["age_group_id"]) == 1:
            raw = raw.drop("age_group_id").squeeze("age_group_id")
        # keep all-ages, not age-standardized
        elif raw.age_group_id.values == agg_ages:
            raw = raw.loc[{"age_group_id": 22}]
        # given the choice between age-specific or all-ages, choose age-specific
        else:
            raw = raw.loc[{"age_group_id": np.setdiff1d(
                    raw["age_group_id"].values, agg_ages)}]

    # similarly, the return data shouldn't include aggregated sexes
    if "sex_id" in raw_coords_keys:
        if sex_id in raw["sex_id"].values:
            raw = raw.loc[{"sex_id": sex_id}].drop("sex_id")
        elif (raw["sex_id"].values == 3):
            raw = raw.drop("sex_id").squeeze("sex_id")
        else:
            print("this covariate doesn't have the sex_id you're looking for")
            raise SystemExit

    if "location_id" in raw_coords_keys:
        locs = db.get_modeled_locations(gbd_round_id).location_id.values
        raw = raw.loc[{"location_id": np.intersect1d(
                locs, raw["location_id"].values)}]

    return raw


def read_sev(rei, sex_id, model, gbd_round_id, draws=100, subnat=False):
    """
    Read in summary exposure value information.

    Args:
        rei (str): Risk, etiology, or impairment to load.
        sex_id (int): Sex to read for the sev; 1 for males, 2 for females
        model (str): version name of sevs to read
        gbd_round_id (int): gbd round under which sev data is saved
        draws (int): Number of draws to return
        subnat (bool): whether to append the 93 CSU subnational locations

    Returns:
        DataArray: xarray with sev data loaded and formatted
    """
    da = xr.open_dataarray(str(FILEPATH)).loc[dict(sex_id=sex_id)]
    if subnat:
        subnat_model = ss.VERSIONS_SUBNAT["sev"][0]
        subnat_gri = ss.VERSIONS_SUBNAT["sev"][1]
        da_sub = xr.open_dataarray(str(FILEPATH)).loc[dict(sex_id=sex_id)]
        da = xr.concat([da, da_sub], dim="location_id")
    da = resample(da, draws)
    single_coords = np.setdiff1d(list(da.coords.keys()), da.dims)
    da = da.drop(labels=single_coords)
    return da


def read_paf_covs(acause, sex_id, model, gbd_round_id=4,
                  draws=100, listonly=False, subnat=False,
                  include_uncertainty=False):
    """
    Return a Dataset of sev data for a cause or alternatively a list of the
    applicable sevs.

    Args:
        acause (str): acause to pull data for
        sex_id (int): sex_id for covs/sevs; 1 for males, 2 for females
        model (str): name of sev version to use
        gbd_round_id (int): gbd_round that the sev data are saved under
        draws (int): Number of draws to return
        listonly (bool): if True, return just a list of applicable sevs,
            otherwise return the full dataset
        subnat (bool): whether to append the 93 CSU subnational locations
        include_uncertainty (bool): Whether to include past uncertainty
            (otherwise just copy the mean)

    Returns:
        Dataset: dataset whose datavars are the arrays for each relevant sev
    """

    session = sessionmaker(bind=ENGINE)()
    paf1_risk_pairs = get_strategy_set(session, PAF1_STRATEGY_ID,
                                       CAUSE_RISK_HIERARCHY_ID)
    cause_id = db.get_cause_id(acause)

    rei_ids = paf1_risk_pairs.query("cause_id == @cause_id").rei_id.values

    ds = xr.Dataset()
    all_reis = [db.get_rei(int(rei_id)) for rei_id in rei_ids]
    most_detailed_risks = get_strategy_set(session, MOST_DETAILED_STRATEGY_ID,
                                           RISK_HIERARCHY_ID)["rei"].unique()
    # subset to most detailed reis
    reis = [rei for rei in all_reis if (rei in most_detailed_risks) and not
            (rei in DIRECT_MODELED_PAFS)]

    for r in reis:
        da = read_sev(r, sex_id, model, gbd_round_id, draws)
        ds[r] = da
    if not include_uncertainty and len(reis) > 0:
        ds = ds.median('draw') * xr.DataArray(np.ones(draws),
                                             dims='draw',
                                             coords={'draw': np.arange(draws)})
    if listonly:
        return reis
    else:
        return ds


def load_cod_dataset(acause, sex_id, years, draws, oos, addcovs, log=True,
                     logsdi=False, subnat=False, sev_covariate_draws=False):
    """
    Load in acause-sex specific mortality rate along with scalars, sevs, sdi,
    and other covariates if applicable.

    Args:
        acause (str): valid acause for cod outcome.
        sex_id (int): sex to load; 1 for males, 2 for females
        years (YearRange): years to load and model (e.g. 1990:2017:2040)
        draws (int): number of draws to return in the dataset
        oos (bool): whether the forecasts are to be run using inputs that
            have been forecasted on a held-out time-series
        addcovs (list[str]): list of cause-specific covariates to add to the
            dataset
        log (bool): Whether to take the natural log of the cod values.
        logsdi (bool): Whether to take the natural log of the sdi values.
        sev_covariate_draws (bool): Whether to include draws of the past SEVs
            used covariates or simply take the mean.

    Returns:
        Dataset: xarray dataset with cod mortality rate and scalar, sev, sdi,
            and other covariate information
    """
    versions = ss.VERSIONS_OOS if oos else ss.VERSIONS

    # check validity of inputs
    validate_cause(acause)
    validate_version(versions["scalar"][0], "scalar", versions["scalar"][1])
    validate_version(versions["sev"][0], "sev", versions["sev"][1])

    regdf = db.get_modeled_locations(ss.GBD_ROUND_ID)[["location_id",
                                                    "region_id",
                                                    "super_region_id"]]
    if subnat:
        regdf = pd.concat([regdf, SUBNAT_DF])[["location_id",
                                               "region_id",
                                               "super_region_id"]]
    regdf.set_index("location_id", inplace=True)
    year_list = years.years
    time_array = xr.DataArray(year_list - years.past_start, dims=["year_id"],
                              coords=[year_list])
    codda = load_cod_data(acause, sex_id,
                          versions["past_mortality"][0],
                          versions["past_mortality"][1],
                          years, log, draws=draws)
    agevals = codda.coords["age_group_id"].values
    locvals = codda.coords["location_id"].values
    demdict = dict(year_id=year_list, age_group_id=agevals, location_id=locvals)

    ds = xr.Dataset(
        dict(
            y=codda.copy(),
            sdi=load_sdi(versions["sdi"][0], versions["sdi"][1], years,
                         log=logsdi, draws=draws, subnat=subnat).copy(),
            ln_risk_scalar=load_scalar(acause, sex_id, versions["scalar"][0],
                                       versions["scalar"][1],
                                       years, log=True, draws=draws,
                                       subnat=subnat).copy(),
            intercept=xr.DataArray(1).copy(),
            time_var=time_array.copy()))
    for cov in addcovs:
        ds[cov] = read_cov(cov, sex_id, versions[cov][0], versions[cov][1],
                           years, draws=draws, subnat=subnat)
    ds.update(read_paf_covs(acause, sex_id, versions["sev"][0],
                            versions["sev"][1],
                            include_uncertainty=sev_covariate_draws,
                            draws=draws, subnat=subnat))
    ds_sub = ds.copy().loc[demdict]
    ds_sub.update(xr.Dataset(regdf))
    ds_sub.y.values[(ds_sub.y == -np.inf).values] = np.nan
    ds_sub.ln_risk_scalar.values[(np.isnan(ds_sub.ln_risk_scalar.values))] = 0.

    # select just non-aggregate values
    loc_vals = list(set(regdf.reset_index().location_id.values.tolist()) &
                    set(ds_sub.location_id.values.tolist()))
    ds_sub = ds_sub.loc[{"location_id": loc_vals}]
    return ds_sub

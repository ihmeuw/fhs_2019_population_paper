import collections
from functools import lru_cache
import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy.orm import sessionmaker
import xarray as xr

from fbd_core import db
from fbd_core.file_interface import FBDPath
from fbd_core.strategy_set.strategy import get_strategy_set
from fbd_core.strategy_set.query import get_hierarchy, get_hierarchy_version_id


LOGGER = logging.getLogger(__name__)


# define the demographic indices
LOCATION_DIM = "location_id"
YEAR_DIM = "year_id"
SEX_DIM = "sex_id"
AGE_DIM = "age_group_id"
DRAW_DIM = "draw"
SCENARIO_DIM = "scenario"
QUANTILE_DIM = "quantile"

SCENARIOS = (0, 1, 2)
SEXES = (1, 2)

CAUSE_SET_ID = 6  # the forecasting cause set.  Semi-static.
CAUSE_STRATEGY_ID = 16
REI_SET_ID = 6  # the forecasting rei set.
REI_STRATEGY_ID = 15
REI_INTERVENTION_ID = 10
CAUSE_RISK_SET_ID = 3  # the forecasting cause-risk set.
CAUSE_RISK_STRATEGY_ID = 17  # calculated PAFs -- i.e. have SEVs and RRmaxes
CAUSE_RISK_DIRECTLY_MODELED_SET_ID = 32
CAUSE_RISK_MAYBE_NEGATIVE_PAF_SET_ID = 33


@lru_cache(1)
def demographic_coords(gbd_round_id, years):
    """
    Creates and caches an OrderedDict of demographic indices

    Args:
        gbd_round_id (int): gbd round id.
        years (YearRange): [past_start, forecast_start, forecast_end] years.

    Returns:
        OrderedDict: ordered dict of all non-draw dimensions and their
            coordinates.
    """
    if not (hasattr(demographic_coords, "coords") and
            demographic_coords.gbd_round_id == gbd_round_id and
            demographic_coords.years == years):

        location_ids =\
            db.get_modeled_locations(gbd_round_id=gbd_round_id).\
            location_id.values.tolist()

        age_group_ids =\
            db.get_ages(gbd_round_id=gbd_round_id)[AGE_DIM].unique().tolist()

        _coords =\
            collections.OrderedDict([(LOCATION_DIM, location_ids),
                                     (AGE_DIM, age_group_ids),
                                     (SEX_DIM, list(SEXES)),
                                     (YEAR_DIM, years.years),
                                     (SCENARIO_DIM, list(SCENARIOS))])
        demographic_coords.coords = _coords
        demographic_coords.gbd_round_id = gbd_round_id
        demographic_coords.years = years

    return demographic_coords.coords


@lru_cache(1)
def _get_cause_risk_pairs(gbd_round_id):
    """
    Returns most-detailed cause-risk pairs for the scalars pipeline.

    Args:
        gbd_round_id (int): gbd round id.

    Returns:
        pd.DataFrame:  Most detailed cause-risk pairs.

    Raises:
        RuntimeError:
            If the sets of directly-modeled and calculated PAFs are not
            mutually exclusive
    """
    if gbd_round_id == 4:
        return _get_cause_risk_pairs_gbd2016()

    engine = db.db_engine("fbd-dev-read", database="forecasting")
    session = sessionmaker(bind=engine)()

    cause_hierarchy_version_id = get_hierarchy_version_id(
            session, entity_type="cause", entity_set_id=CAUSE_SET_ID,
            gbd_round_id=gbd_round_id)
    causes = get_strategy_set(session, strategy_id=CAUSE_STRATEGY_ID,
                              hierarchy_id=cause_hierarchy_version_id)

    rei_hierarchy_version_id = get_hierarchy_version_id(
            session, entity_type="risk", entity_set_id=REI_SET_ID,
            gbd_round_id=gbd_round_id)
    risks = get_strategy_set(session, strategy_id=REI_STRATEGY_ID,
                             hierarchy_id=rei_hierarchy_version_id)

    cr_hierarchy_version_id = get_hierarchy_version_id(
            session, entity_type="cause_risk_pair",
            entity_set_id=CAUSE_RISK_SET_ID, gbd_round_id=gbd_round_id)
    calculated_paf_set = get_strategy_set(
            session, strategy_id=CAUSE_RISK_STRATEGY_ID,
            hierarchy_id=cr_hierarchy_version_id)
    directly_modeled_paf_set = get_directly_modeled_pafs(gbd_round_id)
    crs = pd.concat([calculated_paf_set, directly_modeled_paf_set])
    if crs.duplicated().any():
        err_msg = ("The sets of directly-modeled and calculated PAFs are not "
                   "mutually exclusive")
        LOGGER.error(err_msg)
        raise RuntimeError(err_msg)

    cause_risk_pairs = crs[["cause_id", "rei_id"]].\
        merge(causes[["cause_id", "acause"]], on="cause_id").\
        merge(risks[["rei_id", "rei"]], on="rei_id")

    session.close()
    engine.dispose()

    return cause_risk_pairs


@lru_cache(1)
def _get_cause_risk_pairs_gbd2016():
    """We only want to include directly-modeled PAF pairs that are in GBD 2017
        since that all what we have FHS data for."""
    gbd2016_round_id = 4

    # This list should include most cause-risk pairs that we calculate PAFs for
    # with the exception of that we drop GBD 2016 directly modeled PAF pairs,
    # and then append GBD 2017 PAF pairs, which are not mutually exclusive.
    initial_cause_risks = db.get_cause_risk_pairs_joint(paf_1=0, in_scalar=1)
    if initial_cause_risks.empty:
        err_msg = (
            "query for all gbd2016 cause-risk pairs did not return any results"
            )
        LOGGER.error(err_msg)
        raise RuntimeError(err_msg)
    else:
        initial_cause_risks = initial_cause_risks[
            ["cause_id", "acause", "rei", "rei_id"]]

    # Filter out the pairs that had directly modeled PAFs
    cause_risk_not_modeled = initial_cause_risks.query(
        "rei not in @GBD2016_MODELED_PAF_RISKS")
    # Filter out the pairs that are only directly modeled with injury causes
    cause_risk_not_modeled = cause_risk_not_modeled[~(
        (cause_risk_not_modeled["acause"].str.startswith("inj_"))
        & (cause_risk_not_modeled["rei"].isin(
            GBD2016_MODELED_PAF_INJURY_RISKS))
        )]

    directly_modeled_paf_set = get_directly_modeled_pafs(gbd2016_round_id)
    if directly_modeled_paf_set.empty:
        err_msg = (
            "query for cause-risk pairs with directly-modeled PAFs did not "
            "return any results"
            )
        LOGGER.error(err_msg)
        raise RuntimeError(err_msg)
    else:
        directly_modeled_paf_set = directly_modeled_paf_set[
            ["cause_id", "acause", "rei_id", "rei"]]

    most_detailed_cause_ids = db.get_death_causes(
        aggregate=False)["cause_id"].unique()
    risk_hierarchy = get_risk_hierarchy(gbd2016_round_id)
    most_detailed_rei_ids = risk_hierarchy.query(
        "most_detailed == 1")["rei_id"].unique()

    most_detailed_dm_paf_set = directly_modeled_paf_set.query(
        "cause_id in @most_detailed_cause_ids"
        " and rei_id in @most_detailed_rei_ids")

    paf_all_set = pd.concat(
        [cause_risk_not_modeled, most_detailed_dm_paf_set])
    if paf_all_set.duplicated().any():
        err_msg = ("The sets of directly-modeled and calculated PAFs are not "
                   "mutually exclusive")
        LOGGER.error(err_msg)
        raise RuntimeError(err_msg)

    return paf_all_set


def is_directly_modeled(acause, rei, gbd_round_id):
    """Returns true if the cause-risk pair has a directly modeled PAF"""
    dm_pafs = get_directly_modeled_pafs(gbd_round_id)
    return not dm_pafs.query("acause == @acause and rei == @rei").empty


@lru_cache(1)
def get_directly_modeled_pafs(gbd_round_id):
    """Get cause-risk pairs that have directly-modeled PAFs"""
    engine = db.db_engine("fbd-dev-write", database="forecasting")
    session = sessionmaker(bind=engine)()

    if gbd_round_id == 4:
        gbd_round_id = 5   # use gbd2017 data

    cause_risk_hierarchy_version_id = get_hierarchy_version_id(
        session, entity_type="cause_risk_pair",
        entity_set_id=CAUSE_RISK_SET_ID, gbd_round_id=gbd_round_id)

    directly_modeled_paf_set = get_strategy_set(
        session, strategy_id=CAUSE_RISK_DIRECTLY_MODELED_SET_ID,
        hierarchy_id=cause_risk_hierarchy_version_id)

    # Set_only has cause_ids and rei_ids, so get acauses
    acause_cause_id_map = _acauses(
        directly_modeled_paf_set["cause_id"].unique())
    directly_modeled_paf_set_with_acause = directly_modeled_paf_set.merge(
        acause_cause_id_map, how="left")

    # Ensure that all cause-ids have acauses
    acauses_missing = (
        directly_modeled_paf_set_with_acause["acause"].notnull().any())
    acause_err_msg = "Some causes don't have acauses"
    assert acauses_missing, acause_err_msg

    # ... and get reis.
    rei_rei_id_map = _reis(
        directly_modeled_paf_set_with_acause["rei_id"].unique())
    directly_modeled_paf_set_with_rei = (
        directly_modeled_paf_set_with_acause.merge(rei_rei_id_map, how="left"))

    # Ensure that all rei-ids have reis
    reis_missing = directly_modeled_paf_set_with_rei["rei"].notnull().any()
    rei_err_msg = "Some reis don't have reis"
    assert reis_missing, rei_err_msg

    session.close()
    engine.dispose()

    return directly_modeled_paf_set_with_rei


def is_maybe_negative_paf(acause, rei, gbd_round_id):
    """Returns true if the cause-risk pair is maybe a negative PAF"""
    maybe_negative_pafs = get_maybe_negative_paf_pairs(gbd_round_id)
    return not maybe_negative_pafs.query(
        "acause == @acause and rei == @rei").empty


@lru_cache(1)
def get_maybe_negative_paf_pairs(gbd_round_id):
    """Get cause-risk pairs that *can* have negative PAFs, because they *can*
    be protective"""
    if gbd_round_id == 4:
        # Unfortunately these have to be hard-coded because we don't have
        # strategy sets for GBD 2016.
        return pd.DataFrame({
            "acause": ["cvd_ihd", "cvd_stroke_isch", "diabetes", "neo_breast",
                       "neuro_parkinsons"],
            "rei": ["drugs_alcohol", "drugs_alcohol", "drugs_alcohol",
                    "metab_bmi", "smoking_direct_prev"],
            "cause_id": [493, 495, 587, 429, 544],
            "rei_id": [102, 102, 102, 108, 166]
            })

    engine = db.db_engine("fbd-dev-write", database="forecasting")
    session = sessionmaker(bind=engine)()

    cause_risk_hierarchy_version_id = get_hierarchy_version_id(
        session, entity_type="cause_risk_pair",
        entity_set_id=CAUSE_RISK_SET_ID, gbd_round_id=gbd_round_id)

    maybe_negative_paf_set = get_strategy_set(
        session, strategy_id=CAUSE_RISK_MAYBE_NEGATIVE_PAF_SET_ID,
        hierarchy_id=cause_risk_hierarchy_version_id)

    # Set only has cause_ids and rei_ids, so get acauses
    acause_cause_id_map = _acauses(
        maybe_negative_paf_set["cause_id"].unique())
    maybe_negative_paf_set_with_acause = maybe_negative_paf_set.merge(
        acause_cause_id_map, how="left")

    # Ensure that all cause-ids have acauses
    acauses_missing = (
        maybe_negative_paf_set_with_acause["acause"].notnull().any())
    acause_err_msg = "Some causes don't have acauses"
    assert acauses_missing, acause_err_msg

    # ... and get reis.
    rei_rei_id_map = _reis(
        maybe_negative_paf_set_with_acause["rei_id"].unique())
    maybe_negative_paf_set_with_rei = (
        maybe_negative_paf_set_with_acause.merge(rei_rei_id_map, how="left"))

    # Ensure that all rei-ids have reis
    reis_missing = maybe_negative_paf_set_with_rei["rei"].notnull().any()
    rei_err_msg = "Some reis don't have reis"
    assert reis_missing, rei_err_msg

    session.close()
    engine.dispose()

    return maybe_negative_paf_set_with_rei


def _acauses(cause_ids):
    """Returns pandas.DataFrame that maps acauses to cause_ids, for the list of
    cause-ids given"""
    acause_query = (
        "SELECT ******, ****** "
        "FROM ******** "
        "WHERE ********* IN %(cause_ids)s"
        )
    params = dict(cause_ids=tuple(map(int, cause_ids)))
    LOGGER.debug("Getting acauses from database.")
    return db.query(
        acause_query, "fbd-dev-read", params=params, database="shared")


def _reis(rei_ids):
    """Returns pandas.DataFrame that maps reis to rei_ids, for the list of
    rei-ids given"""
    rei_query = (
        "SELECT ****, ***** "
        "FROM ******* "
        "WHERE ****** IN %(rei_ids)s"
        )
    params = dict(rei_ids=tuple(map(int, rei_ids)))
    LOGGER.debug("Getting acauses from database.")
    return db.query(
        rei_query, "fbd-dev-read", params=params, database="shared")


def get_acause_related_risks(acause, gbd_round_id):
    """
    Return a list of risks contributing to certain acause, specific for the
    scalars pipeline.

    Args:
        acause (str): analytical cause.
        gbd_round_id (int): gbd round id.

    Returns:
        list: a list of risks associated with this acause, ignoring the ones
            that are in_scalar = 0.
    """
    if acause in ['rotavirus']:
        risks = ['rota']
    else:
        df_acause_risk = _get_cause_risk_pairs(gbd_round_id)
        risks =\
            list(df_acause_risk.query("acause == @acause")["rei"].unique())
    return risks


def get_modeling_causes(gbd_round_id):
    '''Return the causes we are modeling

    Args:
        gbd_round_id (int): gbd round id.

    Returns:
        list: list of all acauses for the scalars pipeline.
    '''
    df_acause_risk = _get_cause_risk_pairs(gbd_round_id)
    acauses = list(df_acause_risk.acause.unique())
    return acauses


@lru_cache(1)
def get_vaccine_reis(gbd_round_id):
    """Returns the list of risks that are interventions, e.g. vaccines, such as
    dtp3.

    Args:
        gbd_round_id (int):
            Numeric ID for the GBD round.
    Returns:
        tuple:
            The reis of the risks that are interventions.
    """
    if gbd_round_id == 4:
        return "dtp3", "measles", "rota", "pcv", "hib"

    engine = db.db_engine("fbd-dev-read", database="forecasting")
    session = sessionmaker(bind=engine)()

    rei_hierarchy_version_id = get_hierarchy_version_id(
        session,
        entity_type="risk",
        entity_set_id=REI_SET_ID,
        gbd_round_id=gbd_round_id)

    return tuple(
        get_strategy_set(session, strategy_id=REI_INTERVENTION_ID,
                         hierarchy_id=rei_hierarchy_version_id)["rei"])


@lru_cache(1)
def get_risk_hierarchy(gbd_round_id):
    """
    A method for pulling the risk table using fbd_core.strategy_set.

    Args:
        gbd_round_id (int): gbd round id.

    Returns:
        pd.DataFrame: all metadata pertaining to risk hierarchy.
    """
    engine = db.db_engine("fbd-dev-read", database="forecasting")
    session = sessionmaker(bind=engine)()

    if gbd_round_id == 4:
        rei_set_id = 2  # GBD Computation set
        rei_hierarchy_version_id = get_hierarchy_version_id(
            session, entity_type="risk", entity_set_id=rei_set_id,
            gbd_round_id=gbd_round_id)
        params = dict(version=rei_hierarchy_version_id)

        q = (f"SELECT * FROM **************** "
             f"WHERE *********** = %(version)s")
        risk_hierarchy = db.query(q, "fbd-dev-read", params, database="shared")
    else:
        rei_hierarchy_version_id = get_hierarchy_version_id(
                session, entity_type="risk", entity_set_id=REI_SET_ID,
                gbd_round_id=gbd_round_id)
        risk_hierarchy = get_hierarchy(
                session, entity_type="risk",
                hierarchy_version_id=rei_hierarchy_version_id)

    session.close()
    engine.dispose()

    return risk_hierarchy


def save_past_and_future(da, gbd_round_id, stage, version, file_name,
                         year_args, sub_dir=None):
    """
    Given dataarray, and relevant FBDPath metadata, saves file as .nc in both
    "past" and "future".

    Args:
        da (xr.DataArray): dataarray to be split and saved separately.
        gbd_round_id (int): gbd round id.
        stage (str): FBDPath stage.
        version (str): FBDPath version.
        file_name (str): name of file, without extension.
        year_args (iterable): [past_start, forecast_start, forecast_end] years.
        sub_dir (str, optional): sub-directory after FBDPath
    """
    for i, past_or_future in enumerate(["past", "future"]):
        out_fbd_path = FBDPath(gbd_round_id=gbd_round_id,
                               past_or_future=past_or_future,
                               stage=stage,
                               version=version)
        if sub_dir:
            out_root = str(out_fbd_path / sub_dir)
        else:
            out_root = str(out_fbd_path)

        if not os.path.exists(os.path.dirname(out_root)):
            os.makedirs(os.path.dirname(out_root))

        outpath = out_root + "/" + (file_name + ".nc")
        out = da.loc[{YEAR_DIM: range(year_args[i], year_args[i+1])}]
        out.to_netcdf(outpath)


def conditionally_triggered_transformations(da, gbd_round_id, years):
    """
    Here encodes dataarray transformations that are triggered if any of the
    following conditions are satisfied:

    1.) da only has sex_id = 3 (both sexes).  In this case, expand to
        sex_id = 1, 2.
    2.) da onyl has age_group_id = 22 (all ages).  In this case, expand to
        all age_group_ids of current gbd_round
    3.) da has different years than expected.  In this case, filter/interpolate
        for just DEMOGRPHY_INDICES[YEAR_DIM].
    4.) da has only location_id = 1 (all locations).  In this case, we replace
        location_id=1 with the latest gbd location ids.
    5.) da has a point dimension of "quantile".  Simply remove.

    Args:
        da (xr.DataArray): may or may not become transformed.
        gbd_round_id (int): gbd round id.
        years (YearRange): [past_start, forecast_start, forecast_end] years.

    Returns:
        (xr.DataArray): transformed datarray, or not.
    """
    if SEX_DIM in da.dims and len(da[SEX_DIM]) == 1 and\
       da[SEX_DIM] == 3:
        # some vaccine SEVs could be modeled this way
        da = _expand_point_dim_to_all(da, gbd_round_id, years, dim=SEX_DIM)

    if AGE_DIM in da.dims and len(da[AGE_DIM]) == 1 and\
       da[AGE_DIM] == 22:
        # if da has a few age groups but not all, no transformation here
        da = _expand_point_dim_to_all(da, gbd_round_id, years, dim=AGE_DIM)

    if LOCATION_DIM in da.dims and len(da[LOCATION_DIM]) == 1 and\
       da[LOCATION_DIM] == 1:
        da = _expand_point_dim_to_all(da, gbd_round_id, years,
                                      dim=LOCATION_DIM)

    # some times we're provided with not enough or too many years.
    # here we first construct a superset, then crop out unwanted years.
    if YEAR_DIM in da.dims and\
       set(da[YEAR_DIM].values) !=\
       set(demographic_coords(gbd_round_id, years)[YEAR_DIM]):
        missing_years =\
            list(set(demographic_coords(gbd_round_id, years)[YEAR_DIM]) -
                 set(da[YEAR_DIM].values))
        if missing_years:
            da = _fill_missing_years_with_mean(da, YEAR_DIM, missing_years)
        # Now I filter for only the ones I need
        da = da.loc[{YEAR_DIM:
                     demographic_coords(gbd_round_id, years)[YEAR_DIM]}]

    # we don't need the the quantile point coordinate
    if QUANTILE_DIM in da.dims and len(da[QUANTILE_DIM]) == 1:
        da = da.drop(QUANTILE_DIM)

    return da


def _fill_missing_years_with_mean(da, year_dim, missing_years):
    """
    Some times the dataarray is missing some year_ids.
    This method fills those years with the mean of other years.

    Args:
        da (xr.DataArray): should contain the year_dim dimension.
        year_dim (str): str name of the year dimension.
        missing_years (list): list of years that are missing.

    Returns:
        (xr.DataArray): dataarray with missing years filled in.
    """
    year_coords = xr.DataArray([np.nan] * len(missing_years),
                               [(year_dim, missing_years)])
    mean_vals = da.mean(year_dim)  # NOTE just fill with mean values
    fill_da = mean_vals.combine_first(year_coords)
    da = da.combine_first(fill_da)
    return da.sortby(year_dim)


def _expand_point_dim_to_all(da, gbd_round_id, years, dim):
    """
    Some times upstream data has only age_group_id = 22 (all ages),
    or sex_id = 3 (both sexes).
    In that case, we'll need to convert it to the full age group dim,
    or sex_id = 1,2.  Such information is stored in self._expected_dims

    Args:
        da (xr.DataArray): da to expand.
        gbd_round_id (int): gbd round id.
        years (YearRange): [past_start, forecast_start, forecast_end] years.
        dim (str): Point dimension of da to expand on.  Typically this
            occurs with age_group_id = 2 (all ages) or sex_id = 3.

    Returns:
        (xr.Dataarray): has the dim coordinates prescribed by
            self.gbd_round_id
    """
    assert dim in da.dims, "{} is not an index dimension".format(dim)
    assert len(da[dim]) == 1, "Dim {} length not equal to 1".format(dim)
    # first I need the proper coordinates
    coords = demographic_coords(gbd_round_id, years)[dim]
    # make a dataarray with values nan and coordinates
    da_coords = xr.DataArray([np.nan] * len(coords), [(dim, coords)])
    # now some xarray magic: first do a combine_first
    out = da.combine_first(da_coords)
    # now fill the NaNs with what's in the original da
    out = out.fillna(da.loc[{dim: da[dim].values[0]}])
    # now get rid of original point
    out = out.loc[{dim: coords}]
    return out


def data_dims_check(da, gbd_round_id, years, exceptions=None):
    """
    Some checks on dataarray shape.

    Args:
        da (xr.DataArray): data in dataarray format.
        gbd_round_id (int): gbd round id.
        years (YearRange): [past_start, forecast_start, forecast_end] years.
        exceptions (str/iterable): str/iterable of dims where one can
            expect some missing coordinates.
            Example: "age_group_id" or ["year_id", "location_id"]
    """
    if exceptions is None:
        exceptions = []
    elif type(exceptions) is str:
            exceptions = [exceptions]
    if not isinstance(exceptions, collections.Iterable):
        raise TypeError("exceptions must be str/iterable. "
                        "Got {}".format(type(exceptions)))

    for dim, coords in demographic_coords(gbd_round_id, years).items():
        if dim in exceptions:  # skip checking this dim
            continue

        assert dim in da.coords, "{} dim not in data".format(dim)

        if not len(da[dim]) == len(coords):
            raise ValueError("{} should have this many coordinate labels: "
                             "{}.  Got {}".
                             format(dim, len(coords), len(da[dim])))

        if not (da[dim] == coords).all():
            raise ValueError("For {}, expect coords: {}.  Got {}".
                             format(dim, coords, da[dim].values))


def data_value_check(da):
    """
    Checks sensibility of dataarray values:
    1.) No NaN/Inf
    2.) No negative values

    Args:
        da (xr.DataArray): data in dataarray format.
    """
    if not xr.ufuncs.isfinite(da).all():
        raise ValueError("Array contains non-finite values")
    if not (da >= 0.0).all():
        raise ValueError("Array contains values < 0")

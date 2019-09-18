r"""
Given an acause, this script computes and exports all the cause-risk-pair PAFs
for this single cause.  That involves:

1) Finding all risks associated with said acause
2) Pulling SEVs and cause-risk-specific RRs from upstreams
3) Compute PAF as ``paf = 1 - 1 / (sev * (rr - 1) + 1)``

About the two upstreams:

1) SEV.  There's a separate directory for non-vaccine SEV and for vaccine SEV.
2) RR.  Like SEV, this is further divided into a non-vaccine and vaccine.

Example call (pulling gbd paf from get_draws):

.. code:: bash

    python calculate_paf.py --acause cvd_ihd --rei metab_bmi --version test \
        --directly-modeled-paf 20190419_dm_pafs \
        --sev 20180412_trunc_widerbounds --rrmax 20180407_paf1_update \
        --vaccine-sev 20180319_new_sdi --vaccine-rrmax 20171205_refresh \
        --gbd-round-id 4 --years 1990:2017:2040 --draws 100

If there's already a cleaned version of gbd cause-risk PAFs stored,
(Ex: 20180521_2016_gbd), one may access it via the --gbd-paf-version flag to
bypass using get_draws():

.. code:: bash

    python calculate_paf.py --acause cvd_ihd --rei metab_bmi --version test \
        --directly-modeled-paf 20190419_dm_pafs \
        --sev 20180412_trunc_widerbounds --rrmax 20180407_paf1_update \
        --vaccine-sev 20180319_new_sdi --vaccine-rrmax 20171205_refresh \
        --gbd-paf-version 20180521_2016_gbd \
        --gbd-round-id 4 --years 1990:2017:2040 --draws 100

Note that the 'draws' input arg is not only required, it also entails
up/down-sampling if any of the upstream files have number of draws not equal
to 'draws'.
"""
import gc
import logging
import os

from frozendict import frozendict
import numpy as np
from scipy.special import logit, expit
import xarray as xr

from get_draws.api import get_draws

from fbd_core import argparse
from fbd_core.etl import df_to_xr, resample
from fbd_core.db import get_cause_id, get_rei_id, get_gbd_round
from fbd_core.file_interface import (
    FBDPath, open_xr, save_xr, symlink_file_to_directory)

from fbd_research.scalars.utils import (
    conditionally_triggered_transformations,
    get_acause_related_risks,
    get_vaccine_reis,
    is_directly_modeled,
    is_maybe_negative_paf)


CAUSE_DIM = "cause_id"
CAUSES_NOT_IN_GBD_MAP = frozendict({"diarrhea_rotavirus": "diarrhea",
                                    "diarrhea_non_rotavirus": "diarrhea"})
PAF_LOWER_BOUND = -0.999999  # six 9's according to central comp
PAF_UPPER_BOUND = 0.999999


LOGGER = logging.getLogger(__name__)


def read_sev(rei, sev, vaccine_sev, gbd_round_id, years, draws):
    """
    Reads in SEV for vaccine.

    Args:
        rei (str): risk, could also be vaccine intervention.
        gbd_round_id (int): gbd round id
        sev (str): upstrem sev version
        vaccine_sev (str): upstream vaccine sev version.
        gbd_round_id (int): gbd round id.
        years (YearRange): [past_start, forecast_start, forecast_end] years.
        draws (int): number of draws for output file.  This means input files
            will be up/down-sampled to meet this criterion.

    Returns:
        (xr.DataArray): SEV in dataarray form.
    """
    if rei in get_vaccine_reis(gbd_round_id):  # vaccine treated as anti-risk
        infile_fbd_path =\
            FBDPath(gbd_round_id=gbd_round_id,
                    past_or_future="future",
                    stage="vaccine",
                    version=vaccine_sev) / (rei + "_new_ref.nc")
        out = 1.0 - open_xr(infile_fbd_path).data  # anti-risk
    else:
        infile_fbd_path =\
            FBDPath(gbd_round_id=gbd_round_id,
                    past_or_future="future",
                    stage="sev",
                    version=sev) / (rei + ".nc")
        out = open_xr(infile_fbd_path).data

    out = conditionally_triggered_transformations(out, gbd_round_id, years)
    if len(out["draw"]) != draws:
        out = resample(out, draws)
    return out


def read_rrmax(acause, rei, rrmax, vaccine_rrmax, gbd_round_id, years, draws):
    """
    Reads in RRmax for vaccine.

    Args:
        acause (str): analytical cause.
        rei (str): risk, could also be vaccine intervention.
        gbd_round_id (int): gbd round id
        rrmax (str): upstream rrmax version
        vaccine_rrmax (str): upstream vaccine rrmax version.
        gbd_round_id (int): gbd round id.
        years (YearRange): [past_start, forecast_start, forecast_end] years.
        draws (int): number of draws for output file.  This means input files
            will be up/down-sampled to meet this criterion.

    Returns:
        (xr.DataArray): vaccine RRmax in dataarray form.
    """
    if rei in get_vaccine_reis(gbd_round_id):
        # The values stored in these data files are actually not RR, but rather
        # r = Incidence[infection | vax] / Incidence[infection | no vax],
        # interpreted as "percent reduction of diseased cases if vaccinated",
        # and should be r < 1.
        # We compute the actual RR as 1/r.
        infile_fbd_path =\
            FBDPath(gbd_round_id=gbd_round_id,
                    past_or_future="future",
                    stage="rrmax",
                    version=vaccine_rrmax) / (rei + ".nc")
    else:
        infile_fbd_path =\
            FBDPath(gbd_round_id=gbd_round_id,
                    past_or_future="past",
                    stage="rrmax",
                    version=rrmax) / "netcdf" / (rei + ".nc")

    cause_id = get_cause_id(acause)

    out = open_xr(infile_fbd_path).data

    if cause_id not in out[CAUSE_DIM].values.tolist():
        error_message = "{} ({}) not in {}'s cause dim: {}".\
                        format(acause, cause_id, infile_fbd_path,
                               out[CAUSE_DIM].values.tolist())
        LOGGER.error(error_message)
        raise KeyError(error_message)
    out = out.loc[{CAUSE_DIM: cause_id}].drop(CAUSE_DIM)
    out = conditionally_triggered_transformations(out, gbd_round_id, years)

    if rei in get_vaccine_reis(gbd_round_id):
        # NOTE if we switch raw data source to burdenator, this algo might
        # need to change.
        # As mentioned above, this value for vaccine should be < 1.
        # Any value > 1 should be capped.
        out = out.where(out <= PAF_UPPER_BOUND).fillna(PAF_UPPER_BOUND)
        out = 1.0 / out  # as mentioned earlier, we compute RR as 1/r.

    if len(out["draw"]) != draws:
        out = resample(out, draws)

    # NOTE some rrmax cell values could be 0, for reasons unclear.
    return out


def get_gbd_paf(acause, rei, cache_version, gbd_round_id, sex_ids,
                location_ids, draws, measure_id=4, metric_id=2):
    """
    Downloads and transforms gbd cause-risk-specific PAF.  The dataarray
    is then cleaned and saved in a FBDPath.

    The gbd paf coming from get_draws::
        >>> df.columns
        Index([u'rei_id', u'modelable_entity_id', u'location_id', u'year_id',
               u'age_group_id', u'sex_id', u'cause_id', u'measure_id',
               u'draw_0', u'draw_1', ... u'draw_991', u'draw_992', u'draw_993',
               u'draw_994', u'draw_995', u'draw_996', u'draw_997', u'draw_998',
               u'draw_999', u'metric_id'], dtype='object', length=1009)

    where we will need to
    1.) use cause_id to slice for the cause-risk pair
    2.) use measure_id (typically 4 for yll) to slice for measure_id
    3.) use metric_id (typically 2 for percent) to slice for metric_id

    Args:
        acause (str): analytical cause.
        rei (str): risk, could also be vaccine intervention.
        cache_version (str): the FBDPath paf version to save the gbd paf in,
            or to read from.
        gbd_round_id (int): gbd round id
        sex_ids (list): sexes.  Typically [1, 2].
        location_ids (list): locations to get pafs from.
        draws (int): number of draws for output file.  This means input files
            will be up/down-sampled to meet this criterion.
        measure_id (int, optional): typically the yll measure id (4).  At the
            most detailed PAF yll is equivalent to death, so measure_id 4 works
            the same as measure_id 1 (death).  Empirically, it seems to pull
            data faster if calling with meausre_id=4.
        metric_id (int, optional): typically the percent metric (2)

    Returns:
        (xr.DataArray/None): Dataarray with complete demographic indices,
            sans "scenario".
    """
    if rei in get_vaccine_reis(gbd_round_id):
        # get_draws won't have anything for vaccines
        return None

    cache_file_fbdpath =\
        FBDPath(gbd_round_id=gbd_round_id,
                past_or_future="past",
                stage="paf",
                version=cache_version) / (acause + "_" + rei + ".nc")

    if cache_file_fbdpath.exists():

        LOGGER.info("{} already exists.  Will read from it for gbd paf.".
                    format(cache_file_fbdpath))

        paf_da = open_xr(cache_file_fbdpath).data

        paf_da = paf_da.sel(location_id=location_ids)

        if len(paf_da["draw"]) != draws:
            paf_da = resample(paf_da, draws)

        return paf_da

    else:  # no cache exists, must download & clean
        rei_id = get_rei_id(rei)

        if acause in CAUSES_NOT_IN_GBD_MAP:  # edge case for diarrhea_*
            cause_id = get_cause_id(CAUSES_NOT_IN_GBD_MAP[acause])
        else:
            cause_id = get_cause_id(acause)

        gbd_round = get_gbd_round(gbd_round_id)

        try:
            # we only need it for year_id=gbd_round, but for every other dim
            # we collect everything.
            paf_df = get_draws(gbd_id_type=['cause_id', 'rei_id'],
                               gbd_id=[cause_id, rei_id],
                               source='burdenator',
                               year_id=gbd_round,
                               gbd_round_id=gbd_round_id,
                               measure_id=measure_id,
                               metric_id=metric_id)
        except Exception as exc:
            error_message = "Error in get_draws for {}_{}".format(acause, rei)
            LOGGER.error(error_message)
            raise IOError(str(exc))

        paf_df = paf_df.drop(columns=["year_id",
                                      "rei_id",
                                      "cause_id",
                                      "measure_id",
                                      "metric_id"])  # don't need these no more

        paf_da = df_to_xr(paf_df,
                          dims=["location_id", "age_group_id", "sex_id"],
                          wide_dim_name='draw',
                          wide_dim_transform=lambda x: int(x.split('_')[1]),
                          fill_value=np.nan)

        paf_da = paf_da.sortby("draw")  # draws don't always come in sorted

        paf_da = _data_cleaning_for_paf(paf_da, acause, rei, "GBD")

        LOGGER.info("Saving downloaded & cleaned {}".
                    format(cache_file_fbdpath))

        save_xr(paf_da, cache_file_fbdpath, metric="percent", space="identity",
                cause_id=cause_id, rei_id=rei_id, gbd_round_id=gbd_round_id,
                year_id=gbd_round, measure_id=measure_id, metric_id=metric_id,
                upper_bound=PAF_UPPER_BOUND, lower_bound=PAF_LOWER_BOUND)

    if len(paf_da["draw"]) != draws:
        paf_da = resample(paf_da, draws)

    return paf_da


def _data_cleaning_for_paf(paf, maybe_negative_paf=False):
    """
    Encodes data cleaning customized for PAF:

    1.) Leave NaNs as NaNs, we assume they are age restrictions.
    2.) set > UPPER_BOUND to UPPER_BOUND
    3.) set < LOWER_BOUND to LOWER_BOUND for protective PAFs and set
        < 1 - PAF_UPPER_BOUND for non-protective PAFs.

    Non-finite PAF values likely come from outer-join mismatches between
    sev and rr, and we set those to 0
    for (2) and (3), per discussion with central comp, PAF values over
    boundaries are simply capped, not resampled.

    Args:
        paf (xr.DataArray): dataarray of PAF values
        maybe_negative_paf (bool, optional): ``True`` is PAF is allowed to be
            negative. Defaults to ``False``.

    Returns:
        (xr.DataArray):
            cleaned dataarray.
    """
    if maybe_negative_paf:
        lower_bound = PAF_LOWER_BOUND
    else:
        lower_bound = 1 - PAF_UPPER_BOUND

    return paf.clip(min=lower_bound, max=PAF_UPPER_BOUND)


def compute_correction_factor(fhs_paf, gbd_paf, maybe_negative_paf=False):
    r"""
    Forecasted PAF is bias-corrected by GBD PAF.  This is essentially
    an "intercept-shift", and it happens at the last year of past (gbd round),
    and hence the input args should be single-year arrays.

    Even though PAF values should logically be in the closed interval
    :math:`[-1, 1]`, we expect both ``fhs_paf`` and ``gbd_paf`` to be in the
    open interval :math:`(-1, 1)` due to upstream data cleaning. Furthermore,
    most cause-risk pairs are *not* protective (i.e. non-negative), so are
    actually expected to be in the open interval :math:`(0, 1)`.

    This method computes correction factor in logit space by transforming
    the PAF values via :math:`x_{\text{corrected}} = (1 + x) / 2`, and with the
    correction factor being the difference between fhs and gbd in logit space:

    .. math::

        \text{correction-factor} =
            \text{logit}(\frac{1 + \mbox{PAF}_{\text{gbd}}}{2})
            - \text{logit}(\frac{1 + \mbox{PAF}_{\text{fhs}}}{2})

    For cause-risk pairs that *cannot* have protective (i.e. negative PAFs),
    the correction factor equation is:

    .. math::

        \text{correction-factor} =
            \text{logit}(\mbox{PAF}_{\text{gbd}})
            - \text{logit}(\mbox{PAF}_{\text{fhs}})

    This correction factor will later be added to the forecasted PAF values
    in logit space, prior to back-transformation.

    By default, non-finite correction factor values are reset to 0.  These
    non-finite values could come from mismatched cells from outer-join
    arithmetic, commonly found along the ``age_group_id`` dimension between
    GBD and FHS (np.nan)

    Args:
        fhs_paf (xr.DataArray): forecasted PAF at gbd round year.
        gbd_paf (xr.DataArray): gbd PAF.  Only contains the gbd round year.
        maybe_negative_paf (bool, optional): ``True`` is PAF is allowed to be
            negative. Defaults to ``False``.

    Returns:
        (xr.DataArray):
            correction factor.
    """
    # first make sure the input args are year-agnostic
    if "year_id" in fhs_paf.coords and fhs_paf["year_id"].size > 1:
        raise ValueError("fhs_paf has year dim larger than size=1")

    if "year_id" in gbd_paf.coords and gbd_paf["year_id"].size > 1:
        raise ValueError("gbd_paf has year dim larger than size=1")

    # There still might be mismatches in other dims (age_group_id, etc.),
    # due to age or sex restrictions. So we outer-join our arithmetic here and
    # Leave the NaNs as NaNs, so at the very last step when we still have NaNs
    # we assume they're due to restriction and thus should just be PAF=0.
    with xr.set_options(arithmetic_join="outer"):
        if maybe_negative_paf:
            correction_factor = (
                logit((1 + gbd_paf) / 2) - logit((1 + fhs_paf) / 2)
                )
        else:
            correction_factor = logit(gbd_paf) - logit(fhs_paf)

    if "year_id" in correction_factor.coords:  # want to be year-agnostic
        if "year_id" in correction_factor.dims:  # "year_id" still an index dim
            correction_factor =\
                correction_factor.sel(year_id=correction_factor["year_id"].
                                      values[0])  # "year_id" no longer index
        correction_factor = correction_factor.drop("year_id")

    return correction_factor


def correct_paf(fhs_paf, cf, maybe_negative_paf=False):
    r"""
    The forecasted PAF is corrected in logit space and back-transformed:

    .. math::

        \mbox{PAF}_{\text{corrected}} = 2 * \text{expit}(\text{logit}
            (\frac{1 + \mbox{PAF}_{\text{FHS}}}{2} + \text{correction-factor}
            ) - 1

    for cause-risk pairs that *can* be protective. If cause-risk pairs are not
    allowed to protective (the majority of them), then the equation for the
    corrected PAF is

    .. math::

        \mbox{PAF}_{\text{corrected}} =
            \mbox{PAF}_{\text{FHS}} + \text{correction-factor}

    Because a logit function's argument x is :math:`[0, 1]` and a protective
    PAF can be in the range :math:`[-1, 1]`, a natural mapping from PAF space
    to logit space is :math:`x_{\text{corrected}} = (1 + x) / 2`. The
    back-transform to PAF space is hence :math:`2 * expit( logit(x) ) + 1`.

    The correction factor ``cf`` is also computed within the same logit space.
    Once the correction is made in logit space, the resultant quantity
    is mapped back to PAF-space via the aforementioned back-transform.

    Args:
        fhs_paf (xr.DataArray): forecasted PAF.  Has many years along year_id.
        cf (xr.DataArray): correction factor, should not have year_id dim.
        maybe_negative_paf (bool, optional): ``True`` is PAF is allowed to be
            negative. Defaults to ``False``.

    Returns:
        (xr.DataArray):
            correct PAF.
    """
    if maybe_negative_paf:
        corrected_paf = 2 * expit(logit((1 + fhs_paf) / 2) + cf) - 1
    else:
        corrected_paf = expit(logit(fhs_paf) + cf)

    return corrected_paf


def compute_paf(acause, rei, version, years, gbd_round_id, draws,
                sev, rrmax, vaccine_sev, vaccine_rrmax, gbd_paf_version,
                **kwargs):
    r"""
    Computes PAF for the given acause-risk pair, and exports said PAF to
    ``/{gbd_round_id}/{past_or_future}/paf/{version}``.

    Args:
        acause (str): analytical cause.
        rei (str): rei, or commonly called risk.
        version (str): FBDPath version to export to.
        years (YearRange): [past_start, forecast_start, forecast_end] years.
        gbd_round_id (int): gbd round id.
        draws (int): number of draws for output file.  This means input files
            will be up/down-sampled to meet this criterion.
        sev (str): upstream sev version
        rrmax (str): upstream rrmax version
        vaccine_sev (str): upstream vaccine sev version
        vaccine_rrmax (str): upstream vaccine rrmax version
        gbd_paf_version (str): gbd_paf version to read from,
            if not downloading from get_draws().
    """
    sev_da = read_sev(rei=rei, sev=sev, vaccine_sev=vaccine_sev,
                      gbd_round_id=gbd_round_id, years=years, draws=draws)
    rrmax_da = read_rrmax(acause=acause, rei=rei, rrmax=rrmax,
                          vaccine_rrmax=vaccine_rrmax,
                          gbd_round_id=gbd_round_id, years=years, draws=draws)

    # estimated cause-risk-specific paf
    with xr.set_options(arithmetic_join="outer"):
        paf = 1 - 1 / (sev_da * (rrmax_da - 1) + 1)

    location_ids = sev_da["location_id"].values.tolist()
    sex_ids = sev_da["sex_id"].values.tolist()

    del sev_da, rrmax_da
    gc.collect()

    maybe_negative_paf = is_maybe_negative_paf(acause, rei, gbd_round_id)

    # Forecasted PAFs are cleaned first before further processing
    paf = _data_cleaning_for_paf(paf, maybe_negative_paf)

    # now ping get_draws for gbd paf values
    LOGGER.info("Got estimated paf for {}_{}.  Pulling gbd paf...".
                format(acause, rei))

    gbd_round = get_gbd_round(gbd_round_id)

    if gbd_paf_version:  # then we read gbd_paf from this folder
        cache_version = gbd_paf_version
    else:  # default to {gbd_round}_gbd
        cache_version = str(gbd_round) + "_gbd"

    gbd_paf = get_gbd_paf(acause, rei, cache_version, gbd_round_id,
                          sex_ids=sex_ids, location_ids=location_ids,
                          draws=draws)

    LOGGER.info("Pulled gbd paf for {}_{}.  Computing adjusted paf...".
                format(acause, rei))

    # compute correction factor and perform adjustment
    if gbd_paf is not None:

        # First make sure there's no COMPLETE mismatch between paf and gbd_paf.
        # If so, an error should be raised
        paf.load()
        gbd_paf.load()  # need to force load() because dask is lazy
        if (paf - gbd_paf).size == 0:  # normal arithmetic is inner-join
            error_message = ("Complete mismatch between computed and GBD in "
                             "{}-{} PAF.  Are you sure you used the correct "
                             "version of GBD PAF?".format(acause, rei))
            LOGGER.error(error_message)
            raise ValueError(error_message)

        gbd_paf = _data_cleaning_for_paf(gbd_paf, maybe_negative_paf)

        correction_factor = compute_correction_factor(
            paf.sel(year_id=gbd_round), gbd_paf, maybe_negative_paf)

        del gbd_paf
        gc.collect()

        paf = correct_paf(paf, correction_factor, maybe_negative_paf)

        LOGGER.info("Adjusted paf for {}_{}.  Now saving...".
                    format(acause, rei))
    else:  # correction factor is 0, and we leave paf as is
        correction_factor = xr.zeros_like(paf)
        LOGGER.info("paf for {}_{} not adjusted because gbd_paf is None".
                    format(acause, rei))

    # If there are still NaNs at this point, then they should indicate age or
    # sex restrictions and should be filled with 0.
    paf = paf.fillna(0)

    # we need to save the results separately in "past" and "future"
    for p_or_f, yrs in {"past": years.past_years,
                        "future": years.forecast_years}.items():

        out = paf.sel(year_id=yrs)
        out_fbd_path = FBDPath(gbd_round_id=gbd_round_id,
                               past_or_future=p_or_f,
                               stage="paf",
                               version=version)

        # first we save cause-risk-specific paf
        outpath = (out_fbd_path / "risk_acause_specific" /
                   (acause + "_" + rei + ".nc"))

        LOGGER.info("Saving {}".format(outpath))
        save_xr(out, outpath, metric="percent", space="identity",
                acause=acause, risk=rei, gbd_round_id=gbd_round_id,
                sev=sev, rrmax=rrmax, vaccine_sev=vaccine_sev,
                vaccine_rrmax=vaccine_rrmax, gbd_paf_version=cache_version)

        del out
        gc.collect()

        # now saving cause-risk-specific correction factor
        if p_or_f == "past":
            outpath = (out_fbd_path / "risk_acause_specific" /
                       (acause + "_" + rei + "_cf.nc"))

            LOGGER.info("Saving {}".format(outpath))
            save_xr(correction_factor, outpath, metric="percent",
                    space="logit", sev=sev, rrmax=rrmax,
                    vaccine_sev=vaccine_sev, vaccine_rrmax=vaccine_rrmax,
                    gbd_paf_version=cache_version)

    del paf, correction_factor
    gc.collect()


def symlink_directly_modeled_paf_file(
        acause, rei, calculated_paf_version, directly_modeled_paf, gbd_round_id
        ):
    """Creates symlink to files with directly-modeled PAF data.

    Creates symlinks of past and future directly-modeled PAF data files to the
    directory with PAFs calculated from SEVs and RRmaxes.

    Args:
        acause (str):
            Indicates the cause of the cause-risk pair
        rei (str):
            Indicates the risk of the cause-risk pair
        calculated_paf_version (str):
            Output version of this script where directly-modeled PAFs are
            symlinked, and calculated PAFs are saved.
        directly_modeled_paf (str):
            The version of PAFs with the directly-modeled PAF to be symlinked
            resides.
        gbd_round_id (int):
            The numeric ID representing the GBD round.

    Raises:
        RuntimeError:
            If symlink sub-process fails.
    """
    for p_or_f in ("past", "future"):
        calculated_paf_dir = FBDPath(
            gbd_round_id=gbd_round_id,
            past_or_future=p_or_f,
            stage="paf",
            version=calculated_paf_version) / "risk_acause_specific"
        calculated_paf_dir.mkdir(parents=True, exist_ok=True)

        directly_modeled_paf_file = (
                FBDPath(
                    gbd_round_id=gbd_round_id,
                    past_or_future=p_or_f,
                    stage="paf",
                    version=directly_modeled_paf)
                / "risk_acause_specific" / f"{acause}_{rei}.nc")

        symlink_file_to_directory(
            directly_modeled_paf_file, calculated_paf_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate cause-risk PAFs")
    parser.add_argument("--acause", type=str, required=True,
                        help="Analytical cause")
    parser.add_argument("--rei", type=str, required=True, help="Risk")
    parser.add_argument("--version", type=str, required=True,
                        help="String denoting file directory. Ex: 20171212_ha")
    parser.add_argument('--directly-modeled-paf', type=str,
                        help=('Version of directly-modeled PAFs to copy over.'
                              'Note that currently the code expects the '
                              'future and past versions to have the same '
                              'name'),
                        required=True)
    parser.add_argument('--sev', type=str,
                        help='sev upstream version, in /future/sev/',
                        required=True)
    parser.add_argument('--rrmax', type=str,
                        help='rrmax upstream version, in /past/rrmax/',
                        required=True)
    parser.add_argument('--vaccine-sev', type=str,
                        help=('vaccine sev upstream version, '
                              'in /future/vaccine/'),
                        required=True)
    parser.add_argument('--vaccine-rrmax', type=str,
                        help=('vaccine rrmax upstream version, '
                              'in /future/rrmax/'),
                        required=True)
    parser.add_argument("--gbd-round-id", type=int, required=True)
    parser.add_arg_years(required=True)
    parser.add_argument("--draws", type=int, required=True,
                        help=("Number of draws for output file. "
                              "This means input files will be up/down-sampled "
                              "to meet this criterion.  It's up to the user "
                              "to quality-check the upstream files to ensure "
                              "data expectations are met."))
    parser.add_argument("--gbd-paf-version", type=str,
                        help=("stored gbd paf version to read from in "
                              " /past/paf.  If not provided, "
                              " defaults to downloading using get_draws()"),
                        required=False, default=None)

    args = parser.parse_args()

    # if this is part of an array job, then SGE_TASK_ID would exist in the
    # environment, which specifies the rei arg
    task_id = os.environ.get("SGE_TASK_ID")
    if task_id:
        risks = get_acause_related_risks(args.acause, args.gbd_round_id)
        idx = int(task_id) - 1
        args.rei = risks[idx]

    LOGGER.debug("arguments {}".format(args))

    if not is_directly_modeled(args.acause, args.rei, args.gbd_round_id):
        LOGGER.info(
            f"Calculating {args.acause}-{args.rei} PAF from SEV and RRmax")
        compute_paf(**args.__dict__)
    else:
        LOGGER.info(
            f"Symlinking {args.acause}-{args.rei} (directly-modeled) PAF")
        symlink_directly_modeled_paf_file(
            args.acause, args.rei, args.version, args.directly_modeled_paf,
            args.gbd_round_id)

    LOGGER.debug("exit from calculate_pafs")

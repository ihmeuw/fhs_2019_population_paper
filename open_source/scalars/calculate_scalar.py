r"""
This script calculates aggregated acause specific PAFs and scalars.

Outputs:

1) Risk-acause specific scalar.  Exported to the upstream paf/{version}
2) Acause specific scalars.  Exported to scalar/{version}

Example call:

.. code:: bash

    python calculate_scalar.py --acause whooping --version 20180321_arc_log \
    --gbd-round-id 4 --years 1990:2017:2040
"""
from collections import defaultdict
from functools import lru_cache
import gc
import logging
import os

from frozendict import frozendict
import numpy as np
import pandas as pd
import xarray as xr

from fbd_core import argparse
from fbd_core.db import get_gbd_round
from fbd_core.file_interface import FBDPath, save_xr, open_xr
from fbd_research.scalars.utils import (get_acause_related_risks,
                                        get_risk_hierarchy, data_value_check)

INPATH_MEDIATION = ("{med_root_dir}/mediation_matrix_draw_gbd_{gbd_year}.csv")

# NOTE the gbd 2016 mediation file uses metab_fpg for both _categ and _cont,
# and acause_csa for both _female and _male.  This would change in gbd 2017.
MEDIATION_FILE_SPECIAL_RISK_MAP = frozendict({"metab_fpg_cont": "metab_fpg",
                                              "metab_fpg_categ": "metab_fpg",
                                              "abuse_csa_female": "abuse_csa",
                                              "abuse_csa_male": "abuse_csa"})


LOGGER = logging.getLogger(__name__)


def read_paf(acause, risk, gbd_round_id, past_or_future, version):
    """
    Read past or forecast PAF.

    Args:
        acause (str): cause name.
        risk (str): risk name.
        gbd_round_id (int): gbd round id.
        past_or_future (str): "past" or "forecast".
        version (str): str indiciating folder where data comes from.

    Returns
        paf (xr.DataArray): dataframe of PAF.

    Raises:
        ValueError: if upstream flat file does not exist.
    """
    infile_fbd_path = (FBDPath(gbd_round_id=gbd_round_id,
                               past_or_future=past_or_future,
                               stage="paf",
                               version=version) /
                       "risk_acause_specific" /
                       "{}_{}.nc".format(acause, risk))
    paf = open_xr(infile_fbd_path).data

    return paf


@lru_cache(1)
def _mediation(mediation_file_path, gbd_round_id):
    """
    LRU cache of the mediation file, to avoid repeatedly opening it.
    Creates a "mean" column over the draws if it doesn't have it already.
    """
    gbd_year = get_gbd_round(gbd_round_id)
    med = pd.read_csv(mediation_file_path.format(gbd_year=gbd_year))
    # we use the mean of the draws as mediation factor
    if "mean" not in med.columns:
        draw_cols = [col for col in med.columns if "draw" in col]
        med["mean"] = med[draw_cols].mean(axis=1)
    return med


def product_of_mediation(acause, risk, mediators, gbd_round_id):
    r"""
    Returns :math:`\prod_{i} (1 - MF_{jic})`, where j is the risk whose
    adjusted PAF to acause c we're considering, and i is the mediator.

    That means this method takes in only 1 risk, and a list of multiple
    mediators, and performs a product over all the :math:`(1 - MF_{jic})`'s.

    "metab_bmi" impacts "cvd_ihd" via "metab_sbp".  Say sbp is the only
    mediator of bmi.  So if PAF_{ihd/bmi} is 0.9 and MF_{sbp/bmi} is 0.6,
    then the adjusted PAF_{ihd/bmi} is 0.9 * (1 - 0.6) = 0.32,
    because 0.54 of the 0.9 comes from sbp.

    The mediation factor for (acause, risk) is provided in a flat file.

    Args:
        acause (str): analytical cause.
        risk (str): risk related to acause.
        mediators (list[str]): the risks that could potentially sit between
            risk and the acause in attribution.  Usually these
            are just all the risks that are paired with acause, and then
            we filter through the mediation file for the ones that matter.
        gbd_round_id (int): gbd round id.

    Returns:
        mediation_products: float, mediation products for (acause, risk).
    """
    # there should be only 1 risk, and potentially multiple mediators
    if type(risk) is not str:
        error_message = "risk should be str, got {}".format(type(risk))
        LOGGER.error(error_message)
        raise TypeError(error_message)
    if type(mediators) is not list:
        error_message =\
            "mediators should be list, got {}".format(type(mediators))
        LOGGER.error(error_message)
        raise TypeError(error_message)

    LOGGER.info("Computing product of mediation of risk {} via mediators {} "
                "on acause {}".format(risk, mediators, acause))

    med_risks = list(mediators)  # just making a copy for later .remove()
    rei = risk  # making copy for possible overwrite later
    # NOTE the list of mediators can include "risk" itself.  Here we remove
    # risk from the list of mediators
    if rei in med_risks:
        med_risks.remove(rei)

    # here we take care of some edge cases for metab_fpg_* and abuse_csa_*
    if rei in MEDIATION_FILE_SPECIAL_RISK_MAP:
        rei = MEDIATION_FILE_SPECIAL_RISK_MAP[rei]

    med_risks = [MEDIATION_FILE_SPECIAL_RISK_MAP[x]
                 if x in MEDIATION_FILE_SPECIAL_RISK_MAP else x
                 for x in med_risks]

    med = _mediation(INPATH_MEDIATION, gbd_round_id)
    # med is a df of columns (omitting some):
    # rei           acause          med_            mean
    # abuse_csa     mental_alcohol  drugs_alcohol   0.9999
    # activity      cvd_ihd         metab_fpg       0.1439286
    # diet_fiber    cvd_ihd         diet_fruit      0.9999

    # "med_" is the risk sitting between a cause and a lower-level risk
    mediation_factors =\
        med.query("acause == @acause and med_ in @med_risks and "
                  "rei == @rei")["mean"].values

    return np.prod(1 - mediation_factors)  # == 1 if mediation_factors is empty


def get_risk_id_dict(risk_table):
    ''' Return dict of (id, risk) for all risks in risk table.
        Example: {104: 'metab', 202: '_env', 203: '_behav'}

        Parameters
        ----------
        risk_table: dataframe with columns of rei, rei_id,
                    path_to_top_parent, level
    '''
    risk_id_dict = dict(risk_table[['rei_id', 'rei']].values)
    return risk_id_dict


def get_cluster_risks(contributing_risks, risk_id_dict, risk_table):
    """
    Take the risks associated with a cause (contributing_risks),
    and make a dict of all the parent risks of these risks,
    with a list of their sub-risks (within contributing_risks) as value.
    So you end up with keys that are risks,
    many outside of contributing_risks,
    and values that are subsets of contributing_risks.

    Returns a dictionary. key: parent risk; value: list of sub-risks.

    Args:
        contributing_risks (list): list of risks contributing to 'cause'.
        risk_id_dict (dict): dictionary of risk_id, risk.
        risk_table (pandas.DataFrame): dataframe with columns of rei,
            rei_id, path_to_top_parent, level.

    Returns:
        (defaultdict): a dict where keys are risks, and values are sub-risks.
            Ex: {'_env': ['air_hap', 'wash', ...],
                 'metab': ['metab_fpg', 'metab_bmi', ...],
                 '_behav': ['activity', 'nutrition_child', ...]}

    Raises:
        TypeError: if path_to_top_parent for a particular risk is not recorded
            as a string type.  That would most likely be error on the db.
    """
    subrisk_dict = defaultdict(list)

    for risk in contributing_risks:
        risk_specific_metadata = risk_table.query("rei == @risk")

        if not risk_specific_metadata.empty:
            # comma-delimited string, like "169,202,82,83", or None
            path_to_top_parent = risk_specific_metadata[
                "path_to_top_parent"].item()
        else:
            # If there is no metadata for the risk, then
            # return an empty dict.
            return subrisk_dict

        # NOTE the vaccines (hib, pcv, rota, measles, dtp3) currently have
        # "None" listed in their path_to_top_parent.
        if path_to_top_parent:  # not None, must be string
            if type(path_to_top_parent) is not str:
                raise TypeError("{} is not of str type".
                                format(path_to_top_parent))
            # first of list is "_all", last of list is itself
            _all_to_self_list = path_to_top_parent.split(",")

            for parent_id in _all_to_self_list[:-1]:  # keep _all, ignore self
                parent_risk = risk_id_dict[int(parent_id)]
                subrisk_dict[parent_risk].append(risk)

    return subrisk_dict


def aggregate_paf(acause, cause_risks, gbd_round_id, past_or_future, version,
                  cluster_risk=None):
    """
    Aggregate PAFs through mediation.

    Args:
        acause (str): acause
        cause_risks (list[str]): risks associated with acause
        gbd_round_id (int): gbd round id
        past_or_future (str): 'past' or 'future'
        version (str): indicating folder where data comes from/goes to
        cluster_risk: whether this is a cluster risk.
                Impacts how it's saved.

    Returns:
        paf_aggregated (pandas.DataFrame): dataframe of aggregated PAF.
    """
    LOGGER.info('Start aggregating {} PAF:'.format(past_or_future))

    LOGGER.info("Acause: {}, Risks: {}, Cluster Risk: {}".
                format(acause, cause_risks, cluster_risk))
    if len(cause_risks) == 0:
        error_message = ("0 risks for acause {}".format(acause))
        LOGGER.error(error_message)
        raise ValueError(error_message)

    for index, risk in enumerate(cause_risks):
        # if cluster_risk is specified, cause_risks is just the subset of this
        # cause's risks that fall under cluster_risk.
        # If cluster_risk is not specified, that cause_risks are all the risks
        # associated with this cause.
        LOGGER.info('Doing risk: {}'.format(risk))

        paf = read_paf(acause, risk, gbd_round_id, past_or_future, version)
        # mediation_prod is just a float
        mediation_prod = product_of_mediation(acause, risk, cause_risks,
                                              gbd_round_id)
        med_product = paf * mediation_prod

        del paf
        del mediation_prod
        gc.collect()

        if index == 0:
            LOGGER.debug("Index is 0. Starting the paf_prod.")
            paf_prod = 1.0 - med_product
        else:
            LOGGER.debug("Index is {}.".format(index))
            # NOTE there no straight-forward way to check if two dataarrays
            # have the same coordinates, so we broadcast indiscriminantly
            paf_prod, one_minus_med =\
                xr.broadcast(paf_prod, 1.0 - med_product)

            paf_prod = paf_prod.where(xr.ufuncs.isfinite(paf_prod)).fillna(1)
            one_minus_med = one_minus_med.\
                where(xr.ufuncs.isfinite(one_minus_med)).fillna(1)
            paf_prod = paf_prod * one_minus_med
            del one_minus_med

        del med_product
        gc.collect()

    LOGGER.info("We got some pafs.")
    paf_prod = 1.0 - paf_prod
    paf_aggregated = paf_prod

    save_paf(paf_aggregated, gbd_round_id, past_or_future, version, acause,
             cluster_risk=cluster_risk)
    if cluster_risk:
        return None
    else:
        return paf_aggregated


def save_paf(paf, gbd_round_id, past_or_future, version, acause,
             cluster_risk=None):
    """
    Save mediated PAF at cause level.

    Args:
        paf (pandas.DataFrame): dataframe of PAF.
        gbd_round_id (int): gbd round id.
        past_or_future (str): 'past' or 'future'.
        version (str): version, dated.
        acause (str): analytical cause.
        cluster_risk (str, optional): if none, it will be just risk.
    """
    if cluster_risk is not None:
        out_fbd_path = (FBDPath(gbd_round_id=gbd_round_id,
                                past_or_future=past_or_future,
                                stage="paf",
                                version=version) /
                        "risk_acause_specific" /
                        "{}_{}.nc".format(acause, cluster_risk))

        LOGGER.info("Saving cause-agg risk paf: {}".format(out_fbd_path))
        save_xr(paf, out_fbd_path, metric="percent", space="identity",
                acause=acause, risk=cluster_risk, version=version,
                gbd_round_id=gbd_round_id)
    else:
        out_fbd_path = (FBDPath(gbd_round_id=gbd_round_id,
                                past_or_future=past_or_future,
                                stage="paf",
                                version=version) /
                        "{}.nc".format(acause))

        LOGGER.info("Saving cause-only paf: {}".format(out_fbd_path))
        save_xr(paf, out_fbd_path, metric="percent", space="identity",
                acause=acause, version=version, gbd_round_id=gbd_round_id)


def compute_scalar(acause, version, gbd_round_id, no_update_past, **kwargs):
    """
    Computes and saves scalars for acause, given upstream paf version.

    Args:
        acause (str): cause to compute scalars for
        version (str): date/version string pointing to folder to pull data from
        gbd_round_id (int): gbd round id.
        no_update_past (boolean): whether to overwrite past scalars.
    """
    risk_table = get_risk_hierarchy(gbd_round_id)
    risk_id_dict = get_risk_id_dict(risk_table)  # {id: risk}

    cause_risks = get_acause_related_risks(acause, gbd_round_id)  # list of risks

    for past_or_future in ['past', 'future']:
        LOGGER.info("OH BOY WE'RE DOING THE: {}".format(past_or_future))
        outpath_scalar =\
            FBDPath(gbd_round_id=gbd_round_id,
                    past_or_future=past_or_future,
                    stage="scalar",
                    version=version) / ("{}.nc".format(acause))

        if os.path.exists(str(outpath_scalar)) and no_update_past:
            continue

        # Aggregate PAF for level-1 cluster risks
        # We don't need to use the PAF for scalar.

        # take the risks associated with the cause (cause_risks),
        # and make a dict of all the level 1, 2, 3 parent risks of these risks,
        # with list of their sub-risks (within cause_risks) as value.
        # So you end up with keys that may have risks outside of cause_risks,
        # and values that are subsets of cause_risks.
        risk_lst = get_cluster_risks(cause_risks, risk_id_dict, risk_table)

        for key in risk_lst.keys():  # loop over all antecedent-risks
            LOGGER.info("Looping over super/parent risks.")
            subrisks = risk_lst[key]

            if len(subrisks) > 0:
                LOGGER.info('Start aggregating cluster risk: {}'.format(key))
                aggregate_paf(acause, subrisks, gbd_round_id, past_or_future,
                              version, cluster_risk=key)
                gc.collect()

        # Aggregate PAF for all risks.
        # We need to use the PAF for scalar.
        paf_mediated = aggregate_paf(acause, cause_risks, gbd_round_id,
                                     past_or_future, version)
        if paf_mediated is None:
            LOGGER.info("No paf_mediated. Early return.")
            return

        scalar = 1.0 / (1.0 - paf_mediated)

        del paf_mediated
        gc.collect()

        LOGGER.debug("Checking data value for {} scalar".format(acause))
        data_value_check(scalar)  # make sure no NaNs or <0 in dataarray

        save_xr(scalar, outpath_scalar, metric="number", space="identity",
                acause=acause, version=version, gbd_round_id=gbd_round_id,
                no_update_past=str(no_update_past))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Scalars")
    parser.add_argument("--acause", type=str, required=True,
                        help="Analytical cause")
    parser.add_argument("--version", type=str, required=True,
                        help="Version of the upstream PAF FBDPath")
    parser.add_argument("--gbd-round-id", type=int, required=True)
    parser.add_arg_years(required=True)  # for run_scalars.run_script()
    parser.add_argument('--no-update-past',
                        help=('Whether to not update past values. '
                              'By default we always update past values.'),
                        action='store_true', required=False, default=False)

    args = parser.parse_args()

    LOGGER.debug("Arguments {}".format(args))

    compute_scalar(**args.__dict__)

    LOGGER.debug("Exit from calculate_scalars")

"""
Models of life expectancy take mortality rate as input and return
a dataset of life expectancy, mortality rate, and mean age of death.
They are models because they have to estimate the mean age of death.

There are two kinds of models here, those for GBD 2015 and those
for GBD 2016. They differ by what the input and output age group IDs are.

For GBD 2015, the input age groups start with IDs (2, 3, 4) for ENN,
PNN, and LNN. They end with IDs (20, 21) for 75-79 and 80+.

For GBD 2016, the input age groups start with IDs (2, 3, 4) for ENN,
PNN, and LNN. They end with IDs 235 for 95+.

GBD 2017 uses the same age group IDs as 2016.

For both 2015/2016, we predict life expectancy on young ages (28, 5), meaning
0-1 and 1-5. For older ages, we predict to age ID 148 for 110+.

For 2017, we take forecasted mx straight up to compute the the life table.
That is, no extrapolation/extension is done to ameliorate the older ages.
"""
from __future__ import division
from collections import OrderedDict
import gc
import logging
import warnings

import numpy as np
import xarray as xr

from fbd_core.demog import nx_from_age_group_ids, consistent_age_group_ids
from fbd_core.etl import expand_dimensions
import fbd_core.demog
import fbd_core.demog.lifetable
import fbd_core.db


LOGGER = logging.getLogger("fbd_research.lex.model")


def without_point_coordinates(ds):
    """
    Remove point coordinates and return them, so you can add them back later.
    The code would look like this::

       no_point, saved_coords = without_point_coordinates(ds)
       # Do things
       return results.assign_coords(**saved_coords)

    Args:
        ds (xr.DataArray or xr.Dataset): A dataarray that may have point coords

    Returns:
        (xr.DataArray, dict): The point coordinates are copied and returned.
    """
    point = dict((pname, ds.coords[pname].values.copy())
                 for pname in ds.coords
                 if ds.coords[pname].shape == ())
    return ds.drop(list(point)), point


def combine_age_ranges(young, middle, old):
    """
    Three Dataarrays have different age groups in them.
    Combine them to make one dataarray. There may be overlap
    among age_group IDs, so anything in the middle
    is overwritten by the young or old. The dims and coords in the middle
    are used to determine the dims and coords in
    output. Point coordinates are kept.
    This works even when young, middle, and old have different age intervals,
    for instance [28, 5] versus [2, 3, 4].

    Args:
        young (xr.DataArray): fit for young ages, or None if there is no fit
        middle (xr.DataArray): middle age groups
        old (xr.DataArray): fit for old ages

    Returns:
        xr.DataArray: With dims in the same order as the middle.
    """
    if young is not None:
        with_young = fbd_core.demog.age_id_older_than(
            young.age_group_id[-1].values, middle.age_group_id.values)
        young_ages = young.age_group_id.values
    else:
        with_young = middle.age_group_id.values
        young_ages = "none"
    # The age IDs are just IDs, and younger ages may have larger IDs, so sort.
    edge_ages = fbd_core.demog.age_id_older_than(
        old.age_group_id[0].values, with_young, True)
    LOGGER.debug("combine_age_ranges young {} middle {} old {} keep {}".format(
        young_ages, middle.age_group_id.values,
        old.age_group_id.values, edge_ages
    ))
    mid_cut = middle.loc[{"age_group_id": edge_ages}]
    if young is not None:
        return xr.concat([young, mid_cut, old], dim="age_group_id")
    else:
        return xr.concat([mid_cut, old], dim="age_group_id")


def append_nLx_lx(ds):
    """
    Adds :math:`{}_nL_x` and :math:`l_x` to the dataset.
    It's sometimes called :math:`{}_nU_x`.

    Arguments:
        ds (xr.Dataset): Dataset containing mx and ax

    Returns:
        The same dataset with nLx and lx appended.
    """
    nx = fbd_core.demog.nx_from_age_group_ids(ds.mx.age_group_id)
    nLx = fbd_core.demog.fm_person_years(ds.mx, ds.ax, nx)
    lx, dx = fbd_core.demog.fm_population(ds.mx, ds.ax, nx, 1e5)
    return ds.assign(nLx=nLx, lx=lx)


def generic_to_gbd(mx, gbd_round, generic):
    """
    Some of the fitting methods don't care what age intervals mx uses,
    so this function adapts any such fitting method to work on
    the GBD age groups and apply both the young ages and old ages fits.

    Args:
        mx (xr.DataArray): Mortality rate without point coordinates
        gbd_round (int): 3 for gbd 2015, 4 for gbd 2016
        generic (callable): This is the function to run

    Returns:
        xr.Dataset: With ex, mx, ax.
    """
    nx_base = fbd_core.demog.nx_contiguous_round(gbd_round, "baseline")
    nx_life = fbd_core.demog.nx_contiguous_round(gbd_round, "lifetable")
    assert (np.array_equal(nx_base.age_group_id, mx.age_group_id) or
            np.array_equal(nx_life.age_group_id, mx.age_group_id)),\
        "Age groups are {}".format(mx.age_group_id.values)

    ax = generic(mx, nx_base)

    under_one_mx = make_under_one_group_for_preston(mx)
    under_one_ax = preston_ax_fit(under_one_mx)
    assert np.array_equal(under_one_mx.age_group_id,
                          under_one_ax.age_group_id),\
        "Age groups are {} and {}".format(
        under_one_mx.age_group_id.values,
        under_one_ax.age_group_id.values)

    if gbd_round is 3:
        old_mx, old_ax = fbd_core.demog.old_age_fit_k2(mx, ax)

    elif gbd_round is 4:
        qx = fbd_core.demog.fm_mortality(mx, ax, nx_base)
        qxf, axf, mxf = fbd_core.demog.old_age_fit_us_counties(qx)
        # This goes past the 95+ limit, so reduce it.
        mxp, axp = condense_ages_to_terminal(qxf, axf, 33, 235)
        v = mxf.age_group_id.values
        term = np.argwhere(v == 33)[0][0]
        old_mx = xr.concat(
            [
                mxf.loc[dict(age_group_id=v[:term])],
                mxp
            ],
            dim="age_group_id"
        )
        old_ax = xr.concat(
            [
                axf.loc[dict(age_group_id=v[:term])],
                axp
            ],
            dim="age_group_id"
        )
    else:
        LOGGER.error("No old age fit for gbd round {}".format(gbd_round))
        raise Exception(
            "Asked for an old age fit for round {}".format(gbd_round))

    mx_combined = combine_age_ranges(under_one_mx, mx, old_mx)
    ax_combined = combine_age_ranges(under_one_ax, ax, old_ax)
    assert np.array_equal(nx_life.age_group_id, mx_combined.age_group_id), \
        ("Mismatch ages lifetable {} mx {}".format(
                nx_life.age_group_id.values, mx_combined.age_group_id.values))

    ex = fbd_core.demog.fm_period_life_expectancy(
            mx_combined, ax_combined, nx_life)
    ds = xr.Dataset(dict(ex=ex, mx=mx_combined, ax=ax_combined))
    return append_nLx_lx(ds)


def gbd3_piecewise(mx):
    """Use piecewise-constant approximation over all age groups."""
    return generic_to_gbd(mx, 3, fbd_core.demog.cm_mean_age)


def gbd4_piecewise(mx):
    """Use piecewise-constant approximation over all age groups."""
    return generic_to_gbd(mx, 4, fbd_core.demog.cm_mean_age)


def gbd3_udd(mx):
    """Use uniform distribution of deaths approximation over all age groups."""
    return generic_to_gbd(mx, 3, fbd_core.demog.ud_mean_age)


def gbd4_udd(mx):
    """Use uniform distribution of deaths approximation over all age groups."""
    return generic_to_gbd(mx, 4, fbd_core.demog.ud_mean_age)


# This is the fit from Demography by Preston, Heuveline, and Guillot.
PHG_CUT = xr.DataArray([0.107, 0.107, 0.107],
                       coords=dict(sex_id=[2, 1, 3]),
                       dims=["sex_id"])
PHG = xr.DataArray([
    [[[0.35, 0],
      [0.33, 0],
      [0.34, 0]],
     [[0.053, 2.8],
      [0.045, 2.684],
      [0.049, 2.742]]],
    [[[1.361, 0],
      [1.352, 0],
      [1.357, 0]],
     [[1.522, -1.518],
      [1.651, -2.816],
      [1.587, -2.167]]]
], coords={"age_group_id": [28, 5], "domain": ["above", "below"],
           "sex_id": [2, 1, 3], "const": ["c", "m"]},
    dims=("age_group_id", "domain", "sex_id", "const"))


def preston_ax_fit(mx):
    """
    This fit is from Preston, Heuveline, and Guillot. It comes from a fit
    that Coale-Demeney made in their life tables for :math:`({}_1a_0, {}_4a_1)`
    from `{}_1q_0`. PHG turned it into a fit from :math:`m_x` to :math:`a_x`.
    This is explored in a notebook in docs to ``fbd_core.demog``.

    Args:
        mx (XArray.DataArray): Mortality rate. Must have age group IDs (28,5).

    Returns:
        XArray.DataArray: Mean age, for only those age groups where predicted.
    """
    LOGGER.info("Calculating Preston mx fit.")
    under_five = [28, 5]
    assert all(ax in mx.age_group_id.values for ax in under_five), \
        "preston_mx_fit requires input with age ID (28,5) {}".format(
            mx.age_group_id.values)
    sexes = mx.sex_id
    msub = mx.loc[dict(age_group_id=under_five)]
    msub.where(msub < 0.107).fillna(0.107)
    ax_fit = (msub * PHG.loc[dict(domain="below", const="m", sex_id=sexes)] +
              PHG.loc[dict(domain="below", const="c", sex_id=sexes)])
    LOGGER.debug("fit coords are {}".format(ax_fit.coords))
    return ax_fit.drop(["domain"])


def mx_on_new_ages(mx, forecast_age_groups):
    """
    Given an :math:`m_x` defined on age gropus that include
    age ids 2, 3, and 4, representing ENN, PNN, and LNN, transform
    it into an :math:`{}_1m_0`. If the forecast age groups add or modify
    old ages, then leave these blank. Just copy what can be copied.
    That makes sense because we usually fill in old ages with a
    regression.

    Args:
        mx (xr.DataArray): Mortality rate with age groups (2, 3, 4)
        forecast_age_groups: A list of old age group ids. You can find these
                             ids in the ``shared`` databases.

    Returns:
        xr.DataArray: Mortality rate copied onto the age groups (28 and up).
        xr.DataArray: Age intervals, :math:`n_x` for those age groups.
    """
    under_five = [2, 3, 4]
    assert all(ax in mx.age_group_id.values for ax in under_five), \
        "mx_on_new_ages requires input with age ID (2, 3, 4) {}".format(
            mx.age_group_id.values)

    nx = fbd_core.demog.nx_from_age_group_ids(mx.age_group_id)
    nxp = nx_from_age_group_ids(forecast_age_groups)

    # Make a new data array with desired age groups.
    coordp = OrderedDict()
    for coord in mx.dims:
        if "age_group_id" != coord:
            if mx.coords[coord].shape:
                coordp[coord] = mx.coords[coord].values.copy()
            else:
                pass  # singleton cause
        else:
            coordp[coord] = nxp.age_group_id.values.copy()
    mxp = xr.DataArray(
        np.zeros(shape=[len(c) for c in coordp.values()], dtype=np.float),
        coords=coordp, dims=coordp.keys())
    common = list(set(mx.age_group_id.values) & set(mxp.age_group_id.values))
    mxp.loc[dict(age_group_id=common)] = mx.loc[dict(age_group_id=common)]

    # Average to get mx for youngest 0-1 year from first week, month, rest
    # of year.
    mxp.loc[dict(age_group_id=28)] = ((
        mx.loc[dict(age_group_id=2)] * nx.loc[dict(age_group_id=2)] +
        mx.loc[dict(age_group_id=3)] * nx.loc[dict(age_group_id=3)] +
        mx.loc[dict(age_group_id=4)] * nx.loc[dict(age_group_id=4)])
        / nxp.loc[dict(age_group_id=28)])
    return mxp, nxp


def gbd3_graduated(mx):
    """
    This function represents our best current guess for a way to calculate
    life expectancy that will match what is done for GBD 2015.
    This means input age groups include ENN, PNN, and LNN, and the terminal
    age group is 80+.

    Args:
        mx (xr.DataArray): Mortality rate

    Returns:
        xr.DataArray: Period life expectancy.
    """
    LOGGER.info("Using single_gbd2015")
    nx = fbd_core.demog.nx_from_age_group_ids(mx.age_group_id)
    forecast_age_groups = ([28, 5] + list(range(6, 21)) + list(range(30, 34))
                           + [44, 45, 148])
    assert consistent_age_group_ids(forecast_age_groups), \
        "The age groups aren't self-consistent"

    mxp, nxp = mx_on_new_ages(mx, forecast_age_groups)
    axp = fbd_core.demog.cm_mean_age(mxp, nxp)
    youth_ax = preston_ax_fit(mxp)
    axp.loc[dict(age_group_id=youth_ax.age_group_id)] = youth_ax

    # Graduation applies only where it's 5-year age groups.
    # These are the 5-year age groups that are nonzero, the ones coming in.
    middle_ages = nx.where(nx == nx.median()).dropna(dim="age_group_id")
    graduation_ages = middle_ages.age_group_id.values
    # But we do the graduation on the new set of age groups to do it in place.
    axp = fbd_core.demog.ax_graduation_cubic(mxp, axp, nxp, graduation_ages)
    # The K2 fit is to the original intervals.
    ax = fbd_core.demog.cm_mean_age(mx, nx)
    mxpp, axpp = fbd_core.demog.old_age_fit_k2(mx, ax)
    LOGGER.debug("mxpp has {} mx has {}".format(mxpp.age_group_id,
                                                mxp.age_group_id))
    shared_ages = list(set(mxp.age_group_id.values) &
                       set(mxpp.age_group_id.values))
    end_ages = list(set(mxpp.age_group_id.values) -
                    set(mxp.age_group_id.values))

    shared_ages.sort()
    end_ages.sort()
    LOGGER.debug("zero dimension {}".format(axp.coords))
    LOGGER.debug("axpp dimensions {}".format(axpp.coords))

    if shared_ages:
        LOGGER.debug("assign {}".format(shared_ages))
        axp.loc[dict(age_group_id=shared_ages)] = \
            axpp.loc[dict(age_group_id=shared_ages)]
        mxp.loc[dict(age_group_id=shared_ages)] = \
            mxpp.loc[dict(age_group_id=shared_ages)]
    if end_ages:
        LOGGER.debug("concat {}".format(end_ages))
        axp = xr.concat([axp, axpp.loc[dict(age_group_id=end_ages)]],
                        dim="age_group_id")
        mxp = xr.concat([mxp, mxpp.loc[dict(age_group_id=end_ages)]],
                        dim="age_group_id")
    nxp = fbd_core.demog.nx_from_age_group_ids(mxp.age_group_id)
    ex = fbd_core.demog.fm_period_life_expectancy(mxp, axp, nxp)
    assert consistent_age_group_ids(ex.age_group_id.values)
    ds = xr.Dataset(dict(ex=ex, mx=mxp, ax=axp))
    return append_nLx_lx(ds)


def us_counties_extrapolation_of_lx(mx, ax):
    r"""
    A backup method for extrapolating the lx for older age groups,
    based on :func:`fbd_core.demog.lifemodel.old_age_fit_us_counties`.

    Args:
        mx (xr.DataArray): mortality rate.

    Returns:
        (xr.DataArray): lx, with age group ids 33, 44, 45,
            and 148 extrapolated.
    """
    nx_base = fbd_core.demog.nx_from_age_group_ids(mx["age_group_id"])
    qxp = fbd_core.demog.fm_mortality(mx, ax, nx_base)
    qxf, axf, mxf = fbd_core.demog.old_age_fit_us_counties(qxp)
    mx_long = combine_age_ranges(None, mx, mxf)
    ax_long = combine_age_ranges(None, ax, axf)
    nx_long = fbd_core.demog.nx_from_age_group_ids(mx_long["age_group_id"])
    lx_long, dx_long =\
        fbd_core.demog.fm_population(mx_long, ax_long, nx_long, 1e5)
    if (lx_long > 1e3).any():
        lx_long = lx_long / 1e5
        dx_long = dx_long / 1e5
    assert tuple(lx_long['age_group_id'].values[-4:]) == (33, 44, 45, 148)
    return lx_long


def demography_team_extrapolation_of_lx(mx, ax):
    r"""
    Computes diff of logit-:math:`q_x` iteratively, starting from age 90-94
    (id 32), to compute the :math:`q_x` for 95-99 (id 33), 100-104 (id 44),
    105-109 (id 45).

    Starting with :math:`{}_5q_{90}`, one first computes

    .. math::
        \Delta_{90} = c_{90} + \beta_{90} + \beta\_logit\_q_{90} *
                      \text{logit}(q_{90})

    where :math:`c`, :math:`\beta`, and :math:`\beta\_logit\_q_x` are all
    regression constants available in
    :func:`fbd_core.demog.lifemodel.old_age_fit_qx_95_plus`.
    From then, for every sex, we have::

        for i in [95, 100]:

    .. math::
        q_i = \text{expit}( \text{logit}(q_{i-5}) + \Delta_{i-5} )

    .. math::
        \Delta_i = c_i + \beta_i + \beta\_logit\_q_{90} * \text{logit}(q_{90})

    Args:
        mx (xr.DataArray): mortality rate, with age_group_id dim.
        ax (xr.DataArray): ax.
        gbd_round_id (int): gbd round id.
        years (YearRange): years for past and future.

    Returns:
        (xr.DataArray): lx where age group id 235 is replaced with
            age group ids 33 (95-100), 44 (100-105), 45 (105-110),
            and 148 (110+, where lx is set to 0).
    """
    # here we compute qx from mx
    nx_base = fbd_core.demog.nx_from_age_group_ids(mx["age_group_id"])

    # make a baseline qx assuming constant mortality
    qx = fbd_core.demog.fm_mortality(mx, ax, nx_base)

    # replace age group id 235 of qx with (33, 44) via demog team extrapolation
    qx = fbd_core.demog.old_age_fit_qx_95_plus(qx)  # overwrite qx
    # make lx from qx
    lx = _qx_to_lx(qx)
    # Another adjustment based on self-consistency requirement
    qx = _self_consistent_qx_adjustment(qx, lx, mx)  # overwrite qx
    lx = _qx_to_lx(qx)  # make lx again, based on adjusted qx

    return lx


def _self_consistent_qx_adjustment(qx, lx, mx):
    r"""
    A universal relationship exists between :math:`l_x`, :math:`q_x`, and
    :math:`m_x` at the terminal age group (95+):

    .. math::
        {}_{\infty}m_{95} = \frac{l_{95}}{T_{95}}

    where :math:`T_{95} = \int_{95}^{\infty} l_x dx`.

    Because we forecast :math:`l_{95}` and :math:`m_{95}`, and that
    :math:`l_{x}` for :math:`x > 95` is extrapolated independently
    (via :func:`fbd_research.lex.model.demography_team_extrapolation_of_lx`),
    the above relationship does not hold.  We therefore need to adjust our
    values of :math:`_5l_{100}`, :math:`{}_5l_{105}` so that the relationship
    holds.

    Note that we set :math:`l_{110} = 0`.

    We begin with the approximation of :math:`T_{95}`,
    using Simpson's 3/8 rule:

    .. math::
        T_{95} = \int_{95}^{\infty} \ l_{x} dx \
               \approx \ \frac{3 \ n}{8}({}_5l_{95} + 3 {}_5l^{\prime}_{100} +
                                         3 {}_5l^{\prime}_{105} +
                                         3 {}_5l^{\prime}_{110}) \
               = \ T^{\prime}_{95}

    where :math:`n` is 5 years, the age group bin size, and :math:`{}^{\prime}`
    denotes the current tentative value.
    Also note that :math:`{}_5l_{110} = 0` in our case.

    The above formula allows us to define

    .. math::
        \alpha &= \frac{T_{95}}{T^{\prime}_{95}} \\
               &= \frac{\frac{l_{95}}{m_{95}}}{T^{\prime}_{95}}
        :label: 1

    as a "mismatch factor".

    We also declare that the ratio between :math:`{}_5q_{95}`
    and :math:`{}_5q_{100}` is fixed:

    .. math::
        \beta = \frac{{}_5q_{95}}{{}_5q_{100}}
              = \frac{{}_5q_{95}^{\prime}}{{}_5q_{100}^{\prime}}
              < 1
        :label: 2

    Hence we may proceed with the following derivation:

    .. math::
        {}_5l_{100} &= {}_5l_{95} \ (1 - {}_5q_{95}) \\
                    &= {}_5l_{95} \ (1 - \beta \ {}_5q_{100})
        :label: 3

    .. math::
        {}_5l_{105} &= {}_5l_{100} \ (1 - {}_5q_{100}) \\
                    &= {}_5l_{95} \ (1 - {}_5q_{95}) \ (1 - {}_5q_{100}) \\
                    &= {}_5l_{95} \ (1 - \beta \ {}_5q_{100}) \
                                    (1 - {}_5q_{100})
        :label: 4

    .. math::
        \alpha &\approx \frac{\frac{15}{8} \ ({}_5l_{95} + 3 \ {}_5l_{100} +
                        3 \ {}_5l_{105})}{\frac{15}{8} ( {}_5l_{95} +
                        3 \ {}_5l^{\prime}_{100} + 3 \ {}_5l^{\prime}_{105})}\\
               &= \frac{{}_5l_{95} + 3 \ {}_5l_{95} \ (1 - {}_5q_{95}) +
                  3 \ {}_5l_{95} \ (1 - {}_5q_{95})(1 - {}_5q_{100})}
                  {{}_5l_{95} + 3 \ {}_5l_{95}(1 - {}_5q^{\prime}_{95}) +
                   3 \ {}_5l_{95}(1 - {}_5q^{\prime}_{95})
                                 (1 - {}_5q^{\prime}_{100})} \\
               &= \frac{{}_5l_{95} + 3 \ {}_5l_{95} \ (1 - \beta \ {}_5q_{100})
                  + 3 \ {}_5l_{95} \ (1 - \beta \ {}_5q_{100})
                                     (1 - {}_5q_{100})}
                  {{}_5l_{95} +
                   3 \ {}_5l_{95} \ (1 - \beta \ {}_5q^{\prime}_{100}) +
                   3 \ {}_5l_{95} \ (1 - \beta \ {}_5q^{\prime}_{100})
                                 (1 - {}_5q^{\prime}_{100})} \\
               &= \frac{1 + 3 \ (1 - \beta \ {}_5q_{100}) +
                  3 \ (1 - \beta \ {}_5q_{100})(1 - {}_5q_{100})}
                  {1 + 3 \ (1 - \beta \ {}_5q^{\prime}_{100}) +
                   3 \ (1 - \beta {}_5q^{\prime}_{100})
                       (1 - {}_5q^{\prime}_{100})} \\
               &= \frac{4 - 3 \ \beta \ {}_5q_{100} + 3 - 3 \ {}_5q_{100} -
                  3 \ \beta \ {}_5q_{100} + 3 \ \beta \ {{}_5q_{100}}^2}
                  {4 - 3 \ \beta {}_5q^{\prime}_{100} + 3 -
                   3 \ {}_5q^{\prime}_{100} -
                   3 \ \beta \ {}_5q^{\prime}_{100} +
                   3 \ \beta \ {{}_5q^{\prime}_{100}}^2} \\
               &= \frac{\frac{7}{3} - (2 \ \beta + 1) \ {}_5q_{100} +
                  \beta \ {{}_5q_{100}}^2}{\frac{7}{3} -
                  (2 \ \beta + 1) \ {}_5q^{\prime}_{100} +
                  \beta \ {{}_5q^{\prime}_{100}}^2}
        :label: 5

    where the denominator is known.
    If we define

    .. math::
        \gamma = \frac{7}{3} - \alpha \ (\frac{7}{3} -
                 (2 \beta + 1) \ {}_5q^{\prime}_{100} +
                 \beta \ {{}_5q^{\prime}_{100}}^2)
        :label: 6

    then we have the quadratic equation

    .. math::
        \beta \ {{}_5q_{100}}^2 - (2 \beta + 1) \ {}_5q_{100} + \gamma = 0
        :label: 7

    with the solution

    .. math::
        {}_5q_{100} = \frac{(2 \ \beta + 1) \pm \sqrt{(2 \ \beta + 1)^2 -
                      4 \beta \ \gamma}}{2 \ \beta}
        :label: 8

    Because :math:`{}_5q_{100} \leq 1`, subtraction in the numerator of :eq:`8`
    is the only viable solution.

    Args:
        qx (xr.DataArray): qx that has age groups (..., 33, 44),
            with 33 (95-100) and 44 (100-105).  Should not have 235 (95+).
        lx (xr.DataArray): lx that has age groups (..., 33, 44, 45, 148),
            with 45 (105-110) and 148 (110+),
            where (lx.sel(age_group_id=148) == 0).all().
        mx (xr.DataArray): mx, needed to compute :math: `T_{95}`.
            Has age_group_id=235 (95+) instead of (33, 44, 45, 148).

    Returns:
        (xr.DataArray): qx where age groups 33 (95-100) and 44 (100-105) are
            "adjusted".
    """
    n = 5.0  # 5 year age group width
    # 33 = 95-100 yrs, 44 = 100-105 yrs, 45 = 105-110 yrs, 148 = 110+ yrs.
    T_95_prime =\
        lx.sel(age_group_id=33) * (3.0 / 8.0 * n) +\
        lx.sel(age_group_id=[44, 45]).sum("age_group_id") * (9.0 / 8.0 * n)
    alpha = (lx.sel(age_group_id=33) / mx.sel(age_group_id=235)) \
        / T_95_prime

    # these are the original, unadulterated q95 & q100
    q95_prime = qx.sel(age_group_id=33)
    q100_prime = qx.sel(age_group_id=44)
    beta = q95_prime / q100_prime

    gamma = 7.0 / 3.0 - alpha * (7.0 / 3.0 - (2 * beta + 1) * q100_prime +
                                 beta * (q100_prime ** 2))

    q100 = ((2 * beta + 1) -
            xr.ufuncs.sqrt((2 * beta + 1) ** 2 - 4 * beta * gamma)) /\
           (2 * beta)

    # unfortunately, ~20% of q100 adjusted via this approx will be > 1.
    # it's even possible to end up with q100 < 0.
    # it makes no sense to cap q100 to 1, because that means l105 == 0,
    # which we do not want.  The only option left is the following
    q100 = q100.where((q100 < 1) & (q100 > 0)).fillna(q100_prime)
    q95 = q100 * beta  # always <= 1 because beta is always <= 1

    # Update original qx with adjusted q95 and q100 values
    qx.loc[dict(age_group_id=33)] = q95
    qx.loc[dict(age_group_id=44)] = q100

    return qx


def _qx_to_lx(qx):
    r"""
    Computes :math:`l_x` based on :math:`q_x`, where :math:`q_x` already
    contains the 95-100 (33) and 100-105 (44) age groups.  Also computes
    :math:`l_x` for 105-110 (45), and then set :math:`l_x` for 110+ to be 0.

    Args:
        qx (xr.DataArray): Probability of dying.

    Returns:
        (xr.DataArray): lx.
    """
    if tuple(qx["age_group_id"].values[-2:]) != (33, 44):
        raise ValueError("qx must have age group ids 33 and 44")

    px = 1.0 - qx  # now we have survival all the way to 100-105 (44) age group

    # Because l{x+n} = lx * px, we can compute all lx's if we start with
    # l_0 = 1 and iteratively apply the px's of higher age groups.
    # So we compute l_105-110, since we have p_100-105 from extrapolated qx.
    # We start with a set of lx's that are all 1.0
    lx = xr.full_like(px, 1)
    # now expand lx to have age groups 105-110 (45)
    lx = expand_dimensions(lx, fill_value=1, age_group_id=[45])

    # Since l{x+n} = lx * px, we make cumulative prduct of px down age groups
    # and apply the product to ages[1:] (since ages[0]) has lx = 1.0
    ages = lx["age_group_id"]

    ppx = px.cumprod(dim="age_group_id")  # the cumulative product of px
    ppx.coords["age_group_id"] = ages[1:]  # need to correspond to ages[1:]
    lx.loc[dict(age_group_id=ages[1:])] *= ppx  # lx all the way to 100-105

    # now artificially sets lx to be 0 for the 110+ age group.
    lx = expand_dimensions(lx, fill_value=0, age_group_id=[148])

    assert (lx.sel(age_group_id=2) == 1).all()
    assert tuple(lx['age_group_id'].values[-4:]) == (33, 44, 45, 148),\
        "final lx should have age group ids 33, 44, 45, and 148."

    return lx


def under_5_ax_preston_rake(mx, ax, nx):
    r"""
    Uses :math:`m_x` and Preston's Table 3.3 to compute ax for FHS under-5
    age groups (enn, lnn, pnn, 1-4yrs).

    The steps are as follows:

    (1) Aggregate neonatal mx's to make :math:`{}_1m_0`, using the method
        outlined in
        :func:`fbd_research.lex.model.make_under_one_group_for_preston_with_nLx`.
        This assumes the nLx values are approximately correct.

    (2) Use Preston's Table 3.3 to compute :math:`{}_1a_0`
        using :math:`{}_1m_0`

    (3) "Rake" the *neonatal ax's such that they aggregate to :math:`{}_1a_0`.
        We start with the definition

        .. math::
            {}_1a_0 = \frac{a_2 \ d_2 + (a_3 + n_2) \ d_3 + (a_4 + n_2 + n_3)
                            \ d_4}{d_2 + d_3 + d_4}

        where subscripts 2, 3, and 4 denote enn, lnn, and pnn age groups.

        The above equation can be rewritten as

        .. math::
            {}_1a_0 - \frac{n_2 \ d_3 + (n_2 + n_3) \ d_4}{d_2 + d_3 + d_4} =
            \frac{a_2 \ d_2 + a_3 \ d_3 + a_4 \ d_4}{d_2 + d_3 + d_4}

        Because we have :math:`{}_1a_0` and assume that the dx values are
        approximately correct, the left-hand side is fixed at this point.
        The entire right-hand side, where the :math:`{}_na_x` values are,
        needs to be "raked" to satisfy the above equation.  We simply multiply
        all three :math:`{}_na_x` by the same ratio.

    Args:
        mx (xr.DataArray): forecated mortality rate.
        ax (xr.DataArray): ax assuming constant mortality for under 5 age
            groups.
        nx (xr.DataArray): nx for FHS age groups, in years.

    Returns:
        (xr.DataArray) ax, with under-5 age groups (2, 3, 4, 5) optimized.
    """
    assert {2, 3, 4, 5}.issubset(mx["age_group_id"].values)

    # first compute the approximately correct dx and nLx values.
    # they are "approximately" correct because they's not sensitive to ax.
    lx, dx = fbd_core.demog.fm_population(mx, ax, nx, 1.0)
    nLx = fbd_core.demog.fm_person_years(mx, ax, nx)

    under_1_dx_sum = dx.sel(age_group_id=[2, 3, 4]).sum("age_group_id")

    ax4 = ax.sel(age_group_id=5)  # the 1-4yr ax

    mx_u5 = make_under_one_group_for_preston_with_nLx(mx, nLx)  # 1m0 and 4m1
    # now compute the 1a0 and 4a1 from Preston's Table 3.3
    ax_u5 = preston_ax_fit(mx_u5)  # 2 age groups: 0-1 (28) and 1-5 (5)
    ax1_p = ax_u5.sel(age_group_id=28)  # Preston's 1a0
    ax4_p = ax_u5.sel(age_group_id=5)  # Preston's 4a1

    left_hand_side = ax1_p - \
        (nx.sel(age_group_id=2) * dx.sel(age_group_id=3) +
         nx.sel(age_group_id=[2, 3]).sum("age_group_id") *
         dx.sel(age_group_id=4)) / under_1_dx_sum

    right_hand_side =\
        (ax.sel(age_group_id=2) * dx.sel(age_group_id=2) +
         ax.sel(age_group_id=3) * dx.sel(age_group_id=3) +
         ax.sel(age_group_id=4) * dx.sel(age_group_id=4)) / under_1_dx_sum

    ax1_raking_const = left_hand_side / right_hand_side
    ax1_raking_const["age_group_id"] = 28

    ax4_ratio = ax4_p / ax4

    # now modify our under-5 ax values
    ax.loc[dict(age_group_id=[2, 3, 4])] =\
        ax.sel(age_group_id=[2, 3, 4]) * ax1_raking_const
    ax.loc[dict(age_group_id=5)] = ax.sel(age_group_id=5) * ax4_ratio

    return ax


def gbd5_no_old_age_fit(mx):
    """
    This is based on :func:`fbd_research.lex.model.gbd4_all_youth`,
    except that age groups [31, 32, 235] are not replaced with fitted values.

    Args:
        mx (xr.DataArray): Mortality rate

    Returns:
        xr.DataArray: Period life expectancy.
    """
    nx_gbd = fbd_core.demog.nx_contiguous_round(5, "baseline")
    # nx_gbd looks like array([1.917808e-02, 5.753425e-02, 9.232877e-01,
    # 4.000000e+00, 5.000000e+00, ... 5.000000e+00, 5.000000e+00, 4.500000e+01]
    try:
        mx = mx.sel(age_group_id=nx_gbd["age_group_id"].values)
    except KeyError:
        raise RuntimeError("Not all ages in incoming data "
                           "have {} want {}".format(mx.age_group_id.values,
                                                    nx_gbd.age_group_id.values
                                                    ))
    assert fbd_core.demog.consistent_age_group_ids(mx.age_group_id.values)
    nx_base = fbd_core.demog.nx_from_age_group_ids(mx.age_group_id)
    # >>> nx_base['age_group_id']
    # <xarray.DataArray 'age_group_id' (age_group_id: 21)>
    # array([ 28,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,
    #         17,  18,  19,  20,  30,  31,  32, 235])
    # 28 is 1 year wide, 5 is 4 years wide

    # We want a baseline on nulls. If there is one null in a draw, we
    # null out the whole draw, so this algorithm may increase null
    # count, but it won't increase the bounding box on nulls.
    mx_null_cnt = mx.where(mx.isnull(), drop=True).size
    if mx_null_cnt > 0:
        warnings.warn("Incoming mx has {} nulls".format(mx_null_cnt))

    # first compute ax assuming constant mortality over interval
    ax = fbd_core.demog.cm_mean_age(mx, nx_base)

    # Graduation applies only where it's 5-year age groups.
    middle_ages = nx_base.where(nx_base == nx_base.median()).dropna(
            dim="age_group_id")
    graduation_ages = middle_ages.age_group_id.values
    # graduation ages are the age group ids that have 5-year neighbors:
    # array([ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30,
    #         31, 32])
    # Note that 6 and 32 will remain unchanged through graduation method,
    # because they do not have 5-year neighbors on both sides

    # use graduation method from Preston to fine-tune ax using neighboring
    # ax values.  ax at age groups [2-6, 32, 235] will be set to cm_mean_age
    ax = fbd_core.demog.ax_graduation_cubic(mx, ax, nx_base, graduation_ages)

    # now fix the under-5 age groups to match GBD methodology
    ax = under_5_ax_preston_rake(mx, ax, nx_base)

    LOGGER.debug("computing period life expectancy")
    ex = fbd_core.demog.fm_period_life_expectancy(mx, ax, nx_gbd)

    ex_null_cnt = ex.where(ex.isnull(), drop=True).size
    assert ex_null_cnt <= mx_null_cnt
    if ex_null_cnt > mx_null_cnt:
        ex_null = ex.where(ex.isnull(), drop=True)
        LOGGER.error("graduation introduced null draws with bounds {}".format(
            ex_null.coords
        ))
        raise RuntimeError("Graduation created {} nulls".
                           format(ex_null.coords))

    ds = xr.Dataset(dict(ex=ex, mx=mx, ax=ax))

    return append_nLx_lx(ds)


def gbd4_all_youth(mx):
    """
    Period life expectancy that matches GBD 2016 except young ages
    aren't fit, so age groups ENN, PNN, and LNN remain. This is called
    the "baseline" set of ages, not the "lifetable" set of ages.
    Input ages include ENN, PNN, and LNN, and the terminal age group
    is 95+. This includes the old age fit from US Counties code
    and uses the graduation method.

    Args:
        mx (xr.DataArray): Mortality rate

    Returns:
        xr.DataArray: Period life expectancy.
    """
    nx_gbd = fbd_core.demog.nx_contiguous_round(4, "baseline")
    # nx_gbd looks like array([1.917808e-02, 5.753425e-02, 9.232877e-01,
    # 4.000000e+00, 5.000000e+00, ... 5.000000e+00, 5.000000e+00, 4.500000e+01]
    try:
        mx = mx.loc[dict(age_group_id=nx_gbd.age_group_id.values)]
    except KeyError:
        raise RuntimeError("Not all ages in incoming data "
                           "have {} want {}".format(mx.age_group_id.values,
                                                    nx_gbd.age_group_id.values
                                                    ))
    assert fbd_core.demog.consistent_age_group_ids(mx.age_group_id.values)
    nx_base = fbd_core.demog.nx_from_age_group_ids(mx.age_group_id)
    # >>> nx_base['age_group_id']
    # <xarray.DataArray 'age_group_id' (age_group_id: 21)>
    # array([ 28,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,
    #         17,  18,  19,  20,  30,  31,  32, 235])
    # 28 is 1 year wide, 5 is 4 years wide

    # We want a baseline on nulls. If there is one null in a draw, we
    # null out the whole draw, so this algorithm may increase null
    # count, but it won't increase the bounding box on nulls.
    mx_null_cnt = mx.where(mx.isnull(), drop=True).size
    if mx_null_cnt > 0:
        warnings.warn("Incoming mx has {} nulls".format(mx_null_cnt))
    expected_good = mx.size - mx_null_cnt

    # first compute ax assuming constant mortality over interval
    ax = fbd_core.demog.cm_mean_age(mx, nx_base)
    # Graduation applies only where it's 5-year age groups.
    middle_ages = nx_base.where(nx_base == nx_base.median()).dropna(
            dim="age_group_id")
    graduation_ages = middle_ages.age_group_id.values
    # graduation ages are the age group ids that have 5-year neighbors:
    # array([ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30,
    #         31, 32])

    # use graduation method from Preston to fine-tune ax using neighboring
    # ax values.  The boundary ax's will be set to cm_mean_age
    ax = fbd_core.demog.ax_graduation_cubic(mx, ax, nx_base, graduation_ages)

    # Mortality from mx is interval-by-interval, so we
    # can leave young out of it.
    # mortality {}_nq_x = \frac{m_x n_x}{1+ m_x(n_x - a_x)}
    qxp = fbd_core.demog.fm_mortality(mx, ax, nx_base)
    assert (qxp < 1.0001).sum() >= expected_good
    assert (qxp > -0.0001).sum() >= expected_good
    qxf, axf, mxf = fbd_core.demog.old_age_fit_us_counties(qxp)
    # qxp['age_group_id'] == [2..20, 30..32, 235]
    # qxf['age_group_id'] == [ 31,  32,  33,  44,  45, 148]
    del qxp  # frees up 1G for 100 draws
    gc.collect()

    # This goes past the 95+ limit, so reduce it.
    mxp, axp = condense_ages_to_terminal(qxf, axf, 33, 235)
    v = mxf.age_group_id.values  # array([ 31,  32,  33,  44,  45, 148])
    term = np.argwhere(v == 33)[0][0]  # 2
    mxf_drop = xr.concat([
        mxf.loc[dict(age_group_id=v[:term])],
        mxp
    ], dim="age_group_id")  # 31, 32, 235
    axf_drop = xr.concat([
        axf.loc[dict(age_group_id=v[:term])],
        axp
    ], dim="age_group_id")  # 32, 32, 235

    del qxf, axf, mxf
    gc.collect()

    LOGGER.debug("combining young and old")
    mx_combined = combine_age_ranges(None, mx, mxf_drop)
    ax_combined = combine_age_ranges(None, ax, axf_drop)
    assert (mx_combined.age_group_id.values[-5:] ==
            nx_gbd.age_group_id.values[-5:]).all()

    del ax, mx  # frees up 2G for 100 draws
    gc.collect()

    LOGGER.debug("computing period life expectancy")
    ex = fbd_core.demog.fm_period_life_expectancy(
        mx_combined, ax_combined, nx_gbd)

    ex_null_cnt = ex.where(ex.isnull(), drop=True).size
    assert ex_null_cnt <= mx_null_cnt
    if ex_null_cnt > mx_null_cnt:
        ex_null = ex.where(ex.isnull(), drop=True)
        LOGGER.error("graduation introduced null draws with bounds {}".format(
            ex_null.coords
        ))
        raise RuntimeError("Graduation created {} nulls".
                           format(ex_null.coords))

    ds = xr.Dataset(dict(ex=ex, mx=mx_combined, ax=ax_combined))

    del mx_combined, ax_combined  # frees up 2G for 100 draws
    gc.collect()

    return append_nLx_lx(ds)


def gbd4_best_guess(mx):
    """
    Period life expectancy that matches GBD 2016 as best as possible.
    Input ages include ENN, PNN, and LNN, and the terminal age group
    is 95+. This includes the old age fit from US Counties code
    and uses the graduation method.

    Args:
        mx (xr.DataArray): Mortality rate

    Returns:
        xr.DataArray: Period life expectancy.
    """
    nx_base = fbd_core.demog.nx_from_age_group_ids(mx.age_group_id)
    nx_life = fbd_core.demog.nx_contiguous_round(4, "lifetable")

    under_one_mx = make_under_one_group_for_preston(mx)
    under_one_ax = preston_ax_fit(under_one_mx)

    ax = fbd_core.demog.cm_mean_age(mx, nx_base)
    # Graduation applies only where it's 5-year age groups.
    middle_ages = nx_base.where(nx_base == nx_base.median()).dropna(
            dim="age_group_id")
    graduation_ages = middle_ages.age_group_id.values
    ax = fbd_core.demog.ax_graduation_cubic(mx, ax, nx_base, graduation_ages)
    assert xr.ufuncs.isfinite(ax).all()
    # Mortality from mx is interval-by-interval, so we
    # can leave young out of it.
    qxp = fbd_core.demog.fm_mortality(mx, ax, nx_base)
    assert xr.ufuncs.isfinite(qxp).all()
    assert (qxp < 1.0001).all()
    assert (qxp > -0.0001).all()
    qxf, axf, mxf = fbd_core.demog.old_age_fit_us_counties(qxp)
    assert xr.ufuncs.isfinite(qxf).all()
    assert xr.ufuncs.isfinite(axf).all()
    # This goes past the 95+ limit, so reduce it.
    mxp, axp = condense_ages_to_terminal(qxf, axf, 33, 235)
    assert xr.ufuncs.isfinite(mxp).all()
    assert xr.ufuncs.isfinite(axp).all()
    v = mxf.age_group_id.values
    term = np.argwhere(v == 33)[0][0]
    mxf_drop = xr.concat([
        mxf.loc[dict(age_group_id=v[:term])],
        mxp
    ], dim="age_group_id")
    axf_drop = xr.concat([
        axf.loc[dict(age_group_id=v[:term])],
        axp
    ], dim="age_group_id")
    assert xr.ufuncs.isfinite(mxf_drop).all()
    assert xr.ufuncs.isfinite(axf_drop).all()

    LOGGER.debug("combining young and old")
    mx_combined = combine_age_ranges(under_one_mx, mx, mxf_drop)
    ax_combined = combine_age_ranges(under_one_ax, ax, axf_drop)
    assert xr.ufuncs.isfinite(mx_combined).all()
    assert xr.ufuncs.isfinite(ax_combined).all()

    LOGGER.debug("computing period life expectancy")
    ex = fbd_core.demog.fm_period_life_expectancy(
        mx_combined, ax_combined, nx_life)

    ds = xr.Dataset(dict(ex=ex, mx=mx_combined, ax=ax_combined))
    return append_nLx_lx(ds)


def condense_ages_to_terminal(qx, ax, terminal, new_terminal):
    """
    Given a life table that includes later ages, truncate it to
    a terminal age group. For the terminal age group, we know
    :math:`{}_nm_x = 1/{}_na_x`, so use that as our guide.

    .. math::

       {}_na_x = {}_na_{x_0} + {}_np_{x_0}(n_{x_0}+{}_na_{x_1})
       +{}_np_{x_0}\:{}_np_{x_1}(n_{x_0}+n_{x_1}+{}_na_{x_2})

    Note we get :math:`q_x` and return :math:`m_x`.

    Args:
        qx: Mortality with age group ids.
        ax: Mean age of death.
        terminal: Which age group will become the terminal one.
        new_terminal: The age group to assign to the last interval.

    Returns:
        mx: :math:`{}_nm_x` for the terminal age group.
        ax: :math:`{}_na_x` for the terminal age group.
    """
    v = qx.age_group_id.values
    term = np.argwhere(v == terminal)[0][0]
    assert term + 1 < v.shape[0]
    axp = ax.loc[dict(age_group_id=[terminal])]
    # This will be survival from the terminal age group
    # to the current age group in the loop.
    npx = 1 - qx.loc[dict(age_group_id=[terminal])]
    # This will be the total time from the terminal age group
    # to the current age group in the loop.
    nx = fbd_core.demog.nx_from_age_group_ids(ax.age_group_id)
    # This is a C-number, no index.
    nx_running = float(nx.loc[dict(age_group_id=[terminal])])
    LOGGER.debug("nx_running {} {}".format(type(nx_running), nx_running))

    for avx in v[term+1:]:
        # Have to set indices so that multiplication can happen.
        axp.coords["age_group_id"] = [avx]
        npx.coords["age_group_id"] = [avx]
        # Here, the invariant is now true.
        # npx is \prod_{i<avx} {}_np_{x_i}
        # nx_running = \sum_{i<avx} n_{x_i}
        axp += npx * (ax.loc[dict(age_group_id=[avx])] + nx_running)
        npx *= (1 - qx.loc[dict(age_group_id=[avx])])
        # This is a C-number, no index.
        nx_running += float(nx.loc[dict(age_group_id=[avx])])

    axp.coords["age_group_id"] = [new_terminal]
    return 1 / axp, axp


def gbd4_best_guess_no_old_age_fit(mx):
    """
    Period life expectancy that matches GBD 2016 as best as possible.
    Input ages include ENN, PNN, and LNN, and the terminal age group
    is 95+.

    Args:
        mx (xr.DataArray): Mortality rate

    Returns:
        xr.DataArray: Period life expectancy.
    """
    # assert 30 in mx.age_group_id.values #this is an issue
    nx = fbd_core.demog.nx_from_age_group_ids(mx.age_group_id)
    forecast_age_groups = ([28, 5] + list(range(6, 21)) + list(range(30,33))
                           + [235])
    assert consistent_age_group_ids(forecast_age_groups), \
        "The age groups aren't self-consistent"

    mxp, nxp = mx_on_new_ages(mx, forecast_age_groups)
    axp = fbd_core.demog.cm_mean_age(mxp, nxp)
    youth = preston_ax_fit(mxp)
    axp.loc[dict(age_group_id=youth.age_group_id)] = youth
    # Graduation applies only where it's 5-year age groups.
    middle_ages = nx.where(nx == nx.median()).dropna(dim="age_group_id")
    graduation_ages = middle_ages.age_group_id.values
    axp = fbd_core.demog.ax_graduation_cubic(mxp, axp, nxp, graduation_ages)

    ex = fbd_core.demog.fm_period_life_expectancy(mxp, axp, nxp)
    ds = xr.Dataset(dict(ex=ex, mx=mxp, ax=axp))
    return append_nLx_lx(ds)


def gbd3_bndp(mx):
    """
    Period life expectancy as it is currently calculated in the
    ``bndp_functions`` repository and used by our code.
    It uses a fit from Haidong for the old ages and, after that,
    applies the piecewise-constant mortality assumption.

    Args:
        mx (xr.DataArray): Mortality rate

    Returns:
        xr.DataArray: Period life expectancy.
    """
    LOGGER.info("Using single_bndp")
    mx.name = "mx"
    nx = fbd_core.demog.nx_from_age_group_ids(mx.age_group_id)
    nx.name = "nx"
    forecast_age_groups = ([28, 5] + list(range(6, 21)) + list(range(30, 34))
                           + [44, 45, 148])
    assert consistent_age_group_ids(forecast_age_groups), \
        "The age groups aren't self-consistent"

    mxp, nxp = mx_on_new_ages(mx, forecast_age_groups)
    mxp.name = "mxp"
    nxp.name = "nxp"

    # This calculation of a_x is just a bootstrap to do the fit.
    ax = fbd_core.demog.ud_mean_age(mx, nx)
    ax.name = "ax_udd"
    # The fit to a_x will be thrown away.
    mxpp, axpp = fbd_core.demog.old_age_fit_k2(mx, ax)
    LOGGER.debug("Coords on mxp {} and on mxpp {}".format(
        mxp.coords, mxpp.coords))
    mxp.loc[dict(age_group_id=mxpp.age_group_id.values)] = mxpp
    axp = fbd_core.demog.cm_mean_age(mxp, nxp)
    ex = fbd_core.demog.fm_period_life_expectancy(mxp, axp, nxp)
    ds = xr.Dataset(dict(ex=ex, mx=mxp, ax=axp))
    return append_nLx_lx(ds)


def gbd3_cubic(mx):
    """
    Calculate period life expectancy without fitting young or old
    age groups. Just apply a cubic fit, which we also call
    the graduation method, across all ages and see how many
    NaNs that produces.

    Args:
        mx (xr.DataArray): mortality rate

    Returns:
        xr.DataArray: Period life expectancy, likely with NaNs.
    """
    nx = fbd_core.demog.nx_from_age_group_ids(mx.age_group_id)
    ax = fbd_core.demog.cm_mean_age(mx, nx)
    # This assumes that age groups of same size are contiguous.
    same_ages = nx.where(nx == nx.median(), drop=True).age_group_id
    ax = fbd_core.demog.ax_graduation_cubic(mx, ax, nx, same_ages)
    lex_b = fbd_core.demog.fm_period_life_expectancy(mx, ax, nx)
    ds = xr.Dataset(dict(ex=lex_b, mx=mx, ax=ax))
    return append_nLx_lx(ds)


def make_under_one_group_for_preston(mx):
    """Create an under one age group (id 28) from the 0-6 days, 7-27 days,
    and 28-364 days (ids 2,3,4) groups. This age group is needed for the
    Preston young age fit. If this is called with scalar, string dimensions on
    mx, then slicing won't work and it will fail.

    Args:
        mx (xr.DataArray): Mortality data with age groups 2,3 and 4

    Returns:
        xr.DataArray: mx for the under one age group (28) and
            the 1-4 years age group (5).

    Raises:
        RuntimeError: if age groups ids 2, 3, and 4 are NOT in input
    """
    if all(age_x in mx.age_group_id.values for age_x in (2, 3, 4)):
        mx_other_ages = mx.loc[dict(age_group_id=[5])]
        mx_under_one = (7 * mx.loc[dict(age_group_id=2)]
                        + 21 * mx.loc[dict(age_group_id=3)]
                        + (365-28) * mx.loc[dict(age_group_id=4)])/365
        mx_under_one.coords["age_group_id"] = 28
        mx_under_one = xr.concat([mx_under_one, mx_other_ages],
                                 dim="age_group_id")
        assert np.array_equal(mx_under_one.age_group_id, [28, 5])
        return mx_under_one
    elif all(age_x in mx.age_group_id.values for age_x in (28, 5)):
        return mx.sel(age_group_id=[28, 5])
    else:
        raise RuntimeError("No known young ages in input")


def make_under_one_group_for_preston_with_nLx(mx, nLx):
    r"""Create an under one age group (id 28) from the 0-6 days, 7-27 days,
    and 28-364 days (ids 2,3,4) groups. This age group is needed for the
    Preston young age fit. If this is called with scalar, string dimensions on
    mx, then slicing won't work and it will fail.

    Note that this method using nLx as the weight, instead of nx, based
    on the definition

    .. math::
        {}_nm_x = \frac{{}_nd_x}{{}_nL_x}

    Args:
        mx (xr.DataArray): Mortality data with age groups 2, 3, 4, and 5.
        nLx (xr.DataArray): nLx that has age groups 2, 3, and 4

    Returns:
        xr.DataArray: mx for the under one age group (28) and
            the 1-4 years age group (5).

    Raises:
        RuntimeError: if age groups ids 2, 3, and 4 are NOT in input.
    """
    if all(age_x in mx.age_group_id.values for age_x in (2, 3, 4)):
        nLx_sum = nLx.sel(age_group_id=[2, 3, 4]).sum("age_group_id")
        mx_other_ages = mx.loc[dict(age_group_id=[5])]
        mx_under_one =\
            (nLx.sel(age_group_id=2) * mx.sel(age_group_id=2)
             + nLx.sel(age_group_id=3) * mx.sel(age_group_id=3)
             + nLx.sel(age_group_id=4) * mx.sel(age_group_id=4)) / nLx_sum
        mx_under_one.coords["age_group_id"] = 28
        mx_under_one = xr.concat([mx_under_one, mx_other_ages],
                                 dim="age_group_id")
        assert np.array_equal(mx_under_one.age_group_id, [28, 5])
        return mx_under_one
    elif all(age_x in mx.age_group_id.values for age_x in (28, 5)):
        return mx.sel(age_group_id=[28, 5])
    else:
        raise RuntimeError("No known young ages in input")

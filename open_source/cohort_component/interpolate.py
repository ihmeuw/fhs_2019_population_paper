"""
Tools for interpolation. These get applied repeatedly, so let's
put them somewhere they can be tested separately.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import copy
import itertools
import logging

import numpy as np
from scipy.interpolate import PchipInterpolator
import scipy.linalg
import xarray as xr

from fbd_core.db import connectors


LOGGER = logging.getLogger("fbd_research.cohort_component.interpolate")


def nx_from_age_group_ids(age_group_id):
    """
    Given a set of age group ids, this returns an XArray of
    the time span in each age group interval.
    Pass in the XArray coordinate::

       nx = nx_from_age_group_ids(past_mx.age_group_id)

    These come from the "shared" database.

    Args:
        age_group_id (np.ndarray or xr.DataArray): DataArray coordinate

    Returns:
         xr.DataArray: :math:`n_x` as an XArray DataArray of nx.
    """
    if isinstance(age_group_id, xr.DataArray):
        age_group_ids = age_group_id.values
    else:
        age_group_ids = age_group_id

    # This uses wrong substitution b/c I couldn't figure out how
    # to use sql string substitution for the list of IDs.
    query = ("SELECT *************************** "
             "FROM ********** "
             "WHERE ******** IN {}"
             "ORDER BY **********;").format(
        str(tuple(age_group_ids)))

    with connectors.db_connect("fbd-dev-read", database="shared") as conn:
        arr = np.array([x[0] for x in conn.execute(query).fetchall()],
                       dtype=np.float)
        if len(arr) != len(age_group_ids):
            raise KeyError(
                "Age group ids not all found or too many found {} {}".format(
                    age_group_ids, arr
                ))
        return xr.DataArray(arr, coords=[age_group_id], dims=["age_group_id"],
                            name="nx")


def nx_subset_contiguous(age_group_id):
    """
    Given a set of age groups, return a subset of those age groups such
    that the subset is contiguous, that the subset is ordered by start
    time of the interval, and that it is the contiguous subset that
    has the smallest age intervals. This exists in order to find those
    age groups that have the most-refined data and exclude aggregated data.

    For example, sometimes a set of age groups includes all ages. This would
    exclude that. Sometimes a set of age groups includes groups of
    ages, such as 0-14, 15-44, 45-59, and 60 up, in addition to the five-year
    age groups. Those would be excluded here.

    Args:
        age_group_id (collections.abs.Iterable): DataArray coordinate

    Returns:
         xr.DataArray: :math:`n_x` as an XArray DataArray of nx in years.
    """
    if isinstance(age_group_id, xr.DataArray):
        age_group_ids = age_group_id.values
    else:
        age_group_ids = age_group_id

    # This uses wrong substitution b/c I couldn't figure out how
    # to use sql string substitution for the list of IDs.
    query = ("SELECT **********, **********, "
             "       ********************** as nxdays "
             "FROM ********* "
             "WHERE ******* IN {} "
             "ORDER BY ********;").format(
        str(tuple(age_group_ids)))
    print(query)

    with connectors.db_connect("fbd-dev-read", database="shared") as conn:
        arr = [[int(y) for y in x] for x in conn.execute(query).fetchall()]
        if len(arr) != len(age_group_ids):
            raise KeyError(
                "Age group ids not all found or too many found {} {}".format(
                    age_group_ids, arr
                ))
        retain = list()
        for age_id, begin, width in arr:
            if retain and retain[-1][1] == begin:
                if retain[-1][2] > width:
                    retain[-1] = [age_id, begin, width]
                # else drop this one.
            else:
                retain.append([age_id, begin, width])

        return xr.DataArray(
            (np.array([wx[2] for wx in retain], dtype=np.double) + 1) / 365,
            coords={"age_group_id": [wx[0] for wx in retain]},
            dims=["age_group_id"],
            name="nx"
        )


class DemographicInterval:
    """
    An ordered set of age intervals, consecutive from some starting age.
    We often need the width of each interval, or the start of intervals,
    their finish times, or the boundaries of all intervals.
    This also compares intervals to create new ones.

    Attributes:
        nx (np.ndarray): Widths of intervals

        bound (np.ndarray): Endpoints of intervals. The length is
            one longer than the length of nx.

        start (np.ndarray): Array of starting ages.

        finish (np.ndarray): Array of finishing ages for each interval.

        omega (np.double): Oldest possible age in these intervals.
    """
    def __init__(self, nx, begin=0):
        """

        Args:
            nx (List[float]|np.ndarray): Width of each interval
            begin (float): Starting age, defaults to 0.
        """
        nx = np.array(nx, dtype=np.double)
        LOGGER.debug("di nx {} begin {} {}".format(nx, begin, nx.cumsum()))
        self.bound = np.hstack([[begin], begin + nx.cumsum()])
        self.nx = nx

    @property
    def start(self):
        return self.bound[:-1]

    @property
    def finish(self):
        return self.bound[1:]

    @property
    def omega(self):
        return self.finish[-1]

    def __getitem__(self, key):
        """Returns a new DemographicInterval subset. Still continuous."""
        if isinstance(key, slice):
            a = key.start or 0
            b = key.stop or len(self.nx)
        else:
            a, b = key, key + 1
        print("di getitem keystart {} {}".format(a, b))
        return DemographicInterval(self.nx[a:b], begin=self.bound[a])

    def overlaps_with(self, other):
        """All age groups in this interval that overlap the other intervals."""
        eps = 1e-6  # Accounts for two intervals nearly lining up.
        past_left = np.where(self.finish > other.start[0] + eps)[0]
        before_right = np.where(self.start < other.finish[-1] - eps)[0]
        return self.__getitem__(slice(past_left[0], before_right[-1] + 1))

    def __len__(self):
        return self.nx.shape[0]

    def __str__(self):
        if len(self.nx) > 2:
            return "({}, {}, {},.. {})".format(self.start[0], self.start[1],
                                               self.start[2], self.finish[-1])
        else:
            return "DemographicInterval({})".format(self.nx.shape)

    def __repr__(self):
        return "DemographicInterval({})".format(self.nx)


def nx_to_intervals(nx):
    """
    The interval is a representation of age groups.
    Demographers use :math:`n_x`, which is the size of each
    interval. This defines a two-dimensional array where::

       intervals[0] = Starts of intervals
       intervals[1] = Finishes of intervals

    Args:
        nx (np.ndarray): Array of nx.

    Returns:
        np.ndarray: Of interval information.
    """
    nx_endpoints = np.hstack([[0], nx.cumsum()])
    return np.array([nx_endpoints[:-1], nx_endpoints[1:]])


def intervals_to_nx(intervals):
    """Convert from intervals representation to an array of :math:`n_x`."""
    print(intervals)
    result = np.diff(intervals, axis=0)[0]
    print(result)
    assert (result > 0).all()
    return result


def overlap_ab(a, b):
    """Returns b x a matrix of overlaps."""
    LOGGER.debug("a {} b {}".format(a, b))
    onea, oneb = [np.ones(len(arr)) for arr in (a, b)]
    res = (np.minimum(np.outer(oneb, a.finish),
                      np.outer(b.finish, onea)) -
           np.maximum(np.outer(oneb, a.start),
                      np.outer(b.start, onea)))
    res[res < 0] = 0.
    return res


def composite_average(a, b):
    """Multiply this by a to get a b that is an average.
    Use with rates."""
    overlap = overlap_ab(a, b)
    return overlap / np.outer(b.nx, np.ones(len(a)))


def composite_sum(a, b):
    """Multiply this by a to get b that is a sum.
    Use with counts."""
    overlap = overlap_ab(a, b)
    assert np.isfinite(overlap).all()
    return overlap / np.outer(np.ones(len(b)), a.nx)


def uneven_average(nx_open):
    r"""
    This constructs a composition matrix that will distribute a total
    to more fine-grained intervals. If we looked at the
    upper left-hand corner, it would look like this.

    .. math::

        \left[\begin{array}{cccc}
        1 & 0   & 0   & 0 \\
        0 & 1/3 & 0   & 0 \\
        0 & 1/3 & 0   & 0 \\
        0 & 1/3 & 0   & 0 \\
        0 & 0  & 1/48 & 0 \\
        0 & 0  & 1/48 & 0 \\
        0 & 0  & 1/48 & 0
        \end{array}\right]

    This matrix does work that would normally be a for loop.
    """
    # The half-open interval will be a huge spike in population because
    # it contains lots of weeks, but that's what we intend.
    limit_open = copy.deepcopy(nx_open)
    limit_open[-1] = 1
    cnt = len(limit_open)
    LOGGER.debug("n_open {} cnt {}".format(limit_open, cnt))
    return np.concatenate(
        [
            np.tile(
                [0] * i + [1 / n] + [0] * (cnt - i - 1),
                (n, 1)
            )
            for (i, n) in enumerate(limit_open)
        ],
        axis=0
    ).reshape(limit_open.sum(), cnt)


def uneven_sum(nx_open):
    r"""
    This constructs a composition matrix that will sum fine-grained
    state variables into coarse-grained ones. If we looked at the
    upper left-hand corner, it would look like this.

    .. math::

        \left[\begin{array}{cccccccc}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 1 & 1 & 1 & 1
        \end{array}\right]

    This matrix would be used for PCLM.
    """
    # The open interval is many weeks but only one, equivalent, open-interval
    # in the new basis.
    limit_open = copy.deepcopy(nx_open)
    limit_open[-1] = 1
    cnt = len(limit_open)
    LOGGER.debug("n_open {} cnt {}".format(limit_open, cnt))
    return np.concatenate(
        [
            np.tile(
                [0] * i + [1] + [0] * (cnt - i - 1),
                (n, 1)
            )
            for (i, n) in enumerate(limit_open)
        ],
        axis=0
    ).reshape(limit_open.sum(), cnt).T


def five_year_projection(n, k=5*52):
    projection = np.concatenate([np.tile(y, k) for y in
                                 np.split(np.eye(n) / k, range(1, n))]) \
                                .reshape((n * k, n))
    return projection


def C0InterpolateRate(v, nta, ntb):
    return np.inner(five_year_projection(len(nta) + 1), v)


PCLMResult = collections.namedtuple("PCLMResult",
                                    "trace deviation gamma aic mu")
"""
trace - The trace of the H matrix.
deviation - Standard deviation
gamma - These are the values you're calculating
aic - Akaike information criterion
mu - The estimated value of predictors, given these gamma values
"""


def wlsq(A, B, w):
    """Weighted least squares"""
    W = np.sqrt(np.diag(w))
    x, *_ = scipy.linalg.lstsq(np.dot(W, A), np.dot(B, W))
    return x


def pclm(y, C, X=None, mu_init=None, lambda_factor=1e4, smoothing_degree=2,
         tolerance=1e-6, max_iterations=50):
    nx = C.shape[1]
    if X is None:
        X = np.eye(nx)
    D = np.diff(np.eye(nx), n=smoothing_degree, axis=0)
    la2 = np.sqrt(lambda_factor)
    if mu_init is not None:
        b = np.log(mu_init)
        LOGGER.debug("Using given initial mi")
    else:
        b_start = np.log(np.sum(y) / nx)
        b = b_start * np.ones((nx,), dtype=np.double)
    LOGGER.debug("input y {} C {} X {} init {}".format(
        y.shape, C.shape, X.shape, b.shape
    ))
    # LOGGER.debug("y {} {} {}".format(y.sum(), y.std(), y.shape))
    # LOGGER.debug("b {} {} {}".format(b.sum(), b.std(), b.shape))

    for idx in range(max_iterations):
        b0 = b
        eta = np.matmul(X, b)
        gamma = np.exp(eta)
        # LOGGER.debug("gamma {} {}".format(gamma.sum(), gamma.shape))
        mu = np.matmul(C, gamma)
        # LOGGER.debug("Delta mu {}".format(np.abs(mu - y).max()))
        w = np.hstack(
            [1 / mu, la2 * np.ones((nx - smoothing_degree,), dtype=np.double)])
        # LOGGER.debug("w {} {}".format(w.sum(), w.shape))
        Gamma = np.outer(gamma, np.ones((nx,), dtype=np.double))
        GX = Gamma * X
        # LOGGER.debug("Gamma {} {}".format(Gamma.sum(), Gamma.shape))
        # LOGGER.debug("X {} {}".format(X.sum(), X.shape))
        # LOGGER.debug("GX {} {}".format(GX.sum(), GX[:5,:5]))
        Q = np.matmul(C, GX)
        # LOGGER.debug("Q {} {} {}".format(Q.sum(), Q[:2,:11], Q.shape))
        z = np.hstack([y - mu + np.matmul(Q, b),
                       np.zeros((nx - smoothing_degree,), dtype=np.double)])
        # LOGGER.debug("z {} {} {}".format(z.sum(), z[:5], z.shape))
        # LOGGER.debug("D {}".format(D[:5, :5]))
        endogenous = np.vstack([Q, D])
        # LOGGER.debug("endog {}".format(endogenous.shape))
        b = wlsq(endogenous, z, w)
        #         Fit = sm.WLS(endogenous, z, weights=w).fit()
        #         print(dir(Fit))
        #         b = Fit.params.flatten()
        # LOGGER.debug("b {} {} {}".format(b.sum(), b.std(), b[:5]))
        db = np.abs(b - b0).max()
        LOGGER.debug("pclm db {}".format(db))
        if db < tolerance:
            break

    # Use the last b to get mu
    eta = np.matmul(X, b)
    gamma = np.exp(eta)
    mu = np.matmul(C, gamma)

    # Regression diagnostic
    R = np.matmul(Q.T, np.matmul(np.diag(1 / mu), Q))
    H = np.matmul(np.linalg.inv(R + np.matmul(lambda_factor * D.T, D)), R)
    trace = np.trace(H)
    ok = (y > 0) & (mu > 0)
    deviation = 2 * np.sum(y[ok] * np.log(y[ok] / mu[ok]))
    aic = deviation + 2 * trace
    return PCLMResult(trace, deviation, gamma, aic, mu)


def pclm_count(value, ntx, nti):
    """Common count when fitting to counts."""
    LOGGER.debug("dims {} ntx {} nti {}".format(value.shape, ntx, nti))
    C = composite_sum(nti, ntx)
    assert np.isfinite(C).all()
    # initial_guess = np.matmul(composite_sum(ntx, nti), value)
    res = pclm(value, C)
    return res.gamma


def constrained_cubic_disaggregation(x_coarse, y_coarse, x_fine):
    """
    Given observed values on a coarse domain, estimate their
    values on a finer domain. This uses constrained cubic splines.
    It constructs the cumulative sum of the y, fits the spline,
    interpolates to the finer x, and then takes differences across
    intervals.

    Args:
        x_coarse (np.ndarray): independent axis
        y_coarse (np.ndarray): dependent axis, same size as :math:`x`
        x_fine (np.ndarray): new independent axis

    Returns:
        np.ndarray: dependent values on finer axis
    """
    fit = PchipInterpolator(x_coarse, y_coarse, extrapolate=True)
    # Last entry is for T_{x+5}
    integrate_once = 1
    integrated = fit.antiderivative(integrate_once)(x_fine)
    return np.diff(integrated, 1, axis=0)


def constrained_cubic_disaggregation_interval(y_coarse, ntx, nti):
    """
    Given observed values on a coarse domain, estimate their
    values on a finer domain. This uses constrained cubic splines.
    It constructs the cumulative sum of the y, fits the spline,
    interpolates to the finer x, and then takes differences across
    intervals.

    Args:
        y_coarse (np.ndarray): dependent axis, same size as :math:`x`.
            This can have more than one dimension, and the algorithm
            treats all but the last dimension as separate runs.
        x_coarse (DemographicInterval): independent axis
        x_fine (DemographicInterval): new independent axis

    Returns:
        np.ndarray: dependent values on finer axis
    """
    out_shape = list(y_coarse.shape[:-1]) + [len(nti)]
    LOGGER.debug("in {} out {}".format(y_coarse.shape, out_shape))
    y_out = np.zeros(out_shape, dtype=y_coarse.dtype)
    for draw in itertools.product(*[range(x) for x in y_coarse.shape[:-1]]):
        y = np.hstack([[0], np.cumsum(y_coarse[draw], axis=-1)])
        fit = PchipInterpolator(ntx.bound, y, extrapolate=True)(nti.bound)
        y_out[draw] = np.diff(fit, 1, axis=-1)
    return y_out


def constrained_cubic_integrate_point(y_coarse, ntx, nti, end=130):
    """
    Like :py:function`constrained_cubic_disaggregation_interval`,
    this integrates over an interpolated curve, but this assumes the
    input data is defined pointwise at the start of each interval,
    so it first does a fit, then integrates, then fits, then takes
    derivatives.

    Args:
        y_coarse (np.ndarray): dependent axis, same size as :math:`x`.
            This can have more than one dimension, and the algorithm
            treats all but the last dimension as separate runs.
        x_coarse (DemographicInterval): independent axis
        x_fine (DemographicInterval): new independent axis
        end (float): Where to set the value to zero.

    Returns:
        np.ndarray: dependent values on finer axis
    """
    x = np.hstack([ntx.start, [end]])
    out_shape = list(y_coarse.shape[:-1]) + [len(nti)]
    y_out = np.zeros(out_shape, dtype=y_coarse.dtype)
    y = np.zeros((y_coarse.shape[-1] + 1,))
    y[-1] = 0
    for draw in itertools.product(*[range(x) for x in y_coarse.shape[:-1]]):
        y[:-1] = y_coarse[draw]
        assert x.shape == y.shape
        fit = PchipInterpolator(x, y, extrapolate=True)
        # Last entry is for T_{x+5}
        integrate_once = 1
        integrated = fit.antiderivative(integrate_once)(nti.bound)
        y_out[draw] = np.diff(integrated, 1, axis=-1)
    return y_out

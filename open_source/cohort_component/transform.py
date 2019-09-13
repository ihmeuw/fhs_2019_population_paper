"""
This transforms data from one set of age intervals to another.
How its done depends heavily on the application, which is construction
of a Leslie matrix.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cProfile as profile
import datetime
from fractions import Fraction
import logging

import numpy as np
import scipy
import xarray as xr

from fbd_core.argparse import ArgumentParser
from fbd_core.file_interface import FBDPath
from fbd_core import demog
import fbd_research.cohort_component.interpolate as interpolate


LOGGER = logging.getLogger("fbd_research.cohort_component.transform")


def lx_to_nLx_pclm(lx, ntx, nti):
    """
    Interpolate dx. Assume piecewise-constant hazard and calculate
    :math:`{}_nL_x`. This first interpolates :math:`d_x` to the finer age
    groups and then assumes they are now small enough to apply
    a piecewise constant assumption, which means that

    .. math::

        {}_na_x = \frac{1}{{}_nm_x} - \frac{n_x e^{-m_xn_x}}{1-e^{-m_xn_x}}.

    Then the usual definition for person-years lived is

    .. math::

        {}_nL_x = l_x({}_nq_x\:{}_na_x + {}_np_x\:n_x)

    so that the resulting value is

    .. math::

        {}_nL_x = -n_x\:l_x\frac{1-{}_np_x}{\ln({}_np_x)}

    The result is person-years lived, starting with :math:`l_x`.

    Args:
        lx (np.ndarray): :math:`l_x`
        ntx (np.ndarray): Intervals for larger time step
        nti (np.ndarray): Intervals for smaller time step

    Returns:
        np.ndarray: :math:`{}_nL_x` on smaller intervals.
    """
    dx = np.ediff1d(lx, to_end=[lx[-1]])
    di = interpolate.pclm_count(dx, ntx, nti)
    cdj = np.cumsum(di)
    li = np.hstack([cdj[-1], cdj[-1] - cdj])
    qi = np.where(li > 0, di / li, 0)
    ni = interpolate.intervals_to_nx(nti)
    nLx = np.where(qi > 0, -ni * li * qi / np.log(1-qi), 0)
    return li, nLx


class IntervalTransform(object):
    def __init__(self, components, nx, time_delta):
        """
        There's a target set of intervals. Each given dataset has its
        own source interval. We assume that ASFR, :math:`l_x`, and pop are on
        standard GBD intervals, with ASFR using a subset of them.
        The migration may have five-year or one-year intervals.

        Args:
            components (dict): From name to xr.Dataarray
            nx (xr.DataArray): age intervals.
            time_delta (Fraction): Fraction of a year for each time step.
        """
        self._asfr = components["asfr"]
        self._lx = components["lx"]
        self._migration = components["migration"]
        self._establish_intervals(
            nx, asfr.age_group_id.values,
            self._migration.age_group_id.values, time_delta)
        self._time_delta = time_delta
        self._t0 = Fraction(1, 2) * time_delta  # Start a half a year.

        self._asfrw = None
        self._lxw = None
        self._nlxw = None
        self._migrationw = None
        self._data_year = None

        self._pop_method = "cubic"

    def disaggregate_population(self, pop):
        """Given population on coarse set of intervals, return it on fine."""
        pop = pop.values
        # Getting rid of point dimensions.
        if len(pop.shape) > 2:
            single_dims = pop.shape[:-2]
            if not all(x==1 for x in single_dims):
                LOGGER.error("Population has extra dimensions {}".format(
                    pop.shape
                ))
            pop_end = np.squeeze(pop, axis=list(range(len(pop.shape) - 2)))
        else:
            pop_end = pop
        if not np.isfinite(pop_end).all():
            raise ValueError("Population isn't finite")
        if self._pop_method == "pclm":
            M = interpolate.pclm_count(pop_end[0], self._ntx, self._nti)
            F = interpolate.pclm_count(pop_end[1], self._ntx, self._nti)
            return np.vstack([M, F])
        elif self._pop_method == "cubic":
            return interpolate.constrained_cubic_disaggregation_interval(
                    pop_end, self._ntx, self._nti)
        else:
            raise RuntimeError("unknown method")

    def group_population(self, population):
        """Given population on fine intervals, return population on coarse."""
        return population

    def components(self, pop, t):
        if self._year(t) != self._data_year:
            self._convert_all(pop, t)

        return (self._asfrw, self._nlxw, self._migrationw)

    def fertile_ages(self):
        return np.zeros((5,))

    def _establish_intervals(self, nx, asfr_ages, migration_ages, time_delta):
        self._ntx = interpolate.DemographicInterval(nx.values)
        closed_extent_years = int(round(self._ntx.bound[-2]))
        assert abs(closed_extent_years - int(closed_extent_years)) < 1e-6
        closed_extent_weeks = int(closed_extent_years / time_delta)
        ni = np.ones(closed_extent_weeks, np.double) * time_delta
        ni = np.hstack([ni, self._ntx.omega - closed_extent_years])
        LOGGER.debug("end of ni {}".format(ni[-5:]))
        self._nti = interpolate.DemographicInterval(ni)

        asfr_idx = np.where(np.isin(nx.age_group_id.values, asfr_ages))[0]
        self._nta = self._ntx[asfr_idx[0]:asfr_idx[-1]]
        self._ntai = self._nti.overlaps_with(self._nta)
        self._first_fertile = self._nta.start[0]

        mig_nx = interpolate.nx_from_age_group_ids(migration_ages)
        self._ntm = interpolate.DemographicInterval(mig_nx.values)

    def _convert_all(self, pop, t):
        year_idx = self._year(t)

        self._nlxw = interpolate.constrained_cubic_integrate_point(
            self._lx.values[year_idx], self._ntx, self._nti)

        # Incoming rate is births per year for women during five-year period.
        # We want to move it to equal time duration and age duration,
        # so multiply by 5.
        self._asfrw = interpolate.C0InterpolateRate(
            5 * self._asfr.values[year_idx], self._nta, self._ntai)

        self._migrationw =\
            interpolate.constrained_cubic_disaggregation_interval(
                self._migration[year_idx], self._ntm, self._nti
        )
        self._data_year = year_idx

    def _year(self, when):
        full_when = self._t0 + when * self._time_delta
        return full_when.numerator // full_when.denominator




_MAIN_TO_TEST_INTERPOLATION = """
This entry point is made to debug interpolation by the
:py:class:`IntervalTransform`. It reads a set of data, interpolates
it, and makes graphs. We have to test them together because the
steps for interpolation will be intertwined.

It makes a lot of assumptions in order to make it easier to find the
required data.
"""
if __name__ == "__main__":
    version_today = datetime.datetime.now().strftime("%Y%m%d")

    parser = ArgumentParser(description=_MAIN_TO_TEST_INTERPOLATION)
    parser.add_argument("--version",
                        default=version_today,
                        help="version for output")
    # Use action=append in case one ASFR file is for the future
    # and another has that last year of the past that we need.
    parser.add_argument("--asfr", action="append", type=str,
                        help="Age-specific fertility rate")
    parser.add_argument("--pop", type=str,
                        default="20171112_gbd2016",
                        help="Starting population")
    parser.add_argument("--lifetable", action="append", type=str,
                        help="Life table with lx and nLx")
    parser.add_argument("--migration", action="append", type=str,
                        help="Migration")
    parser.add_argument("--location-idx", type=int, default=None,
                        dest="location_idx",
                        help=("Zero-based index into the list of locations. "
                              "If this is specified, only do that location."))
    parser.add_argument("--test", action="store_true",
                        default=False,
                        help="Run a small subset as a smoke test")

    args, _ = parser.parse_known_args()

    args.asfr = args.asfr if args.asfr is not None else [
        "20171120_arima_with_terminal_ages_raked"
    ]
    args.lifetable = args.lifetable if args.lifetable is not None else [
        "20171111_glob_sec_meanshift_squeezed"
    ]
    args.migration = args.migration if args.migration is not None else [
        "20171101_wpp_smoothed"
    ]

    gbd_round = 4
    location_id = 6
    nx = demog.nx_contiguous_round(gbd_round, purpose="baseline")
    time_delta = Fraction(1, 52)

    pop_file = FBDPath(gbd_round_id=gbd_round, past_or_future="past",
        stage="population", version=args.pop) / "population.nc"
    pop = xr.open_dataarray(str(pop_file))
    pop_cut = pop.loc[{"year_id": 2016, "location_id": location_id,
                       "sex_id": [1, 2]}].transpose(
        "sex_id", "age_group_id"
    )

    asfr_file = FBDPath(
        gbd_round_id=gbd_round, past_or_future="future",
        stage="asfr", version=args.asfr[0]) / "asfr.nc"
    asfr = xr.open_dataarray(str(asfr_file))
    asfr_cut = asfr.loc[{"draw": 0, "year_id": slice(2016, None),
                         "scenario": 0, "location_id": location_id}].transpose(
        "year_id", "age_group_id"
    )

    life_file = FBDPath(
        gbd_round_id=gbd_round, past_or_future="future",
        stage="life_expectancy", version=args.lifetable[0]) / "lifetable_ds.nc"
    life = xr.open_dataset(str(life_file))
    life_cut = life.lx.loc[{"draw": 0, "year_id": slice(2016, None),
        "scenario": 0, "location_id": location_id}].transpose(
        "year_id", "sex_id", "age_group_id"
    )

    migration_file = FBDPath(
        gbd_round_id=gbd_round, past_or_future="future",
        stage="migration", version=args.migration[0]) / "migration.nc"
    mig = xr.open_dataarray(str(migration_file))
    mig_cut = mig.loc[{"year_id": slice(2016, None),
                       "location_id": location_id}].transpose(
        "year_id", "sex_id", "age_group_id"
    )

    transform = IntervalTransform(
        dict(asfr=asfr_cut, lx=life_cut, migration=mig_cut), nx, time_delta)

    pop_fix = transform.disaggregate_population(pop_cut)
    asfr_fix, nlx, mig_fix = transform.components(pop_fix, 0)

    import fbd_research.cohort_component.plot as plot
    plot.diagnostic_interpolate(
        [asfr_cut, asfr_fix, transform._nta, transform._ntai],
        [life_cut, transform._lx, transform._ntx, transform._nti],
        [mig_cut, mig_fix, transform._ntm, transform._nti],
        [pop_cut, pop_fix, transform._ntx, transform._nti]
    )

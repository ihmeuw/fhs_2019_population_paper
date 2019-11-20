r"""
This section describes what options are available and what choices have been
made for forward projection.

 #. Time steps are implemented as one week, but they save each year at the
    half year. Implementation of half-year, year, or five year time steps
    are all reasonable and would require minor modifications.

    Age groups intervals are usually set to equal the time steps. A previous
    version of this code separated under-one time steps into single weeks
    and over-one time steps into single years. This code uses the same
    time step for all years because it makes the code clearer and is
    #actually faster2

 #. Initial populations come from the GBD, taken from `db_queries`.
    These are interpolated onto weekly populations. The population is
    defined at the half-year, so all other data must not only include
    future values but also past values for the last known half-year.

 #. Survivorship can be calculated a number of ways. Only one is implemented
    here.

    #. A ratio of :math:`{}_nL_{x+n}/{}_nL_x` is the chosen method. This
       requires, as with all other steps, interpolation.

    #. There is an adjustment to survivorship that uses an estimate
       of the local growth rate, meaning the estimated growth rate
       for each age interval of the population. We could calculate this
       if the model isn't responsive enough to rapid changes in fertility
       or mortality.

    #. Given the very small time steps, it could be reasonable to use
       an approximation that takes a geometric mean of survivals,
       :math:`({}_np_{x+n}\:{}_np_x)^{1/2}`.

 #. Migration data is internally computed.  Please refer to the migration/ folder.


 #. Age-specific fertility rates (ASFR) come from IHME data, are on five-year
    time steps and GBD intervals. There are several methods for interpolation.

    #. Interpolation uses a cubic spline constrained so that ASFR is
       non-negative. There is code to do a better cubic spline
       using PCLM, as above.

    #. There is a possible improvement on the interpolation of ASFR
       that would ensure the leading eigenvalue of the Leslie matrix
       for weekly projection is equal to the weekly rate of growth
       given by the initial five-year estimates of ASFR, survivorship,
       and population. It's described in the technical report and would
       require some extra steps in the interpolation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import itertools
import logging
from fractions import Fraction

import numpy as np
from scipy.interpolate import PchipInterpolator
import xarray as xr

from fbd_core import demog
from fbd_research.cohort_component.transform import IntervalTransform
from fbd_research.cohort_component.interpolate import uneven_average, \
    uneven_sum, five_year_projection, nx_from_age_group_ids
from fbd_research.cohort_component.leslie import has_dims, LeslieMatrix
from fbd_research.cohort_component.tracing import plot as traceplot


LOGGER = logging.getLogger("fbd_research.cohort_component.single")

MAX_AGE = 125
POP_MINIMUM = 1.0


def lx_to_nLx_weeks(lx, ruler, ruler_extended):
    """
    Integrates a fitted lx curve to compute nLx, which would be used to compute
    survivorship downstream.

    Args:
        lx (np.array): lx with extrapolated values beyond fhs terminal age
            group start.
        ruler (Ruler): ruler for the fhs age groups.  This ruler determines
            #the age group starts that need to be returned2
        ruler_extended (Ruler): ruler that matches lx.  This ruler, along with
            lx, provide the points for interpolation.

    Returns:
        (np.array): nLx, in shape of (sex, age), where age corresponds to
            fhs age group ids.
    """
    assert has_dims(lx, "sex age")
    LOGGER.debug("lx dims {}".format(lx.shape))

    # We need 2 extra data points beyond the terminal age group start to
    # compute its survivorship: terminal_age_start + 1 week, and 110 years.
    x_all_weeks = np.concatenate([ruler.x_ww,
                                  [ruler.x_ww[-1] + 1,
                                   ruler_extended.x_ww[-1]]])  # all in weeks

    integrated_lx = list()
    for sex in [0, 1]:
        # first interpolate lx over the "extended" fhs age groups (in weeks)
        fit = PchipInterpolator(ruler_extended.x_gw, lx[sex], extrapolate=True)
        # now integrate lx and evaluate over all weeks
        integrate_once = 1
        integrated = fit.antiderivative(integrate_once)(x_all_weeks)
        LOGGER.debug("integrated lx end {}".format(integrated[-5:]))
        traceplot("integrated_lx", x_all_weeks, integrated)
        integrated_lx.append(integrated)
    lx_int_sexed = np.vstack(integrated_lx)
    # Will be one less than number of weeks
    return np.diff(lx_int_sexed, 1, axis=1)


def nLx_to_survivorship(nLx):
    """
    survivorship is computed as :math: `_nL_{x+n} / _nL_x`.

    Special treatment of the survivorship of the last week:
        :math: `_nL_{x + \inf} / (_nL_x + _nL_{x + \inf})`

    :math:`_nL_0` is also conveniently returned to help compute the survival
        of new borns.

    Args:
        nLx (np.array): person-years lived, shaped in (sex, age).  Ages are
            in weeks.

    Returns:
        (np.array): nL0 and survivorship.  The first slice is nLx for the
            newborns, passed to downstream.  The rest of the array is
            survivorship.
    """
    # Assumes nLx has even time intervals up to last entry, which is T_{x+5}.
    LOGGER.debug("nLx {}".format(nLx))
    assert len(nLx.shape) == len("sex age".split())

    # the first slice, nLx[:, :1] is conveniently returned here to help
    # compute the survival of newborns
    survivorship = np.hstack([
        nLx[:, :1],  # not survivorship, but returned for downstream uses
        nLx[:, 1:-1] / nLx[:, :-2],  # survivorship between weekly age groups
        nLx[:, -1:] / (nLx[:, -2:-1] + nLx[:, -1:])])  # survivorship right before open age group
    no_low = np.where(survivorship > 0, survivorship, 0.0)
    no_high = np.where(no_low < 1, no_low, 1.0)
    return no_high


def population_to_weeks(population, ruler):
    assert population.shape[0] == 2
    # Put population of half-open interval all the way out at omega age.
    x_open = np.concatenate([ruler.x_gw, [ruler.omega]])
    x_all_weeks = np.concatenate([ruler.x_ww, [ruler.omega]])
    LOGGER.debug("length of new basis {} {}".format(len(x_all_weeks),
                                                    ruler.x_ww[-5:]))

    pops = list()
    for sex_idx in [0, 1]:
        integrated_pop = np.concatenate([[0], np.cumsum(population[sex_idx])])
        fit = PchipInterpolator(x_open, integrated_pop)(x_all_weeks)
        pop_weeks = np.diff(fit, 1)
        pops.append(pop_weeks)

    assert pops[0].shape[-1] == 4941
    return np.vstack(pops)


def population_to_weeks_by_projection(population, ruler):
    """
    Converts population on age group Ids to weekly populations
    using a simple projection matrix, so you get a stairwise step
    function of population.
    """
    assert population.shape[0] == 2
    averaging = uneven_average(ruler.nx_gw)
    LOGGER.debug("averaging shape {}".format(averaging.shape))
    pop_weeks = np.einsum("wa,sa->sw", averaging, population)
    traceplot("pop_weeks", pop_weeks[0])
    assert pop_weeks.shape[0] == 2
    return pop_weeks


def asfr_to_weeks(asfr):
    """
    ASFR is per year. Preston, Heuveline, and Guillot defines age-specific
    fertility rate, :math:`{}_nF_x[0,T]`, as
    "Births in the period 0 to T to women aged
    x to x + n, divided by person-years lived in the period 0 to T by women
    aged x to x + n. In our case, the T is one year. The demographic notation
    for n is misleading for this variable because it isn't an integral over an
    interval of time. It's an integral over ages. Long story short, we need to
    multiply by n=5 in order to get what PHG uses as a matrix element for
    five-year age groups.
    """
    return np.inner(5 * five_year_projection(asfr.shape[-1]), asfr)


def population_to_groups(pop_buffer, nx_gw):
    return np.einsum("aw,sw->sa", uneven_sum(nx_gw), pop_buffer)


Ruler = namedtuple("Ruler", "nx_gw x_gw y_gw x_ww fertile ratio omega")
"""
This class holds matrices and arrays to help transformation among
age and time intervals.

Attributes:
    nx_gw (ndarray): is age intervals for original age groups, time in weeks.
    x_gw (ndarray): is start times of original age groups, time in weeks.
    y_gw (ndarray): is end times of original age groups, time is in weeks.
    x_ww (ndarray): is start times of weekly age groups, time is in weeks.
    fertile (int, int): is tuple of (first fertile week, last fertile week + 1)
    ratio (int): is 52, for weeks per year.
    omega (float): is end of life, which is either 125 or 140.
"""


def timeline(age_groups, asfr_ages):
    """
    This works out time intervals for conversion from one set of age
    groups to another.

     * nx is age intervals, x is start of an interval, y is end of an interval.
     * gy is on GBD age group IDs, expressed as years.
     * gw is on GBD age group IDs, expressed as weeks.
     * ww is on weekly intervals, expressed as weeks.
     * fertile_begin, fertile_end are limits on fertile age weeks.

    Assumes nx ends in half-open interval

    Args:
        age_groups (np.ndarray): Age group IDs
        asfr_ages (np.ndarray): Age group IDs for just ASFR.

    Returns:
        Ruler: As described above.
    """
    nx_gy_da = nx_from_age_group_ids(age_groups)
    LOGGER.debug("nx_gy {}".format(nx_gy_da.values))
    nx_gw_da = (nx_gy_da * 52).round().astype(int)
    # end times for each interval
    y_gw_da = nx_gw_da.cumsum("age_group_id")
    # start times for each interval
    x_gw_da = y_gw_da.shift(age_group_id=1)\
                     .fillna(0.0)\
                     .astype("int")
    # Determine start of last week by start of last GBD age group.
    # The +1 is so that we include the start of the half-open interval.
    x_ww = np.arange(0, x_gw_da[-1] + 1)
    fertile_x = x_gw_da.loc[dict(age_group_id=asfr_ages)]
    fertile_nx = nx_gw_da.loc[dict(age_group_id=asfr_ages)]
    fertile_begin = int(fertile_x[0].values)
    fertile_end = int(fertile_x[-1] + fertile_nx[-1])
    ruler = Ruler(
        nx_gw=nx_gw_da.values,
        x_gw=x_gw_da.values,
        y_gw=y_gw_da.values,
        x_ww=x_ww,
        fertile=(fertile_begin, fertile_end),
        ratio=52,
        omega=y_gw_da.values[-1]
        )
    LOGGER.debug("nx_gw {} x_gw {} y_gw {} x_ww {} fertile {} ratio {}".format(
        len(ruler.nx_gw), len(ruler.x_gw), len(ruler.y_gw), len(ruler.x_ww),
        ruler.fertile, ruler.ratio
    ))
    return ruler


def _age_last(da, match_ordering=None):
    """
    Move age_group_id last in the dimensions.
    """
    if match_ordering is not None:
        dims = [d for d in match_ordering if d in da.dims]
    else:
        dims = list(da.dims)
    if "age_group_id" in dims:
        dims.remove("age_group_id")
        dims.append("age_group_id")
    try:
        return da.transpose(*dims)
    except ValueError:
        LOGGER.error("Array {} dims {} Transpose dims {}".format(
            da.name, da.dims, dims
        ))
        raise RuntimeError("Incoming data axes incommensurate")


def _pop_aging_iterator(yr_to_wk_func, iterated, ruler, **kwargs):
    """Abstracts pop-specific aging by week.

    FHS population forecast is performed on a week-to-week basis.  For every
    week in the future, this method returns a quantity that modifies
    the population by what happens during the course of that one week.

    Since quantities that change the population (migration, asfr, etc.) are
    provided in 1-year or 5-year age groups, one must write a function that
    converts the yearly numbers to weekly numbers.

    Hence the user must supply a func:

    yr_to_wk_func(iterated_year, ruler, **kwargs)

    That converts the yearly values to weekly values.

    Because pop is reported at the mid-year point, we first evolve the last
    past year by 1/2 year.  Then, for all forecast years, we evolve all weeks of
    each year by that year's inputs (lx, migration, asfr, etc.).  For example,
    If 2017 is the last past year, we use lx from 2017 to age the 2017 pop for
    half a year (26 weeks) to reach 01-01-2018.  Then we use lx from 2018 to
    age the pop for an entire year (52 weeks) to reach 01-01-2019, and so on.

    Args:
        yr_to_wk_func (function): a method that takes in iterated, ruler, and other args,
            to return an iterable
        iterated (np.array): array of yearly asfr/migration/lx/etc numbers that
            contribute to population change.  Usually shaped as
            (year_id, age_group_id) or (year_id, sex_id, age_group_id), where
            year_id is always the first dim and age_group_id always the last.
            The first year_id corresponds to the last past year.
        ruler (Ruler): ruler object that maps fbd age groups to weeks.

    Returns:
        (np.array): Leslie matrix element for each week.
    """
    for year_idx, iterated_year in enumerate(iterated):
        out = yr_to_wk_func(iterated_year, ruler, **kwargs)

        if year_idx == 0:  # for last past year, only yield for 1/2 year
            for _ in range(ruler.ratio // 2):
                yield out
        else:
            for _ in range(ruler.ratio):
                yield out

    # For the years beyond prescription, just keep constant
    while True:
        yield out


def _survivorship(lx_year, ruler, ruler_extended, **kwargs):
    """Iterate over survivorship by week.

    The survivorship is used to age the population week-by-week, starting
    from the last past year.

    Because pop is reported at the mid-year point, we first age the last
    past year by 1/2 year (using last past year's lx).  Then, for all forecast
    years, we age all weeks of each year by that year's lx.  For example,
    if 2017 is the last past year, we use lx from 2017 to age the 2017 pop for
    half a year (26 weeks) to reach 01-01-2018.  Then we use lx from 2018 to
    age the pop for an entire year (52 weeks) to reach 01-01-2019, and so on.

    Args:
        lx_year (np.array): lx in shape (sex_id, age_group_id).
        ruler (Ruler): ruler object that maps fbd age groups to weeks.
        ruler_extended (Ruler): identical to ruler, but with 5-year age groups
            within the fhs terminal age group.

    Returns:
        (np.array): for every week, returns np.array in the shape of
            (sex_id, 1 + len(ruler.x_ww)), where 1 corresponds to nL0 and
            len(ruler.x_ww) corresponds to the survivorship elements of the
            Leslie matrix.
    """
    nLx = lx_to_nLx_weeks(lx_year, ruler, ruler_extended)
    traceplot("nLx", nLx[1])
    survivorship = nLx_to_survivorship(nLx)
    traceplot("surv", survivorship[1])
    return survivorship


def _asfr(asfr_year, ruler, **kwargs):
    """Iterate over ASFR by week, analogous to method in _survivorship().

    Args:
        asfr_year (np.array): array in shape of (year_id, age_group_id).
        ruler (Ruler): maps asfr age groups to weeks.

    Returns:
        (np.array): weekly iterated asfr.
        (np.array): fertile age group ids.
    """
    projected = asfr_to_weeks(asfr_year)
    traceplot("asfr_projected", projected)
    return projected, ruler.fertile


def _migration(migration_year, ruler, **kwargs):
    """Iterate over migration by week, analogous to method in _survivorship().

    Args:
        migration_year (np.array): in shape of (year_id, sex_id, age_group_id).
        ruler (Ruler): maps migration data age groups to weeks.

    Returns:
        (np.array): weekly iterated migration.
    """
    migration_weekly = _migration_to_weeks(migration_year, ruler)
    traceplot("migrationw", migration_weekly[0])
    return migration_weekly


def _migration_to_weeks(migration_year, ruler):
    composition = (1 / 52) * uneven_average(ruler.nx_gw)
    LOGGER.debug("composition {} migration {}".format(composition.shape,
                                                      migration_year.shape))
    migration_weekly = np.einsum("wa,sa->sw", composition, migration_year)

    return migration_weekly


def project_draw(asfr, lx, migration, pop_start_year, srb, ruler,
                 ruler_extended, migration_ruler, forecast_years):
    """
    Take datasets and produce future populations.
    The datasets have to start with the year for the observed population.
    This data is all Numpy arrays, all for the same
    location, scenario, and draw. That means it will be organized by
    (year_id, sex_id, age_group_id) with exceptions as noted below.

    Args:
        asfr (np.ndarray): Age-specific fertility rate, organized as
            (year_id, age_group_id).

        lx (np.ndarray): :math:`l_x`, organized as
            (year_id, sex_id, age_group_id).

        migration (np.ndarray): Migration, organized as
            (year_id, sex_id, age_group_id).

        pop_start_year (np.ndarray): Starting population, organized as
            (sex_id, age_group_id).

        srb (np.ndarray): Sex ratio at birth with just one value
            for the last past year

        ruler (Ruler): Time scales for conversion to weeks.

        ruler_extended (Ruler): identical to ruler, but with 5-year age groups
            within the fhs terminal age group.

        migration_ruler (Ruler): Time scales for conversion from years to
            weeks.

        forecast_years (list): The list of all years in the forecast

    Returns:
        np.ndarray: Population at starting and all future years.
    """
    # All ndarrays at this point.
    assert has_dims(asfr, "year age")
    assert has_dims(lx, "year, sex, age")
    assert has_dims(migration, "year sex age")
    assert has_dims(pop_start_year, "sex, age")

    # Project from middle of first year to middle of last year.
    year_cnt = len(forecast_years)
    assert year_cnt <= lx.shape[0] - 1, "Not enough years in lx"
    assert year_cnt <= asfr.shape[0] - 1, "Not enough years in asfr"
    pop_save = np.zeros((year_cnt,) + pop_start_year.shape, dtype=np.double)
    pop_buffer = population_to_weeks(pop_start_year, ruler)

    survivorship_iter = _pop_aging_iterator(_survivorship, lx, ruler,
                                            ruler_extended=ruler_extended)
    asfr_iter = _pop_aging_iterator(_asfr, asfr, ruler)
    migration_iter = _pop_aging_iterator(_migration, migration, migration_ruler)

    save_year = 0

    for step_idx in range(ruler.ratio * year_cnt):
        asfr, age_limits = next(asfr_iter)
        survivorship = next(survivorship_iter)
        migration = next(migration_iter)
        traceplot("pop_buffer_in", pop_buffer[1])

        leslie = LeslieMatrix(asfr, age_limits[0], survivorship, float(srb))
        pop_t_plus_one = leslie.increment(pop_buffer)

        pop_buffer = pop_t_plus_one + migration

        if np.any(pop_buffer < 0):
            leslie_error_cnt = np.sum(pop_t_plus_one < 0)
            survivorship_neg = np.sum(survivorship < 0)
            asfr_neg = np.sum(asfr < 0)
            if leslie_error_cnt > 0:
                LOGGER.error("Leslie matrix negative values {} {} {}".format(
                    leslie_error_cnt, survivorship_neg, asfr_neg
                ))
            pop_buffer[pop_buffer < 0] = 0

        LOGGER.debug("pop_mig young F {}".format(pop_buffer[1, :10]))
        traceplot("pop_buffer_out", pop_buffer[1])
        LOGGER.debug("pop {:3g}".format(pop_buffer.sum()))

        year_idx = (step_idx + 1) // ruler.ratio
        if save_year != year_idx:
            pop_save[year_idx - 1] = \
                population_to_groups(pop_buffer, ruler.nx_gw)
            LOGGER.debug("pop save step {} {:3g}".format(
                step_idx, pop_save[year_idx - 1].sum()))
            save_year = year_idx

    if POP_MINIMUM is not None:
        pop_save[pop_save < POP_MINIMUM] = POP_MINIMUM
    return pop_save


def _ccmp_by_draw(da_dict, time_delta, nx):

    # Project from middle of first year to middle of last year.
    year_cnt = da_dict["lx"].shape[0] - 1
    pop_save = np.zeros((year_cnt,) + da_dict["pop"].shape, dtype=np.double)

    transformer = IntervalTransform(da_dict, nx, time_delta)
    fertile_ages = transformer.fertile_ages()
    pop_buffer = transformer.disaggregate_population(da_dict["pop"])

    save_year = 0
    for t in range(int(year_cnt / time_delta)):

        asfr, lx, migration = \
            IntervalTransform.components(pop_buffer, t)
        survivorship = nLx_to_survivorship(lx)

        leslie = LeslieMatrix(asfr, fertile_ages, survivorship)
        pop_buffer = leslie.increment(pop_buffer) + migration

        year_idx = int((t + 1) * time_delta)
        if save_year != year_idx:
            pop_save[year_idx - 1] = transformer.group_population(pop_buffer)
            LOGGER.debug("pop save step {} {:3g}".format(
                t, pop_save[year_idx - 1].sum()))
            save_year = year_idx
    return pop_save


def _transpose_if_present(da):
    desired_order = ["location_id", "scenario", "draw",
                     "year_id", "sex_id", "age_group_id"]
    goal_order = [dx for dx in desired_order
                  if dx in da.dims]
    return da.transpose(*goal_order)


def entry(pop_start, lx, asfr, migration,
          year_cnt=1, gbd_round=4, draw_cnt=100):
    """
    This is what you call to run CCMP. All data is in XArray.
    All locations, and scenarios, should match. If something lacks
    draws, the same data will be used across draws, but we expect
    some input to have draws and all given draws to match.

     * Years must be present for all, where pop_start defines first year.
     * Sex must include 1, 2.
     * Age will be subset to finest age groups.
     * Locations must all match.
     * Scenarios, data will be copied for those lacking scenarios.
     * Draws, data will be copied here, too.

    The data will be subset to the working group, and then this calls
    load() to bring them into memory from XArray's use of dask.

    Args:
        pop_start (xr.Dataarray): Population on the half-year.

        lx (xr.Dataarray): :math:`l_x` with :math:`l_0=1`, defined from
            the start of the year. This will need at least two years
            of data to match the population years. This applies to the
            population during the year.

        asfr (xr.Dataarray): Age-specific fertility rate, which is fertility
            per year. Like `lx`, needs an extra half-year back to reach pop.
            This applies to the population during the year.

        migration (xr.Dataarray): Migration in count space, not rate space.
            This is the number of people to migrate over the course
            of the year. This is not thousands of people, but the count.

        year_cnt (int): How many years to compute.

        gbd_round (int): GBD round. 4 is GBD 2016.

        draw_cnt (int): Number of draws to use across all calculations.

    Returns:
        xr.Dataarray: Population in the age groups for this GBD round.
    """
    for la, lb in itertools.combinations([pop_start, lx, asfr, migration], 2):
        if not la.location_id.values.array_equal(lb.location_id.values):
            raise ValueError("location ids don't match")

    try:
        start_year = int(pop_start.year_id)
    except TypeError:
        start_year = int(pop_start.year_id[-1])
    year_range = slice(start_year, start_year + year_cnt + 1)
    nx = demog.nx_contiguous_round(gbd_round, purpose="baseline")

    sex_slice = slice(1, 3)  # Use a slice so that ASFR is OK without sex=1.
    subset_da = {name: xda.loc[{
        "year_id": year_range,
        "sex_id": sex_slice,
        "age_group_id": nx.age_group_id.values
        }]
        for (name, xda) in [("pop", pop_start), ("lx", lx)]}

    # If someone adds a total birth rate age, we don't want it.
    keep_gbd_ages = [aid for aid in asfr.age_group_id.values
                     if aid in nx.age_group_id.values]
    subset_da["asfr"] = asfr.loc[{
        "year_id": year_range,
        "age_group_id": keep_gbd_ages
    }]

    subset_da["migration"] = migration.loc[{
        "year_id": year_range,
        "sex_id": [1, 2],
    }]

    # Load disables dask parallelism. Do this b/c we have OpenMP parallelism.
    transposed = {
        yname: _transpose_if_present(yda).load()
        for (yname, yda) in subset_da.items()
    }

    locations = transposed["pop"].location_id.values
    scenarios = [-1, 0, 1]
    result_coords = [
        ("location_id", locations),
        ("scenario", scenarios),
        ("draw", list(range(draw_cnt))),
        ("year_id", range(start_year + 1, start_year + year_cnt + 1)),
        ("sex_id", [1, 2]),
        ("age_group_id", nx.age_group_id.values)
    ]
    pop_result = xr.DataArray(
        np.zeros([len(bx[1]) for bx in result_coords], dtype=np.double),
        coords=dict(result_coords),
        dims=[ax[0] for ax in result_coords]
    )

    for l, s, d in itertools.product(locations, scenarios, range(draw_cnt)):
        each_draw = {"location_id": l, "scenario": s, "draw": d}
        draw_slice = {
            zname: zda.loc[{x: each_draw[x]
                            for x in each_draw.keys() & zda.dims}]
            for (zname, zda) in transposed.items()
        }

        pop_result.loc[each_draw] = _ccmp_by_draw(
            draw_slice,
            Fraction(1, 52),
            nx
        )

    return pop_result

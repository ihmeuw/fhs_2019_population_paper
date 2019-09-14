"""
Command line to run cohort-component model.

Four kinds of input files: one year of population, age-specific fertility
rate, lifetable, and migration. They can have dimensions in
location, sex, draw, scenario, year, and age group. There will be only
one starting population, but each of the other datasets can be in two
files, where one has the past, including the year for the given population,
and the other file has future values.

We choose the first year by taking the year of the population, but the
population may include past values or estimated future values from another
source. The first year will either be the only year of population, the last
year, or a year specified on the command line.

There may or may not be scenarios in the various files. We need to pair
all the scenarios with each other and, for those that lack scenarios,
pair the single estimation with the scenario ones.

The number of draws can differ, so we take a subset of draws or upsample
them. Draws and scenarios might differ between past and future versions
kof asfr, lx, or migration.

There may be a need to subset these by location, if it gets slow.

Here are some of the versions that were used for testing:

asfr: 5/future/asfr/20171120_arima_with_terminal_ages_raked
lifetable: 4/future/life_expectancy/20171111_glob_sec_meanshift_squeezed
migration: /ihme/forecasting/data/migration/170830_cleaned.nc
population: 4/future/past/population/20180112_all_past_years

Example call:

.. code::bash
	python driver.py --asfr 20190604_edu --pop 20181207_gbd105_y2017_draw --lifetable future/20190628_test --lifetable past/20190115_aggregated_from_20190109_version90_etl --migration 20190617_model6_arima --srb 20181205_107 --version test_something --location-idx 6 --years 1990:2018:2100 --gbd-round-id 5 --draws 1000
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import logging
import os
from pathlib import Path
import time

import numpy as np
import xarray as xr

from fbd_core.argparse import ArgumentParser
import fbd_core.db
from fbd_core.etl import broadcast_and_fill, resample, expand_dimensions
from fbd_core.file_interface import FBDPath
from fbd_core.demog import consistent_age_group_ids
from fbd_research.cohort_component.single import project_draw, timeline
from fbd_research.cohort_component.exc import MigrationError
from fbd_research.lex import model
from fbd_core.file_interface import roots, save_xr

LOGGER = logging.getLogger("fbd_research.cohort_component.driver")


def perf_time():
    if hasattr(time, "perf_counter"):
        return time.perf_counter()
    else:
        return time.clock()


def run_against(
            version, pop_version, asfr_version, lifetable_version,
            migration_version, srb_version, gbd_round_id, location_idx, years,
            location_id, draws, test=False):
    """
    Takes versions for files, finds the files, and computes future
    populations. It then saves those files. This is what you call from
    the pipeline.

    Args:
        version (str): Version name for output
        pop_version (str): Version for population
        asfr_version (str): version for asfr
        lifetable_version (list[str]): List of versions for lifetable
        migration_version (list[str]): List of versions for migration
        gbd_round_id (int): GBD Round ID, 4 is 2016
        location_idx (int|None): Zero-based index into list of locations.
        years (YearRange): years for past and forecast.
        location_id (int|None): A location ID.
        test (bool): Run a reduced subset of locations and draws.

    Returns:
        None
    """
    out_path = FBDPath("/{}/future/population/{}".format(gbd_round_id, version))
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except OSError as ose:
        LOGGER.error("Could not create output directory {}: {}".format(
            out_path, ose
        ))

    asfr_lim, lifetable_lim, pop, migration, srb =\
        agreement_rules(
            *read_datasets(
                asfr_version, gbd_round_id,
                lifetable_version, pop_version, migration_version, years,
                srb_version, draws),
            years
        )

    ruler = timeline(pop.age_group_id.values, asfr_lim.age_group_id.values)

    locations = pop.location_id.values
    if location_idx is not None:
        try:
            locations = [locations[location_idx]]
            LOGGER.info("Using location_id {} from location_idx {}".format(
                locations, location_idx
            ))
        except IndexError:
            LOGGER.warning("Asked for out-of-bounds location {} of {}".format(
                location_idx, locations.shape[0]
            ))
            exit(0)  # Maybe you ask for 200 jobs but have 195 countries. OK.
    elif location_id is not None:
        locations = [location_id]
    else:
        locations = pop.location_id.values

    for location in locations:
        begin_time = perf_time()
        loc_idx = dict(location_id=location)

        future = one_location(
            pop.loc[loc_idx],
            asfr_lim.loc[loc_idx],
            lifetable_lim.loc[loc_idx],
            migration.loc[loc_idx],
            srb.loc[loc_idx],
            ruler,
            gbd_round_id,
            years,
            test
        )
        out_name = out_path / "{}.nc".format(location)
        future.coords["location_id"] = location
        summary = summarize_pop(future)
        elapsed = perf_time() - begin_time
        LOGGER.info("Elapsed {}".format(elapsed))
        write_begin = perf_time()
        save_xr(summary, out_name, metric="number", space="identity",
                death=version, pop=pop_version, asfr=asfr_version,
                lifetable=lifetable_version, migration=migration_version,
                srb=srb_version)
        LOGGER.info("Wrote {}".format(out_name))
        LOGGER.info("Write time Elapsed {}".format(perf_time() - write_begin))


def summarize_pop(population):
    """
    Makes summary values for calculated populations.
    Naming of the quantile coordinates follows conventions
    for ``scenario-bokeh-app``, which uses Bokeh.

    Args:
        population (xr.DataArray): Future population

    Returns:
        xr.Dataset: With variables for summary values.
    """
    LOGGER.debug("population to summarize {}".format(population))
    dim_order = ["year_id", "scenario", "sex_id", "age_group_id", "draw"]
    pop = population.transpose(*dim_order)
    percentiles = [5, 95, 50]
    percentile_by_age = np.percentile(
        a=pop.values,
        q=np.array(percentiles),
        axis=len(dim_order) - 1
    )
    LOGGER.debug("pop shape {} percentile shape {}".format(
        pop.shape, percentile_by_age.shape))
    dim_order.insert(0, "quantile")
    dim_order.remove("draw")
    assert percentile_by_age.shape[0] == len(percentiles)
    by_age = xr.DataArray(
        percentile_by_age,
        coords={"scenario": pop.scenario.values,
                "year_id": pop.year_id.values,
                "age_group_id": pop.age_group_id.values,
                "sex_id": pop.sex_id.values,
                "quantile": 0.01 * np.array(percentiles)},
        dims=dim_order
    )
    total = by_age.sum("age_group_id")
    ds = xr.Dataset(
        {"population": pop, "binned": by_age, "total": total})
    ds["mean"] = ds["population"].mean("draw")
    return ds


def one_location(pop_start, asfr, lifetable, migration, srb, ruler,
                 gbd_round_id, years, test=False):
    """
    Calculates population for one location. The code is split this way
    in order to simplify exploration of different techniques. It might be
    faster to vectorize across populations, but this does not allow that.
    The incoming data is in XArray, and the outgoing data is in XArray,
    but this function translates the XArray into individual Numpy ndarrays
    and sends it to the function that does the actual computation.

    Args:
        pop_start (xr.DataArray): Initial year of population.
        asfr (xr.DataArray): Age-specific fertility rate.
        lifetable (xr.Dataset): The complete lifetable.
        migration (xr.DataArray): Migration per country.
        srb (xr.DataArray): Sex ratio at birth per country
        ruler (single.Ruler): Tracks time intervals.
        gbd_round_id (int): gbd round id.
        years (YearRange): years for past and forecast.
        test (bool): Whether to run on just a subset.

    Returns:
        xr.DataArray: Future populations for this location.
    """
    pop = xr.zeros_like(lifetable.lx)
    LOGGER.debug("pop entry year_id {}".format(pop.year_id.values))

    all_draws = list(set(list(asfr.draw.values)) &
                     set(list(lifetable.draw.values)))
    if test:
        all_draws = all_draws[:10]
        pop = pop.loc[{"draw": all_draws}]

    pop.loc[{"year_id": years.past_end}] = pop_start

    year_ruler = timeline(
        migration.age_group_id.values,
        migration.age_group_id.values)

    # need extrapolated lx values within terminal age group
    # to compute survivorship for the terminal age group
    lx_extended = model.demography_team_extrapolation_of_lx(lifetable["mx"],
                                                            lifetable["ax"])
    ruler_extended = timeline(lx_extended["age_group_id"].values,
                              lx_extended["age_group_id"].values)

    for scenario_idx in asfr.scenario.values:
        for draw_idx in all_draws:
            idx = dict(scenario=scenario_idx, draw=draw_idx)

            asfr_draw = asfr.sel(**idx)

            if "draw" in pop_start.dims:
                pop_start_draw = pop_start.sel(draw=draw_idx)
            else:
                pop_start_draw = pop_start

            LOGGER.debug("asfr_draw dims {}".format(asfr_draw.dims))
            future_years = dict(idx)
            future_years["year_id"] = years.forecast_years
            future_years["age_group_id"] = pop.age_group_id.values

            lx = lx_extended.sel(**idx)

            migrate_years = migration.year_id.values
            only_years = migrate_years[migrate_years >= years.past_end]
            migration_future = migration.sel(year_id=only_years)
            # migration might have draws as well now
            if "draw" in migration_future.dims:
                migration_future = migration_future.sel(**draw_idx)

            projected = project_draw(
                    asfr_draw.values,
                    lx.values,
                    migration_future.values,
                    pop_start_draw.values,
                    srb.values,
                    ruler,
                    ruler_extended,
                    year_ruler,
                    years.forecast_years
                )
            LOGGER.debug("projected shape: {} to {}".format(
                    projected.shape, future_years))
            LOGGER.debug("pop coords {}\njust years {}".format(
                    pop.coords, pop.year_id.values))
            LOGGER.debug("shape of now")
            LOGGER.debug("shape of future {}".format(
                    pop.loc[future_years].shape))
            if np.isnan(projected).sum() or np.isinf(projected).sum():
                threshold = np.get_printoptions()["threshold"]
                np.set_printoptions(threshold=np.nan)
                LOGGER.error("scenario {} draw {} nan or inf".format(
                        scenario_idx, draw_idx))
                LOGGER.error("input lx {} {}".format(lx.shape, lx))
                LOGGER.error("input pop {} {}".format(
                        pop_start_draw.shape, pop_start_draw.values))
                LOGGER.error("input asfr {} {}".format(
                        asfr_draw.shape, asfr_draw.values))
                # LOGGER.error("input mig {}".format(migration_future.values))
                LOGGER.error("output pop {} {}".format(
                        projected.shape, projected))
                np.set_printoptions(threshold=threshold)
            pop.loc[future_years] = projected

    check_out_of_bounds(pop, "pop_out")
    return pop


def agreement_rules(asfr, lifetable, pop, migration, srb, years):

    """
    This is where we put all rules for how data from different sources
    must agree with each other in terms of what domain it lives on.

    Args:
        asfr (xr.DataArray): Age-specific fertility rate
        lifetable (tuple): The lifetable as a tuple of datasets, past and future
        pop (xr.DataArray): Population for starting year
        migration (xr.DataArray): Migration values
        srb (xr.DataArray): Sex ratio at birth
        years (YearRange): years for past and forecast.

    Returns:
        xr.DataArray: ASFR
        xr.Dataset: Lifetable
        xr.DataArray: First year of population
        xr.DataArray: Migration
    """
    # There could be a separate past and future lifetable, so we take part
    # in order to determine the subsets. Take the second one, the future.
    part_of_lifetable = lifetable[-1]

    desired_locations = {int(dl) for dl in
                         fbd_core.db.get_locations_by_level(3).location_id}
    la, ll, lp, lm = [set(l.location_id.values) for l
                      in [asfr, part_of_lifetable, pop, migration]]
    LOGGER.info("location id count for asfr {} life {} pop {} migration {}"
                .format(len(la), len(ll), len(lp), len(lm)))
    loc_all = la & ll & lp  # Missing migration will be set to 0.
    if desired_locations - la:
        LOGGER.warning("asfr locations missing {}".format(
            desired_locations - la))
    if desired_locations - ll:
        LOGGER.warning("lifetable locations missing {}".format(
            desired_locations - ll))
    if desired_locations - lp:
        LOGGER.warning("population locations missing {}".format(
            desired_locations - lp))
    if desired_locations - lm:
        LOGGER.warning("migration locations missing {}".format(
            desired_locations - lm))
    subset = dict(location_id=np.array(list(sorted(loc_all)), dtype=np.int))

    subset["sex_id"] = [1, 2]

    assert not (set(asfr.age_group_id.values) -
                set(part_of_lifetable.age_group_id.values))

    ages = part_of_lifetable.age_group_id.values[:]
    al = set(part_of_lifetable.age_group_id.values)
    ap = set(pop.age_group_id.values)
    if al ^ ap:
        LOGGER.info("lifetable ages {} pop ages {}".format(al - ap, ap - al))
        ages = [a for a in ages if a in ap]

    assert consistent_age_group_ids(ages),\
        "Ages don't match for ids {}".format(ages)
    subset["age_group_id"] = ages

    pop_sub = pop.loc[subset]
    subset_lives = list()

    if len(lifetable) > 1:  # should be just 2 elements
        min_draw_count = min([ds["draw"].size for ds in lifetable])
        # make sure the subset draw labels are the same
        assert lifetable[0]["draw"][0:min_draw_count].\
            identical(lifetable[1]["draw"][0:min_draw_count])
        subset["draw"] = lifetable[0]["draw"][0:min_draw_count]

    for incoming_life in lifetable:
        LOGGER.debug("life subsets {}".format(incoming_life))
        subset_life = incoming_life.loc[subset]
        if "scenario" not in subset_life.dims:
            d3 = xr.concat([subset_life] * 3, dim="scenario")
            d3.coords["scenario"] = [-1, 0, 1]
            subset_life = d3
        subset_lives.append(subset_life)

    if len(subset_lives) > 1:
        lifetable_lim = xr.concat(subset_lives, dim="year_id")
    else:
        lifetable_lim = subset_lives[0]

    life_sub = lifetable_lim.transpose("location_id", "scenario", "draw",
                                       "year_id", "sex_id", "age_group_id")

    assert consistent_age_group_ids(life_sub.age_group_id.values)
    assert consistent_age_group_ids(pop_sub.age_group_id.values)

    # Migration will be missing locations, most likely, and may have extras.
    # Migration will have strict five-year age groups.
    # This version creates a dataset of zeros. We should give missing countries
    # a value that is the average over the region.
    migration_years = migration.year_id.values
    migration_years = migration_years[migration_years >= years.past_end]

    if np.in1d(subset["age_group_id"], migration.age_group_id.values).all():
        migration_ages = subset["age_group_id"]
        LOGGER.info("migration using GBD age groups")
    else:
        migration_ages = migration.age_group_id.values
        LOGGER.info("migration using the age groups it has")

    migration_sub = xr.DataArray(
        data=np.zeros((len(subset["location_id"]),
                       len(migration_years),
                       2,
                       len(migration_ages)),
                      dtype=np.double
                      ),
        coords=dict(
            location_id=subset["location_id"],
            year_id=migration_years,
            sex_id=[1, 2],
            age_group_id=migration_ages
        ),
        dims=["location_id", "year_id", "sex_id", "age_group_id"]
    )

    common_locations = [l for l in subset["location_id"]
                        if l in migration.location_id.values]
    copy_idx = dict(location_id=common_locations,
                    year_id=migration_years,
                    sex_id=[1, 2],
                    age_group_id=migration_ages)
    LOGGER.debug("migration {} migration_sub {}".format(
        migration.age_group_id.values, migration_sub.age_group_id.values
    ))

    if "draw" in migration.dims:
        migration_sub = expand_dimensions(migration_sub, draw=migration["draw"])
        migration_sub = migration_sub.transpose(*list(migration.dims))
        copy_idx["draw"] = migration["draw"].values.tolist()

    for index_obj in copy_idx.values():
        assert len(set(index_obj)) == len(index_obj)
    for name, coord in migration.coords.items():
        assert len(set(coord.values)) == len(coord.values), name

    LOGGER.debug("migration {}".format(migration))

    migration_sub.loc[copy_idx] = migration.loc[copy_idx]

    if 1 in migration_sub.age_group_id.values:
        LOGGER.info("Migration WPP Straight")
        from_under_five = copy_idx.copy()
        from_under_five["age_group_id"] = [1]
        five_year = 1 / (5 * 365)
        for aid, frac in [(2, 7 * five_year), (3, 21 * five_year),
                          (4, 337 * five_year), (5, 4 * 365 * five_year)]:
            copy_idx["age_group_id"] = [aid]
            migration_sub.loc[copy_idx] = frac * migration.loc[from_under_five]
        migration_sized = migration_sub
    else:
        LOGGER.info("Migration WPP Smoothed")
        # It's the one-year migration.

        # NOTE these age groups were particular of the migration file provided
        migration_age_ids = migration_sub['age_group_id'].values
        end_ages_0 = [143, 144, 145, 146, 273]  # 95, 96, 97, 98, 99+
        end_ages_1 = [235]  # 95+
        if set(end_ages_0).issubset(migration_age_ids):
            assert not set(end_ages_1) & set(migration_age_ids)
            end_ages = end_ages_0
            early_ages = [x for x in migration_sub.age_group_id.values
                          if x not in end_ages]
            # sum over these granular age groups to form one terminal age group
            end_years = migration_sub.loc[{"age_group_id": end_ages}].sum(
                dim="age_group_id"
            )
            end_years.coords["age_group_id"] = 235
            end_years.expand_dims("age_group_id", axis=len(end_years.dims))
            lop_end = migration_sub.loc[{"age_group_id": early_ages}]
            migration_sized = xr.concat([lop_end, end_years],
                                        dim="age_group_id")
        # if 235 is the only age group id beyond 142, no change is needed
        elif set(end_ages_1).issubset(migration_age_ids):
            assert not set(end_ages_0) & set(migration_age_ids)
            migration_sized = migration_sub
        else:
            raise Exception("end_ages do not exists in migration age_group_id")

    ordered =\
        consistent_age_group_ids(migration_sized.age_group_id.values)
    if not ordered:
        raise RuntimeError("Age group ids not ordered")

    subset.pop("sex_id")
    subset.pop("age_group_id")
    asfr_sub = asfr.loc[subset]
    # We assert order because the internal methods will use numpy arrays.
    # Locations are first because we will parallelize over locations.
    # The year and age group are last because we work by draw, so they
    # are needed for each forecast.
    asfr_sub = asfr_sub.transpose("location_id", "scenario", "draw",
                                  "year_id", "age_group_id")

    idx = 0
    for year_arr in [asfr_sub, life_sub, migration_sized]:
        assert year_arr.year_id.values[0] == years.past_end, (
            "Start is {} years for {} are {}".format(
                years.past_end, idx, year_arr.year_id.values)
        )
        idx += 1

    LOGGER.debug("Pop from agree years {}".format(pop_sub.year_id.values))
    LOGGER.debug("Life from agree years {}".format(life_sub.year_id.values))
    check_out_of_bounds(asfr_sub, "asfr")
    check_out_of_bounds(life_sub.lx, "life")
    check_out_of_bounds(pop_sub, "pop_in")
    check_out_of_bounds(migration_sub, "migration")

    return asfr_sub, life_sub, pop_sub, migration_sized, srb


def check_out_of_bounds(dataset, name):
    nan_cnt = dataset.size - dataset.count()
    if nan_cnt > 0:
        LOGGER.error("Nan count {} in {}".format(nan_cnt, name))
    inf_cnt = int(np.where(np.isnan(dataset))[0].size)
    if inf_cnt > 0:
        LOGGER.error("Inf count {} in {}".format(inf_cnt, name))


def read_datasets(
            asfr_version, gbd_round_id, lifetable_version,
            pop_version, migration_version, years, srb_version, draws):
    """
    This reads files, orders their axes, and ensures that data arrays
    aren't presented as datasets. This enforces rules about how many
    files get read, how they are found, and how they are assembled into
    the incoming data. It doesn't address what the data means.

    Args:
        asfr_version (str): Version string for ASFR
        gbd_round_id (int): GBD Round as an integer
        lifetable_version (list[str]): Lifetable version
        pop_version (str): Population start version
        migration_version (list[str]): Migration version
        years (YearRange): years for past and forecast
        srb_version (str): sex ratio at birth version
        draws (int): the number of draws to take from the future versions.

    Returns:
        xr.DataArray: ASFR
        tuple: Either one lifetable file or (past, futue).
        xr.DataArray: Starting population
        xr.DataArray: Migration
        xr.DataArray: SRB
    """
    # Do this in a subroutine so it's memory can be released.
    # pop etl (pop version is in the past)
    data_read_start = perf_time()
    pop_file = FBDPath("/{}/past/population/{}".format(
        gbd_round_id, pop_version)) / "population.nc"
    try:
        LOGGER.info("Reading {}".format(pop_file))
        pop = xr.open_dataarray(str(pop_file))
        # if there's a draw dimension, take the mean
        if "draw" in pop.dims:
            pop = pop.mean("draw")
    except OSError as ose:
        LOGGER.error("Cannot open pop {}: {}".format(pop_file, ose))
        exit()

    # we may or may not have draws for past pops, but we should certainly
    # expect location, age, sex, and year
    assert {"location_id", "year_id", "age_group_id",
            "sex_id"}.issubset(set(pop.dims))
    if len(pop.year_id) > 1:
        pop = pop.loc[{"year_id": years.past_end}]
    else:
        pop = pop.squeeze(dim="year_id")
        assert pop.year_id == years.past_end
    LOGGER.debug("pop {}".format(pop))

    # we like age_group_id to be the last dim to expedite later computation.
    if "draw" in pop.dims:  # if past pop has draws, resample.
        pop = pop.transpose("draw", "location_id", "sex_id", "age_group_id")
        pop = resample(pop, draws)
    else:
        pop = pop.transpose("location_id", "sex_id", "age_group_id")

    if pop.name is None:
        pop.name = "population"

    # asfr etl (draws expected)

    asfr_gbd_round_id = gbd_round_id if gbd_round_id >= 5 else 5
    asfr_file = FBDPath("/{}/future/asfr/{}".format(
        asfr_gbd_round_id, asfr_version
    )) / "asfr.nc"
    try:
        LOGGER.info("Reading {}".format(asfr_file))
        # ASFR is reported per thousand people.
        asfr = xr.open_dataarray(str(asfr_file))
    except OSError as ose:
        LOGGER.error("Cannot open asfr {}: {}".format(asfr_file, ose))
        #exit(2

    assert set(asfr.dims) == {
        "draw", "year_id", "location_id", "scenario", "age_group_id"
    }, "asfr dims {}".format(asfr.dims)
    asfr_lim = asfr.sel(year_id=slice(years.past_end, years.forecast_end + 1))
    if asfr_lim.name is None:
        asfr_lim.name = "asfr"

    asfr_lim = resample(asfr_lim, draws)

    # lifetable etl (draws expected)
    lifetables = list()
    for lfilename in lifetable_version:
        lifetables.append(read_lifetable(gbd_round_id, lfilename, draws))
    if len(lifetables) > 1:
        lpast, lfuture = (None, None)
        lyears = [llx.year_id.values for llx in lifetables]
        if lyears[0][-1] > lyears[1][-1]:
            lfuture, lpast = lifetables
        elif lyears[1][-1] > lyears[0][-1]:
            lpast, lfuture = lifetables
        elif lyears[0][0] < lyears[1][0]:
            lpast, lfuture = lifetables
        elif lyears[1][0] < lyears[0][0]:
            lfuture, lpast = lifetables
        else:
            LOGGER.error("Cannot figure out which is the future lifetable")
            exit()

        if years.past_end in lfuture.year_id.values:
            LOGGER.info("All needed years were in the future lifetable"
                        "Ignoring the past data.")
            lifetable_lim = lfuture.sel(
                    year_id=slice(years.past_end, years.forecast_end + 1))
            lifetable_out = (lifetable_lim,)
        else:
            assert years.past_end in lpast.year_id.values
            past_slice = lpast.loc[{"year_id": [years.past_end]}]
            LOGGER.debug("Life past slice {}".format(
                    past_slice.year_id.values))
            LOGGER.debug("Life future slice {}".format(
                    lfuture.year_id.values))
            lifetable_out = (past_slice, lfuture)
    else:
        lifetable_lim = lifetables[0].sel(
                year_id=slice(years.past_end, years.forecast_end + 1))

        lifetable_out = (lifetable_lim,)

    # migration etl (no draws expected)
    try:
        migration_file = FBDPath("/{}/future/migration/{}".format(
            gbd_round_id, migration_version[0]
        )) / "migration.nc"
    except Exception:
        if os.path.exists(migration_version[0]):
            migration_file = migration_version[0]
        else:
            raise Exception("Cannot construct {}".format(migration_file))

    try:
        LOGGER.info("Reading {}".format(migration_file))
        migration = xr.open_dataarray(str(migration_file))
    except OSError as ose:
        LOGGER.error("Cannot open migration {}: {}".format(
            migration_file, ose
        ))
        exit()
    assert set(("location_id", "age_group_id", "sex_id", "year_id")).\
           issubset(migration.dims)

    # Currently we don't use or make migration scenarios -- if a scenario dim
    # exists for some reason ensure that only reference is used and that the
    # scenario dim is dropped.
    if "scenario" in migration.dims:  # scenario dim
        migration = migration.sel(scenario=0, drop=True)
    elif "scenario" in migration.coords:  # scenario point coord
        migration = migration.drop("scenario")
    else:
        pass  # no scenario dim or point coord

    # if pop has draws, we want migration to have draws as well.
    # this becomes important in _fill_missing_locations().
    if "draw" in pop.dims:
        if "draw" not in migration.dims:
            migration = expand_dimensions(migration, draw=pop["draw"])
        else:
            migration = resample(migration, draws)
        migration = migration.transpose("draw", "location_id", "year_id",
                                        "sex_id", "age_group_id")
    else:  # pop has no "draw", so migration doesn't need it either
        if "draw" in migration.dims:
            migration = migration.mean("draw")
        migration = migration.transpose("location_id", "year_id", "sex_id",
                                        "age_group_id")

    if migration.name is None:
        migration.name = "migration"
    # Use the last past year's all age population proportions to compute
    # regional migration averages to fill in missing data.
    migration_locs_fixed = _clean_migration_locations(
            migration, pop.sum("age_group_id"), gbd_round_id)

    LOGGER.info("Read data Elapsed {}".format(
            perf_time() - data_read_start))

    # Migration counts drive small nations to zero population.
    # This is a way to ensure we show the trend of health.
    migration_locs_fixed.loc[dict(
        location_id=list(SMALL_NATIONS_ZERO_MIGRATION.values())
    )] = 0.

    LOGGER.debug("Pop from read years {}".format(pop.year_id.values))

    # Not FBDPath at the moment since it doesn't recognize covariate as a
    # valid stage. May need to change location of files.
    # srb etl (no draws)
    srp_path = FBDPath(
        "/{}/past/sex_ratio_at_birth/{}".format(gbd_round_id, srb_version))
    srb_file = srp_path / "sex_ratio_at_birth.nc"

    try:
        LOGGER.info("Reading {}".format(srb_file))
        srb = xr.open_dataarray(str(srb_file))
    except OSError as ose:
        LOGGER.error("Cannot open srb {}: {}".format(
            srb_file, ose
        ))
        exit()

    # Subset to last year of past
    srb = srb.sel(year_id=years.past_end)

    return asfr_lim, lifetable_out, pop, migration_locs_fixed, srb


def _clean_migration_locations(migration, pop, gbd_round_id):
    """Migration uses weird locations. Sometimes, locations are missing
    migration data. Other times, locations have migration data but they
    shouldn't.

    In the case where locations have migration data, but they should really be
    part of another location (e.g. Macao is part of China), that migration will
    be added into the "parent" location.

    In the case where locations are missing migration data, those locations
    will get the average migration of their regions. This averaging happens
    AFTER too-specific locations are merged into their parents.
    """
    merged_migration = _merge_too_specific_locations(migration)
    filled_migration = _fill_missing_locations(
            merged_migration, pop, gbd_round_id)
    return filled_migration


def _fill_missing_locations(data_per_capita, pop, gbd_round_id):
    """Missing locations need to be filled in with region averages."""
    avail_locs = set(data_per_capita.location_id.values)
    desired_locs = fbd_core.db.get_modeled_locations(gbd_round_id)
    missing_locs = set(desired_locs.location_id.values) - avail_locs
    if not missing_locs:
        return data_per_capita
    LOGGER.info("These locations are missing: {}".format(missing_locs))
    parent_locs = desired_locs.query("location_id in @missing_locs")[
            "parent_id"].values
    LOGGER.info("Children of these locations will be averaged to fill in "
                "missing data: {}".format(parent_locs))
    hierarchy = desired_locs.query(
            "parent_id in @parent_locs and location_id in @avail_locs"
        )[
            ["location_id", "parent_id"]
        ].set_index(
            "location_id"
        ).to_xarray()["parent_id"]
    hierarchy.name = "location_id"
    pop_location_slice = pop.sel(location_id=hierarchy.location_id.values)

    data = data_per_capita * pop_location_slice

    mean_data = data.sel(
            location_id=hierarchy.location_id.values
            ).groupby(hierarchy).mean("location_id")
    pop_agged = pop_location_slice.sel(
            location_id=hierarchy.location_id.values
            ).groupby(hierarchy).mean("location_id")
    mean_data_per_capita = (mean_data / pop_agged).fillna(0)
    location_da = xr.DataArray(
            desired_locs.location_id.values,
            dims="location_id",
            coords=[desired_locs.location_id.values])

    filled_data_per_capita, _ = xr.broadcast(data_per_capita, location_da)

    for missing_location in desired_locs.query(
            "location_id in @missing_locs").iterrows():
        loc_slice = {"location_id": missing_location[1].location_id}
        loc_parent_slice = {"location_id": missing_location[1].parent_id}
        filled_data_per_capita.loc[loc_slice] = (
                mean_data_per_capita.sel(**loc_parent_slice))
    match_already_existing_locations = (
            filled_data_per_capita == data_per_capita).all()
    does_not_match_err_msg = (
           "Result should match input data for locations that are present.")
    assert match_already_existing_locations, does_not_match_err_msg
    if not match_already_existing_locations:
        LOGGER.error(does_not_match_err_msg)
        raise MigrationError(does_not_match_err_msg)
    has_new_locations = missing_locs.issubset(
            filled_data_per_capita.location_id.values)
    does_not_have_new_locs_err_msg = (
           "Missing locations {} are still missing.".format(missing_locs))
    assert has_new_locations, does_not_have_new_locs_err_msg
    if not has_new_locations:
        LOGGER.error(does_not_have_new_locs_err_msg)
        raise MigrationError(does_not_have_new_locs_err_msg)
    return filled_data_per_capita


def _merge_too_specific_locations(data):
    """Locations that are too specific (i.e. level 4) need to be merged into
    their respective parent locations."""
    avail_locs = set(data.location_id.values)
    level_four_locs = fbd_core.db.get_locations_by_level(4)
    too_specific_locs = set(level_four_locs.location_id) & avail_locs
    if len(too_specific_locs) == 0:  # nothing to merge to parent.  just return input.
        return data
    LOGGER.info("These locations are too specific: {}".format(
        too_specific_locs))
    children_into_parents = level_four_locs.query(
            "location_id in @too_specific_locs"
        )[
            ["location_id", "parent_id"]
        ].set_index(
            "location_id"
        ).to_xarray()["parent_id"]
    children_into_parents.name = "location_id"
    children_merged = data.sel(
            location_id=children_into_parents.location_id
        ).groupby(children_into_parents).sum("location_id")
    good_locations = avail_locs - too_specific_locs
    good_data = data.sel(location_id=list(good_locations))
    merged_data = sum(broadcast_and_fill(children_merged, good_data,
                                         fill_value=0))
    unchanged_locs = good_locations - set(children_into_parents.values)
    good_locs_didnt_change = (
        data == merged_data.sel(location_id=list(unchanged_locs))).all()
    good_locs_did_change_err_msg = (
           "Error: good locations were changed during the merge.")
    assert good_locs_didnt_change, good_locs_did_change_err_msg
    if not good_locs_didnt_change:
        LOGGER.error(good_locs_did_change_err_msg)
        raise MigrationError(good_locs_did_change_err_msg)
    return merged_data


def _drop_point_coordinates(ds):
    point_coordinates = [pc for pc in ds.coords.keys() if pc not in ds.dims]
    return ds.drop(point_coordinates)


def read_lifetable(gbd_round_id, lifetable_version, draws):
    """

    Args:
        gbd_round_id (int):
        lifetable_version (str): Of the form "past/versionname"
           or "future/versionname"
        draws (int): desired number of draws

    Returns:
        The life table.
    """
    if "/" in lifetable_version:
        past_or_future, version = lifetable_version.split("/")
    else:
        past_or_future = "future"
        version = lifetable_version

    # lifetable from the future includes last year of the past.
    lifetable_file = FBDPath("/{}/{}/life_expectancy/{}".format(
        gbd_round_id, past_or_future, version
    )) / "lifetable_ds.nc"
    try:
        LOGGER.info("Reading {}".format(lifetable_file))
        lifetable = xr.open_dataset(str(lifetable_file))
    except OSError as ose:
        LOGGER.error("Cannot open lifetable {}: {}".format(
            lifetable_file, ose))
        exit()

    if "draw" in lifetable.dims:
        lifetable = resample(lifetable.sortby("draw"), draws)

    return _drop_point_coordinates(lifetable)


if __name__ == "__main__":
    version_today = datetime.datetime.now().strftime("%Y%m%d")

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True,
                        help="version for output")
    # Use action=append in case one ASFR file is for the future
    # and another has that last year of the past that we need.
    parser.add_argument("--asfr", required=True,
                        help="Age-specific fertility rate")
    parser.add_argument("--pop", required=True, help="Starting population")
    parser.add_argument("--lifetable", action="append", required=True,
                        help="Life table with lx and nLx")
    parser.add_argument("--migration", action="append", required=True,
                        help="Migration")
    parser.add_argument("--srb", type=str, required=True,
                        help="Sex ratio at birth")
    parser.add_argument("--location-idx", type=int, default=None,
                        dest="location_idx",
                        help=("Zero-based index into the list of locations. "
                              "If this is specified, only do that location."))
    parser.add_argument("--location-id", type=int, default=None,
                        dest="location_id", help="Location ID")
    parser.add_argument("--test", action="store_true",
                        default=False,
                        help="Run a small subset as a smoke test")
    parser.add_argument("--data-path", type=Path,
                        default=Path("/ihme/forecasting/data"),
                        help="Set new base path for directories.")
    parser.add_argument("--gbd-round-id", type=int, required=True,
                        help="GBD round id")
    parser.add_argument("--draws", type=int, required=True,
                        help="Number of draws")
    parser.add_arg_years(required=True)

    args, _ = parser.parse_known_args()

    new_roots = dict(
        data=args.data_path,
    )
    with roots.RootDirectory.update_roots(**new_roots):
        run_against(
            args.version, args.pop, args.asfr, args.lifetable, args.migration,
            args.srb, args.gbd_round_id, args.location_idx, args.years,
            args.location_id, args.draws, args.test
        )

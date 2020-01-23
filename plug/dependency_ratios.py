'''In 2100, if labour force participation by age and sex does not change,
the ratio of the non-working adult population to the working population could
reach 1.13 globally, up from 0.80 in 2017. This implies that at the global
level each person working would have to support, through taxation and
intra-family income transfers, 1.13 individuals age 15 years and above.
Moreover, the number of countries with this dependency ratio above 1 is
expected to increase from 59 in 2017 to 140 in 2100. '''

import xarray as xr
import pandas as pd
import numpy as np
import glob

from fbd_core import db
from fbd_core.file_interface import FBDPath, open_xr
from fbd_core.etl import expand_dimensions

import settings

gbd_round_id = settings.BASELINE_VERSIONS["population_mean_ui"].gbd_round_id

# lex versions
PAST = settings.PAST_VERSIONS["population"].version
REF = settings.BASELINE_VERSIONS["population_mean_ui"].version
# REF = "20190808_sdg_ref_15_agg_combined" # first submission version

YOUNG_AGES = range(2, 8, 1)
OLD_AGES = (19, 20, 21)
WORKING_AGES = range(8, 19, 1) # this is special for this calculation only


def _etl_total_emp():
    location_metadata = db.get_locations_by_max_level(3)
    location_hierarchy = location_metadata.set_index(
        "location_id").to_xarray()["parent_id"]

    total_emp_files = \
        ["/ihme/covariates/ubcov/model/output/54629/draws_temp_0/",#male
        "/ihme/covariates/ubcov/model/output/54758/draws_temp_0/"]#female
    total_emp = pd.DataFrame()
    col_list = ["location_id", "year_id", "age_group_id", "sex_id", "mean_emp"]
    for file in total_emp_files:
        for location_csv in glob.glob(file + "*.csv"):
            temp_csv = pd.read_csv(location_csv)
            temp_csv["mean_emp"] = temp_csv.filter(like="draw_").mean(axis=1)
            total_emp = total_emp.append(temp_csv[col_list].query(
                "location_id in @location_hierarchy.location_id"))

    total_emp_ds = total_emp.set_index(["location_id", "year_id", "sex_id",
                                 "age_group_id"]).to_xarray().mean_emp
    assert xr.ufuncs.isfinite(total_emp_ds).all()
    return total_emp_ds


def get_total_emp():
    total_emp = xr.open_dataarray(
        '/ihme/forecasting/working/trobalik/data/20191209/total_employment.nc')

    return total_emp


def load_pop(gbd_round_id, past_version, forecast_version):

    forecast_file = FBDPath(
        f"/{gbd_round_id}/future/population/"
        f"{forecast_version}/population_combined.nc")

    past_file = FBDPath(
        f"/{gbd_round_id}/past/population/{past_version}/population.nc")

    future_pop = open_xr(forecast_file).data
    past_pop = expand_dimensions(open_xr(past_file).data,
                                 scenario=future_pop.scenario,
                                 quantile = future_pop["quantile"])
    pop = xr.concat([past_pop, future_pop],
                    "year_id")

    return pop


def dep_ratio(pop, total_emp, working_ages, old_ages):
    '''
    This calculates the dependency ratio for the old population plus the
    population within the working age groups that are unemployed (as
    determined by the total employment adjustment).
    '''

    wp = pop.sel(year_id=[2017, 2100],
                 age_group_id=list(working_ages)) * \
         total_emp.sel(year_id=2017)
    wp_sum = wp.sum("age_group_id")

    non_wp = pop.sel(age_group_id=list(old_ages)).sum("age_group_id") + \
             (pop.sel(age_group_id=list(working_ages)) * \
              (1 - total_emp.sel(year_id=2017))).sum("age_group_id")

    ratio = non_wp.sum(["sex_id", "location_id"]) / wp_sum.sum(
        ["sex_id", "location_id"])

    print(f"Global 2017: {ratio.sel(year_id=2017).round(2).values.item()}, \n" 
          f"Global 2100: {ratio.sel(year_id=2100).round(2).values.item()}")


def dep_ratio_above_one(pop, total_emp, working_ages, old_ages):
    wp = pop.sel(year_id=[2017, 2100],
                 age_group_id=list(working_ages)) * \
         total_emp.sel(year_id=2017)
    wp_sum = wp.sum("age_group_id")

    non_wp = pop.sel(age_group_id=list(old_ages)).sum("age_group_id") + \
             (pop.sel(age_group_id=list(working_ages)) * \
              (1 - total_emp.sel(year_id=2017))).sum("age_group_id")

    ratio = non_wp.sum(["sex_id"]) / wp_sum.sum(
        ["sex_id"])

    above1_2017 = len(ratio.sel(year_id=2017) \
                      .where(ratio.sel(year_id=2017) > 1) \
        .dropna("location_id"))

    above1_2100 = len(ratio.sel(year_id=2100) \
                      .where(ratio.sel(year_id=2100) > 1) \
                      .dropna("location_id"))

    print(f"Moreover, the number of countries with this dependency ratio above "
          f"1 is expected to increase from {above1_2017} in 2017 to "
          f"{above1_2100} in 2100.")


def china_decline(pop):
    china = pop.sel(location_id=6, age_group_id=9, sex_id=3, scenario=0)

    decline = np.round(
        (china.sel(year_id=2100) - china.sel(year_id=2017)) / china.sel(
            year_id=2017), 2)

    lower = decline.sel(quantile="lower").item()
    mean = decline.sel(quantile="mean").item()
    upper = decline.sel(quantile="upper").item()

    print(f"The {mean}% decline (95% UI {lower}% to {upper}%) China’s "
          f"population aged 20–24 is a factor that cannot be ignored when "
          f"considering possible shifts in global power later in the century")



if __name__ == "__main__":

    total_emp = get_total_emp()
    pop = load_pop(gbd_round_id, PAST, REF)
    pop_slice = pop.sel(scenario=0, quantile="mean", drop=True)

    dep_ratio(pop_slice, total_emp, WORKING_AGES, OLD_AGES)
    dep_ratio_above_one(pop_slice, total_emp, WORKING_AGES, OLD_AGES)
    china_decline(pop)


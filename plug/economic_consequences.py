"""
Figure 11 shows the number of working age individuals (from 20 to 64 years) for
the 10 largest countries in 2017, in the reference forecast. Huge declines are
expected in the number of workers in China and India and steady increases in
Nigeria. By 2100, India will still have the largest working age population
followed by Nigeria and China. Coming in fourth in the world will be the United
States. In our reference scenario, despite below-replacement fertility,
in-migration sustains the US workforce. Figure 12 translates these forecasts of
numbers of working aged adults into scenarios for total GDP showing the rank
order of the top 25 national economies in 2017, 2030, 2050 and 2100 in the
reference forecast. China rises to the top in XX year in the reference scenario
but then is superseded by the US again in XX as population decline curtails
economic growth. Other countries bolstered by in-migration that rise up in the 
global rankings by GDP are Australia and Israel. Despite huge declines in
population expected this century, Japan remains the fourth largest economy in
2100. 
"""
import pandas as pd
import xarray as xr

from fbd_core import argparse, db, great_job
from fbd_core import great_job
from fbd_core.etl import resample
from fbd_core.etl.aggregator import Aggregator
from fbd_core.etl.transformation import expand_dimensions
from fbd_core.file_interface import FBDPath, open_xr, save_xr

import settings

ALL_AGE_ID = 22
BOTH_SEX_ID = 3

N_DECIMALS = 2

COUNTRIES = db.get_locations_by_level(3).location_id.values.tolist()


def _helper_round(val, divide_by_this):
    return round(val/divide_by_this, N_DECIMALS)

def _location_id_to_name(location_ids):
    locs = db.get_locations_by_max_level(3)[['location_id', 'location_name']]
    locs = locs[locs['location_id'].isin(location_ids)]
    location_names = locs['location_name'].tolist()

    return location_names

def _add_location_name(df):
    locs = db.get_locations_by_max_level(3)[['location_id', 'location_name']]
    df = df.merge(locs, how='left', on='location_id')
    
    return df


# By 2100, India will still have the largest working age population followed by
# Nigeria and China. Coming in fourth in the world will be the United States
def working_age():
    pop_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['population_mean_ui'].gbd_round_id}/"
        f"future/population/"
        f"{settings.BASELINE_VERSIONS['population_mean_ui'].version}")
    pop_path = pop_dir / "population_combined.nc"
    # working age is 20-64 age id 163, but have to combine 9-17 to get 163
    working_age_ids = list(range(9, 18))
    pop_da = open_xr(pop_path).data.sel(
        sex_id=BOTH_SEX_ID, scenario=0, quantile="mean",
        age_group_id=working_age_ids, year_id=2100, location_id=COUNTRIES)
   
    pop_da = pop_da.sum('age_group_id')
    
    pop_df = pop_da.to_dataframe()
    pop_df = _add_location_name(pop_df)
    pop_df = pop_df.sort_values(by='value', ascending=False).reset_index()
    
    first = pop_df.location_name[0]
    second = pop_df.location_name[1]
    third = pop_df.location_name[2]
    fourth = pop_df.location_name[3]

    print(f"By 2100, {first} will still have the largest working age "
          f"population followed by {second} and {third}. Coming in fourth in "
          f"the world will be {fourth}.\n")


# China rises to the top in XX year in the reference scenario but then is
# superseded by the US again in XX as population decline curtails economic
# growth
def largest_gdp():
    # This is a duplicate method from the abstract
    gdp_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['gdp'].gbd_round_id}/"
        f"future/gdp/"
        f"{settings.BASELINE_VERSIONS['gdp'].version}")
    gdp_path = gdp_dir / "gdp.nc"
    gdp_da = open_xr(gdp_path).data.sel(scenario=0)
    
    max_da = gdp_da.where(gdp_da==gdp_da.max('location_id'), drop=True)
    # Find 1st year where China is not nan (ie it is the max gdp for that year)
    china_year = max_da.sel(
        location_id=6).dropna(dim='year_id').year_id.values.item(0)
    # Subset to post-china_year years for when USA retakes the lead
    ok_usa_years = range(china_year, max_da.year_id.max().values.item(0) + 1)
    past_china_usa_da = max_da.sel(location_id=102, year_id=ok_usa_years)
    # Find 1st year where USA retakes lead
    usa_year = past_china_usa_da.dropna(dim='year_id').year_id.values.item(0)

    print(f"China rises to the top in {china_year} year in the reference "
          f"scenario but then is superseded by the US again in {usa_year} as "
          f"population decline curtails economic growth")

# Despite huge declines in population expected this century, Japan remains the
# fourth largest economy in 2100.
def japan_econ():
    gdp_dir = FBDPath(
        f"/{settings.BASELINE_VERSIONS['gdp'].gbd_round_id}/"
        f"future/gdp/"
        f"{settings.BASELINE_VERSIONS['gdp'].version}")
    gdp_path = gdp_dir / "gdp.nc"
    gdp_da = open_xr(gdp_path).data.sel(scenario=0, year_id=2100)
    
    gdp_df = gdp_da.to_dataframe()
    gdp_df = _add_location_name(gdp_df)
    gdp_df = gdp_df.sort_values(by='value', ascending=False).reset_index()

    japan_index = gdp_df[gdp_df['location_name']=="Japan"].index[0]
    japan_rank = japan_index + 1

    print(f"Despite huge declines in population expected this century, Japan "
          f"remains the {japan_rank} largest economy in 2100.\n")

if __name__ == "__main__":

    working_age()
    largest_gdp()
    japan_econ()


    great_job.congratulations()
"""
Population Other than the central Europe, eastern Europe, and central Asia 
super-region, where the population peaked in 1992, population size in the 
remaining super-regions will peak in a future before year 2100. Only the 
sub-Saharan Africa and north Africa and Middle East super-regions, at 3.07 
(2.48-3.83) billion and 978.78 (712.27-1411.30) million, respectively, will have
higher populations in 2100 than 2017. All super-regions except sub-Saharan 
Africa will experience substantial population declines in the coming seven 
decades. The declines are expected to be most severe in south Asia; southeast 
Asia, east Asia, and Oceania; and central Europe, eastern Europe, and central 
Asia.

Figure 10 shows the trajectory of total population for the ten largest countries
in 2017 to 2100 for all five scenarios. The reference forecasts for the five 
largest countries in 2100 are India (1067.26 [95% UI 703.97-1686.61] million), 
Nigeria (790.67 [597.08–1048.40] million), China 
(716.36 [437.11–1491.05] million), USA (334.69 [247.67–453.77] million), and DR 
Congo (245.94 [170.08–346.18] million). These forecasts, however, show different
future trajectories. Nigeria is forecasted with continued growth through 2100 
and is expected to be the 2^nd^ most populous country by then. The reference 
forecasts for China and India peak well before 2050 and both countries 
thereafter follow steep declining trajectories to a level of 50.01% 
(30.88-102.17) and 66.04% (47.07-95.53), respectively, of their peak 
populations. The USA is projected to experience population growth until 
mid-century, followed by only moderate decline of less than 10% by 2100. We 
forecast that the number of countries in sub-Saharan Africa among the countries 
with the 10 largest populations will increase from only Nigeria in 2017 to also 
include DR Congo, Ethiopia, and Niger in 2100.

We forecast large changes in age structures across the different super-regions 
(supplementary results figure X). Under the reference scenario, sub-Saharan 
Africa is to have only a moderate increase in the population <15 years of age, 
but large increases in working age population 20-64 and the elderly population. 
Under the slower pace scenario, the picture is dramatically different with a 
4-5-fold increase in the <15 year-old population. High-income countries are 
forecasted to accentuate the concave shape of age structures with strong 
absolute decreases in the population under 65 and absolute increases in the 65 
and older populations under all scenarios. The south Asia; southeast Asia, east 
Asia, and Oceania; and Latin America and Caribbean super-regions are forecasted 
to have an even more accentuated V-shaped age structures in 2100 under all 
scenarios. For all super-regions except sub-Saharan Africa and north Africa and 
Middle East, we forecast similar age structures under all scenarios. In these 
two super-regions and globally, the slower and faster paced scenarios on future 
trajectories for education and met need will result in dramatically different 
age structures in 2100. Among large countries, similar patterns are forecasted 
for Nigeria on one side, and China, India, and Indonesia on the other.
"""

import pandas as pd
import xarray as xr
import warnings

from fbd_core import db
from fbd_core.etl.transformation import expand_dimensions
from fbd_core.file_interface import FBDPath, open_xr, save_xr
import settings as sett

ALL_AGE_ID = 22
BOTH_SEX_ID = 3
GBD_ROUND = 5

N_DECIMALS = 2

COUNTRIES = db.get_locations_by_level(3).location_id.values.tolist()

PAST_POP_VERSION = sett.PAST_VERSIONS["population"].version
print(PAST_POP_VERSION)
FUTURE_POP_VERSION = sett.BASELINE_VERSIONS["population_mean_ui"].version
print(FUTURE_POP_VERSION)

def pull_tfr(future_tfr_version, scenarios = [-1,0,1]):
    tfr_path = FBDPath("{gbd_round_id}/future/tfr/{version}".format(
        gbd_round_id=GBD_ROUND, version=future_tfr_version))
    tfr = open_xr(tfr_path / "tfr_combined.nc").data.sel(
        scenario=scenarios)
    return tfr


def pull_pop(past_pop_version, future_pop_version,
             scenarios = 0, age_groups=ALL_AGE_ID):
    past_path = FBDPath("{gbd_round_id}/past/population/{version}".format(
        gbd_round_id=GBD_ROUND, version=past_pop_version))
    future_path = FBDPath("{gbd_round_id}/future/population/{version}".format(
        gbd_round_id=GBD_ROUND, version=future_pop_version))
    
    future_pop = open_xr(future_path / "population_combined.nc").data
    past_pop = open_xr(past_path / "population.nc").data.sel(
        location_id=future_pop.location_id)
    
    pop = past_pop.combine_first(future_pop).sel(
        age_group_id=age_groups, sex_id=BOTH_SEX_ID, scenario=scenarios)
    
    return pop


def super_region_pop(pop_da):
    locs = db.get_locations_by_level(1)[['location_id', 'location_name']]
    supreg_pop = pop_da.sel(location_id=locs.location_id.tolist())
    supreg_df = supreg_pop.sel(quantile="mean").drop(
        ["scenario", "sex_id", "age_group_id"]).to_dataframe(
    ).reset_index()
    
    peak_vals = supreg_df.groupby("location_id")["population"].max()
    peaks = supreg_df.merge(peak_vals).merge(locs)
    peak_years = peaks.year_id
    if (peak_years < 2100).all() != True:
        warnings.warn("Not all super regions peak!")
    
    supreg_17_100 = supreg_df[supreg_df.year_id.isin([2017, 2100])].drop(
        "quantile", axis=1).set_index(
        ["location_id", "year_id"]).unstack("year_id").droplevel(
        0, axis=1).reset_index()
    supreg_17_100["diff"] = supreg_17_100[2100] - supreg_17_100[2017]
    pos_diff_locs = supreg_17_100[supreg_17_100["diff"]>0].location_id.tolist()
    supreg_higher100 = supreg_pop.sel(year_id=2100, location_id=pos_diff_locs)
    supreg_higher100_mean_ui = supreg_higher100.drop(
        ["scenario", "sex_id", "age_group_id", "year_id"]).to_dataframe(
    ).unstack().droplevel(0, axis=1).reset_index().merge(locs)
    
    print("SUPER-REGIONS WITH HIGHER POP IN 2100 THAN 2017 (IN 2100)")
    print(supreg_higher100_mean_ui)
    print("\nREMINDER: check plots for most severe declines\n")

    
def pop_largest_countries(pop_da):
    locs_w_supreg = db.get_locations_by_level(3)[['location_id',
                                                  'location_name',
                                                  "super_region_name"]]
    locs = locs_w_supreg.drop("super_region_name", axis=1)
    
    largest_5_2100 = pop_da.sel(year_id=2100, location_id=COUNTRIES).drop(
        ["year_id", "age_group_id", "sex_id", "scenario"]).to_dataframe(
    ).unstack("quantile").droplevel(0, axis=1).reset_index().nlargest(
        5, "mean").merge(locs)
    
    print("LARGEST 5 COUNTRIES IN 2100")
    print(largest_5_2100)
    
    print("\nCHECK VIA PLOTS:\n"
          "'Nigeria is forecasted with continued growth through 2100 "
          "and is expected to be the 2nd most populous country by then. The "
          "reference forecasts for China and India peak well before 2050 and "
          "both countries thereafter follow steep declining trajectories...'")
    
    
    chn_peak_year = pop_da.sel(location_id=6, quantile="mean")
    chn_peak_year = chn_peak_year.where(
        chn_peak_year == chn_peak_year.max(), drop=True).year_id.values[0]
    chn_peak_yr_da = pop_da.sel(location_id=6, year_id=chn_peak_year).drop(
        "year_id")
    
    ind_peak_year = pop_da.sel(location_id=163, quantile="mean")
    ind_peak_year = ind_peak_year.where(
        ind_peak_year == ind_peak_year.max(), drop=True).year_id.values[0]
    ind_peak_yr_da = pop_da.sel(location_id=163, year_id=ind_peak_year).drop(
        "year_id")

    usa_peak_year = pop_da.sel(location_id=102, quantile="mean")
    usa_peak_year = usa_peak_year.where(
        usa_peak_year == usa_peak_year.max(), drop=True).year_id.values[0]
    usa_peak_yr_da = pop_da.sel(location_id=102, year_id=usa_peak_year).drop(
        "year_id")
    
    chn_ind_usa_peak = xr.concat(
        [chn_peak_yr_da, ind_peak_yr_da, usa_peak_yr_da], dim="location_id")
    chn_ind_usa_100 = pop_da.sel(year_id=2100, location_id=[6, 163, 102])
    
    pcnt_2100_to_peak_chn_ind_usa = (
        (chn_ind_usa_100 / chn_ind_usa_peak) * 100).round(
        N_DECIMALS).drop(
        ["year_id", "age_group_id","sex_id", "scenario"]).to_dataframe(
        ).unstack("quantile").droplevel(0, axis=1).reset_index().merge(locs)
    
    print("\nCHINA INDIA USA PCT OF PEAK IN 2100")
    print(pcnt_2100_to_peak_chn_ind_usa)
    
    print("\nCHECK WITH TABLES BELOW:\n"
          "'The USA is projected to experience population growth until "
          "mid-century, followed by only moderate decline of less than 10% by "
          "2100. We forecast that the number of countries in sub-Saharan Africa "
          "among the countries with the 10 largest populations will increase "
          "from only Nigeria in 2017 to also include DR Congo, Ethiopia, and "
          "Niger in 2100.'")
    
    largest_10_2017 = pop_da.sel(year_id=2017, location_id=COUNTRIES).drop(
        ["year_id", "age_group_id", "sex_id", "scenario"]).to_dataframe(
    ).unstack("quantile").droplevel(0, axis=1).reset_index().nlargest(
        10, "mean").merge(locs_w_supreg).drop(
        ["lower", "upper"], axis=1).query(
        "super_region_name == 'Sub-Saharan Africa'")
    
    print("\nSUB-SAHARAN COUNTRIES FROM 10 LARGEST COUNTRIES 2017")
    print(largest_10_2017)
    
    largest_10_2100 = pop_da.sel(year_id=2100, location_id=COUNTRIES).drop(
        ["year_id", "age_group_id", "sex_id", "scenario"]).to_dataframe(
    ).unstack("quantile").droplevel(0, axis=1).reset_index().nlargest(
        10, "mean").merge(locs_w_supreg).drop(
        ["lower", "upper"], axis=1).query(
        "super_region_name == 'Sub-Saharan Africa'")
    
    print("\nSUB-SAHARAN COUNTRIES FROM 10 LARGEST COUNTRIES 2100")
    print(largest_10_2100)
    
    

def super_region_age_structure():
    u15_subsah = pull_pop(PAST_POP_VERSION, FUTURE_POP_VERSION,
                          scenarios = [0,-1], age_groups=range(2,8)).sel(
        location_id=166, year_id=[2017, 2100], quantile="mean").sum(
        "age_group_id")
    
    u15_subsah_17 = u15_subsah.sel(year_id=2017).drop("year_id")
    u15_subsah_100 = u15_subsah.sel(year_id=2100).drop("year_id")
    u15_subsah_fold_chng = (u15_subsah_100 / u15_subsah_17).to_dataframe(
        ).reset_index()
    
    print("\nSUB-SAHARAN AFRICA FOLD CHANGE FROM 2017 TO 2100 UNDER 15")
    print(u15_subsah_fold_chng)
    
    print("\nCHECK VIA PLOTS: 'For all super-regions except sub-Saharan Africa "
        "and north Africa and Middle East, we forecast similar age structures "
        "under all scenarios.'")


if __name__ == '__main__':

    pop_da = pull_pop(PAST_POP_VERSION, FUTURE_POP_VERSION)
    
    super_region_pop(pop_da)
    pop_largest_countries(pop_da)
    super_region_age_structure()
"""
Three GBD super-regions had reached a below-replacement TFR (<2·1) by 2017:
high-income; central Europe, eastern Europe, and central Asia; and southeast
Asia, east Asia, and Oceania. By 2100, our reference TFR forecasts for the
super-regions were between 1·33 (95% UI 0·96–1·90) for south Asia and 1·80
(1·25–2·40) for central Europe, eastern Europe, and central Asia (table 1).
Sub-Saharan Africa is forecasted to decline from 4·62 in 2017 to 1·73
(1·42–2·06) in 2100 and reach below-replacement in 2063.

Figure 10 shows trajectories of the TFR for the ten countries with the largest
populations today or in 2100. Our reference forecast suggests that India will
reach below-replacement in 2018. Thereafter, India is forecasted to have a
continued steep decline to about 2040 and reach a TFR of 1·29 (95% UI 0·99–1·89)
in 2100 (table 1). The GBD estimate of China’s TFR was 1·53 in 2017. China’s TFR
is forecasted to decline moderately to 1·42 (1·04–2·04) around 2030, and
thereafter increase slowly to 1·47 (0·96–2·55) by 2100. The USA is forecasted
to decline from 1·81 in 2017 to 1·53 (1·10–2·02) in 2100, while Indonesia and
Pakistan will decline from 1·97 and 3·40 to 1·51 (1·01–2·21) and 1·31
(0·99–2·04) in 2100, respectively. Not shown in Figure 10, Japan and Russia are
forecasted to stay stable or moderately decline from 1·33 and 1·61 in 2017 to
1·32 (0·96–2·03) and 1·43 (0·96–2·22), respectively, in 2100, contributing to
population declines that led them to no longer be forecasted among the ten most
populous countries in 2100.

In 2017, estimates of TFR were 5·11, 7·08, and 2·29 in Nigeria, Niger, and
South Africa, respectively; by 2100, our reference forecasts were 1·69
(95% UI 1·25–2·16), 1·79 (1·34–2·25), and 1·70 (1·28–2·19) (table 1). Only four
countries in sub-Saharan Africa are forecasted to stay above replacement in
2100: Somalia with a TFR of 2·57 (2·24–2·88), South Sudan (2·46 [2·07–2·83]),
Zimbabwe (2·22 [1·46–3·12]), and Chad (2·19 [1·90–2·54]). Lowest TFR reference
forecasts for sub-Saharan Africa in 2100 were for São Tomé and Príncipe,
Mozambique, Eritrea, Togo, and Sierra Leone, all between 1·25 (1·00–1·63) and
1·28 (1·02–1·66).

There is large variation in forecasted country and territory TFRs across
scenarios (figure 10). For example, in China the forecasted TFR range for 2100
is narrow, between 1·41 (95% UI 0·95–2·49) for the SDG pace scenario (table 1)
and 1·47 (0·96–2·59) for the other scenarios. For Nigeria, the TFR scenario
range is much wider, from 1·57 (1·11–2·07) in the SDG pace scenario (table 1)
to 3·11 (2·61–3·61) in the slower contraceptive met need and education scenario.
The variation in fertility forecasts across scenarios is explained by the
currently attained levels of education and contraceptive met need and their
modelled future effects on fertility, which are much larger in high-fertility
than low-fertility settings. See appendix 2 (section 4) for additional results.
"""

import pandas as pd
import xarray as xr

from fbd_core import db
from fbd_core.etl.transformation import expand_dimensions
from fbd_core.file_interface import FBDPath, open_xr, save_xr
import settings as sett

ALL_AGE_ID = 22
BOTH_SEX_ID = 3
GBD_ROUND = 5

N_DECIMALS = 2

COUNTRIES = db.get_locations_by_level(3).location_id.values.tolist()

TFR_VERSION = sett.BASELINE_VERSIONS["tfr_mean_ui"].version

PAST_POP_VERSION = sett.PAST_VERSIONS["population"].version
FUTURE_POP_VERSION = sett.BASELINE_VERSIONS["population_mean_ui"].version


def pull_tfr(future_tfr_version, scenarios = [-1,0,1,2,3]):
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


def get_highest_pop_locs_past_fut(pops, past_year, fut_year, num_locs):
    # Get num_locs highest locs in past and future and combine

    highest_past = set(pops.sel(
        year_id=past_year,
        quantile="mean").to_dataframe().reset_index().nlargest(
        num_locs, "population").location_id)
    highest_fut = set(pops.sel(
        year_id=fut_year,
        quantile="mean").to_dataframe().reset_index().nlargest(
        num_locs, "population").location_id)
    highest = list(highest_past | highest_fut)
    return highest


def get_quantiles_and_round(da, round_to):
    da_mean = da.sel(quantile="mean").values.round(round_to)
    da_lower = da.sel(quantile="lower").values.round(round_to)
    da_upper = da.sel(quantile="upper").values.round(round_to)

    return da_mean, da_lower, da_upper


def super_region_replacement(tfr_da):
    locs = db.get_locations_by_level(1)[['location_id', 'location_name']]
    tfr_df = tfr_da.drop("scenario").to_dataframe().reset_index()
    tfr_wide = tfr_df.set_index(
        ["location_id", "year_id", "quantile"]).unstack().droplevel(
        level=0, axis=1).reset_index()

    supreg_df = tfr_wide.merge(locs, on="location_id")

    below_rep_17 = supreg_df.query("year_id==2017 & mean<2.1").round(N_DECIMALS)
    below_rep_17_names = "; ".join(below_rep_17.location_name.unique())
    num_below_17 = len(below_rep_17.location_id.unique())

    supreg_100 = supreg_df.query("year_id==2100").round(N_DECIMALS)
    supreg_100_min = supreg_100[supreg_100["mean"]==supreg_100["mean"].min()]
    supreg_100_max = supreg_100[supreg_100["mean"]==supreg_100["mean"].max()]

    subsah = supreg_df.query("location_name=='Sub-Saharan Africa'").round(
        N_DECIMALS)
    subsah_17 = subsah.query("year_id==2017")
    subsah_100 = subsah.query("year_id==2100")
    subsah_1st_yr_below_rep = subsah.query("mean<2.1").year_id.min()

    print(
        f"{num_below_17} GBD super-regions had reached a below-replacement TFR "
        f"(<2·1) by 2017: {below_rep_17_names}. By 2100, our reference TFR "
        f"forecasts for the super-regions were between "
        f"{supreg_100_min['mean'].values[0]} (95% UI "
        f"{supreg_100_min.lower.values[0]}–{supreg_100_min.upper.values[0]}) "
        f"for {supreg_100_min.location_name.values[0]} "
        f"and {supreg_100_max['mean'].values[0]} ({supreg_100_max.lower.values[0]}– "
        f"{supreg_100_max.upper.values[0]}) for "
        f"{supreg_100_max.location_name.values[0]} "
        "(table 1). Sub-Saharan Africa is forecasted to decline from "
        f"{subsah_17['mean'].values[0]} in 2017 to {subsah_100['mean'].values[0]} "
        f"({subsah_100.lower.values[0]}–{subsah_100.upper.values[0]}) "
        f"in 2100 and reach below-replacement in {subsah_1st_yr_below_rep}.\n"
    )


def tfr_largest_countries(tfr_da, pop_da):
    pop_countries = pop_da.sel(location_id=COUNTRIES)
    highpop_nats = get_highest_pop_locs_past_fut(
        pop_countries, 2017, 2100, 8) # this should give 10 countries

    assert (len(highpop_nats) == 10), "Not 10 high-pop countries."
    highpop_set = set(highpop_nats)
    select_nat_set = set([163, 6, 102, 11, 165])
    assert (select_nat_set.issubset(highpop_set)), ("Not all referenced "
        "countries in high-pop set")

    highpop_tfr = tfr_da.sel(location_id=highpop_nats)

    tfr_ind = highpop_tfr.sel(location_id=163)
    ind_mean = tfr_ind.sel(quantile="mean") # india
    ind_1st_yr_below_rep = ind_mean.where(
        ind_mean<2.1, drop=True).year_id.min().values.round(0)
    ind_100_mean = ind_mean.sel(year_id=2100).values.round(N_DECIMALS)
    ind_100_lower = tfr_ind.sel(
        quantile="lower", year_id=2100).values.round(N_DECIMALS)
    ind_100_upper = tfr_ind.sel(
        quantile="upper", year_id=2100).values.round(N_DECIMALS)

    chn_tfr = highpop_tfr.sel(location_id=6)
    chn_30 = chn_tfr.sel(year_id=2030)
    chn_100 = chn_tfr.sel(year_id=2100)
    chn_30_mean, chn_30_lower, chn_30_upper = get_quantiles_and_round(
        chn_30, N_DECIMALS)
    chn_100_mean, chn_100_lower, chn_100_upper = get_quantiles_and_round(
        chn_100, N_DECIMALS)

    usa_100 = highpop_tfr.sel(location_id=102, year_id=2100)
    usa_100_mean, usa_100_lower, usa_100_upper = get_quantiles_and_round(
        usa_100, N_DECIMALS)

    indo_100 = highpop_tfr.sel(location_id=11, year_id=2100) # indonesia
    indo_100_mean, indo_100_lower, indo_100_upper = get_quantiles_and_round(
        indo_100, N_DECIMALS)

    pak_100 = highpop_tfr.sel(location_id=165, year_id=2100)
    pak_100_mean, pak_100_lower, pak_100_upper = get_quantiles_and_round(
        pak_100, N_DECIMALS)

    # these countries are not in the high-pop set
    jap_100 = tfr_da.sel(location_id=67, year_id=2100)
    jap_100_mean, jap_100_lower, jap_100_upper = get_quantiles_and_round(
        jap_100, N_DECIMALS)

    rus_100 = tfr_da.sel(location_id=62, year_id=2100)
    rus_100_mean, rus_100_lower, rus_100_upper = get_quantiles_and_round(
        rus_100, N_DECIMALS)

    print(
        "Figure 10 shows trajectories of the TFR for the ten countries with "
        "the largest populations today or in 2100. Our reference forecast "
        f"suggests that India will reach below-replacement in "
        f"{ind_1st_yr_below_rep}. Thereafter, India is forecasted to have a "
        "continued steep decline to about 2040 and reach a TFR of "
        f"{ind_100_mean} (95% UI {ind_100_lower}–{ind_100_upper}) in 2100 "
        "(table 1). The GBD estimate of China’s TFR was 1·53 in 2017. China’s "
        f"TFR is forecasted to decline moderately to {chn_30_mean} "
        f"({chn_30_lower}–{chn_30_upper}) around 2030, and thereafter increase "
        f"slowly to {chn_100_mean} ({chn_100_lower}–{chn_100_upper}) by 2100. "
        "The USA is forecasted to decline from 1·81 in 2017 to "
        f"{usa_100_mean} ({usa_100_lower}–{usa_100_upper}) in 2100, while "
        "Indonesia and Pakistan will decline from 1·97 and 3·40 to"
        f"{indo_100_mean} ({indo_100_lower}–{indo_100_upper}) and "
        f"{pak_100_mean} ({pak_100_lower}–{pak_100_upper}) in 2100, "
        "respectively. Not shown in Figure 10, Japan and Russia are forecasted "
        "to stay stable or moderately decline from 1·33 and 1·61 in 2017 to "
        f"{jap_100_mean} ({jap_100_lower}–{jap_100_upper}) and "
        f"{rus_100_mean} ({rus_100_lower}–{rus_100_upper}), respectively, in "
        "2100, contributing to population declines that led them to no longer "
        "be forecasted among the ten most populous countries in 2100.\n"
    )


def tfr_subsaharan_africa(tfr_da):
    locs = db.get_locations_by_level(3)[['location_id', 'location_name',
                                         'super_region_name']]
    subsah_countries = locs.query(
        "super_region_name=='Sub-Saharan Africa'").location_id.tolist()
    subsah_100 = tfr_da.sel(location_id=subsah_countries, year_id=2100).drop(
        ["scenario", "year_id"])

    # specific countries
    nga_100 = subsah_100.sel(location_id=214) # nigeria
    nga_100_mean, nga_100_lower, nga_100_upper = get_quantiles_and_round(
        nga_100, N_DECIMALS)

    ner_100 = subsah_100.sel(location_id=213) # niger
    ner_100_mean, ner_100_lower, ner_100_upper = get_quantiles_and_round(
        ner_100, N_DECIMALS)

    sa_100 = subsah_100.sel(location_id=196) # south africa
    sa_100_mean, sa_100_lower, sa_100_upper = get_quantiles_and_round(
        sa_100, N_DECIMALS)

    # countries above replacement in 2100
    above_rep_100 = subsah_100.round(N_DECIMALS).to_dataframe().unstack().\
        transpose().reset_index().query("mean>2.1").merge(locs).\
        sort_values(by="mean", axis=0, ascending=False).reset_index(drop=True)

    abvrep_rows = [] # subset each rows
    for row in range(0, len(above_rep_100)):
        abvrep_rows.append(above_rep_100.iloc[row])

    abvrep_strings = [] # turn each row into a string
    for row in range(0, len(abvrep_rows)):
        if row == 0:
            string = (f"{abvrep_rows[0].location_name} with a TFR of "
                      f"{abvrep_rows[0]['mean']} ({abvrep_rows[0].lower}–"
                      f"{abvrep_rows[0].upper})")
        else:
            string = (f"{abvrep_rows[row].location_name} "
                      f"({abvrep_rows[row]['mean']} [{abvrep_rows[row].lower}–"
                      f"{abvrep_rows[row].upper}])")
        abvrep_strings.append(string) # make list of string

    abvrep_string = ", ".join(abvrep_strings) # join list to make final string

    # lowest 5 tfr countries
    lowest_5_100 = subsah_100.to_dataframe().unstack().transpose().reset_index().\
        nsmallest(5, "mean").reset_index().merge(locs).round(N_DECIMALS)
    # names ascending order
    lowest_ascending = ", ".join(lowest_5_100.location_name)
    # get range
    lowest_low = (f"{lowest_5_100.iloc[0]['mean']} ({lowest_5_100.iloc[0].lower}–"
                  f"{lowest_5_100.iloc[0].upper})")
    highest_low = (f"{lowest_5_100.iloc[4]['mean']} ({lowest_5_100.iloc[4].lower}–"
                   f"{lowest_5_100.iloc[4].upper})")

    print(
        "In 2017, estimates of TFR were 5·11, 7·08, and 2·29 in Nigeria, Niger, "
        "and South Africa, respectively; by 2100, our reference forecasts were "
        f"{nga_100_mean} (95% UI {nga_100_lower}–{nga_100_upper}), {ner_100_mean} "
        f"({ner_100_lower}–{ner_100_upper}), and {sa_100_mean} ({sa_100_lower}–"
        f"{sa_100_upper}) (table 1). Only four countries in sub-Saharan Africa "
        f"are forecasted to stay above replacement in 2100: {abvrep_string}. "
        f"Lowest TFR reference forecasts for sub-Saharan Africa in 2100 were for "
        f"{lowest_ascending}, all between {lowest_low} and {highest_low}.\n"
    )


def tfr_scenarios(da):

    da_scens = da.assign_coords(
        scenario=["slower contraceptive met need and education", "reference",
                  "faster contraceptive met need and education",
                  "fastest contraceptive met need and education",
                  "SDG contraceptive met need and education"])

    # china
    chn_100 = da_scens.sel(
        location_id=6, year_id=2100).drop(
        ["year_id", "location_id"]).to_dataframe().reset_index()
    chn_wide = chn_100.round(N_DECIMALS).set_index(
        ["scenario", "quantile"]).unstack().droplevel(0, axis=1).reset_index()
    chn_min = chn_wide[chn_wide["mean"] == chn_wide["mean"].min()]
    chn_max = chn_wide[chn_wide["mean"] == chn_wide["mean"].max()]

    chn_min_str = (f"{chn_min['mean'].values[0]} (95% UI {chn_min.lower.values[0]}"
                   f"–{chn_min.upper.values[0]})")
    chn_min_scen = chn_min.scenario.values[0]
    # assuming same mean for rest of scenarios
    chn_max_str = (f"{chn_max['mean'].unique()[0]} ({chn_max.lower.min()}"
                   f"–{chn_max.upper.max()})")
    if len(chn_max) == 4:
        chn_max_scen = "other"
    else:
        chn_max_scen = chn_max.scenario.values[0]

    # nigeria
    nga_100 = da_scens.sel(
        location_id=214, year_id=2100).drop(
        ["year_id", "location_id"]).to_dataframe().reset_index()
    nga_wide = nga_100.round(N_DECIMALS).set_index(
        ["scenario", "quantile"]).unstack().droplevel(0, axis=1).reset_index()
    nga_min = nga_wide[nga_wide["mean"] == nga_wide["mean"].min()]
    nga_max = nga_wide[nga_wide["mean"] == nga_wide["mean"].max()]

    nga_min_str = (f"{nga_min['mean'].values[0]} ({nga_min.lower.values[0]}"
                   f"–{nga_min.upper.values[0]})")
    nga_min_scen = nga_min.scenario.values[0]

    nga_max_str = (f"{nga_max['mean'].values[0]} ({nga_max.lower.values[0]}"
                   f"–{nga_max.upper.values[0]})")
    nga_max_scen = nga_max.scenario.values[0]

    print(
        "There is large variation in forecasted country and territory TFRs across "
        "scenarios (figure 10). For example, in China the forecasted TFR range for "
        f"2100 is narrow, between {chn_min_str} for the {chn_min_scen} scenario "
        f"(table 1) and {chn_max_str} for the {chn_max_scen} scenarios. For "
        f"Nigeria, the TFR scenario range is much wider, from {nga_min_str} in the "
        f"{nga_min_scen} scenario (table 1) to {nga_max_str} in the {nga_max_scen} "
        "scenario. The variation in fertility forecasts across scenarios is "
        "explained by the currently attained levels of education and contraceptive "
        "met need and their modelled future effects on fertility, which are much "
        "larger in high-fertility than low-fertility settings. See appendix 2 "
        "(section 4) for additional results."
    )


if __name__ == '__main__':

    pop_da = pull_pop(PAST_POP_VERSION, FUTURE_POP_VERSION)
    tfr_da = pull_tfr(TFR_VERSION)
    tfr_da_ref = tfr_da.sel(scenario=0)

    super_region_replacement(tfr_da_ref)
    tfr_largest_countries(tfr_da_ref, pop_da)
    tfr_subsaharan_africa(tfr_da_ref)
    tfr_scenarios(tfr_da)

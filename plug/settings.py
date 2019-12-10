"""
Collection of versions used for 2019 populations paper.

The 99 and SDG scenario version are the same as baseline except met need, edu,
asfr, sdi, death, pop, lex, and tfr.

SDI and death only have one alt version where -1 is SDG and 1 is 99.


Example usage:

import settings
settings.BASELINE_VERSIONS["met_need"].gbd_round_id
settings.BASELINE_VERSIONS["met_neec"].version
"""
from collections import namedtuple


VersionGBDRoundID = namedtuple("VersionGBDRoundID", "version gbd_round_id")


# first entry is version, second entry is gbd_round_id
BASELINE_VERSIONS = dict(
  met_need=VersionGBDRoundID("20190228_1980_trunc_2140_just_nationals", 5),
  education=VersionGBDRoundID("20190228_from1950_till2140_cohort_corr", 5),
  ldi=VersionGBDRoundID("20181205_VL_103_fixed_ref_only", 5),
  asfr=VersionGBDRoundID("20190806_141418_fix_draw_bound_ccfx_to2110", 5),
  sdi=VersionGBDRoundID("20190606_venezuela_ldi_fix", 5),
  sevs=VersionGBDRoundID("20191011_ref_ref_ref_test_omega_max3", 5),
  scalars=VersionGBDRoundID("20191015_ref_ref_ref_test_omega_max3", 5),
  vehicles_2_plus_4wheels_pc=VersionGBDRoundID(
    "20190104_trunc_20ci_ref_only", 5),
  hiv=VersionGBDRoundID("20190720_hiv_20190104_ref_only", 5),
  shocks=VersionGBDRoundID("20191108_shocks_only_decay_weight_15", 5),
  death=VersionGBDRoundID("20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all", 5),
  population=VersionGBDRoundID("20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all", 5),
  population_mean_ui=VersionGBDRoundID("20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all_combined", 5),
  lex=VersionGBDRoundID("20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all_gbd4_extrap", 5),
  migration=VersionGBDRoundID("20190801_20190607_arima_cap", 5),
  tfr=VersionGBDRoundID("20190806_141418_fix_draw_bound_ccfx_to2110_5_scen", 5),
  tfr_mean_ui=VersionGBDRoundID(
    "20190806_141418_fix_draw_bound_ccfx_to2110_5_scen_combined", 5),
  gdp=VersionGBDRoundID("20191030_15_ref_85_resub_agg_arima_squeeze_shocks_only_decay_wt_15_hiv_all_ordered_gdp", 5)
)

FASTEST_SCENARIO_VERSIONS = dict(
  met_need=VersionGBDRoundID("20190509_non_trunc_better_99th", 5),
  education=VersionGBDRoundID(
    "20190611_alt_scenario_better_99th_cohort_corrected", 5),
  ldi=VersionGBDRoundID("20181205_VL_103_fixed_ref_only", 5),
  asfr=VersionGBDRoundID("20190807_163915_fix_draw_bound_ccfx_99th_to2110", 5),
  sdi=VersionGBDRoundID("20191106_venezuela_ldi_reordered", 5),
  sevs=VersionGBDRoundID("20191011_ref_ref_ref_test_omega_max3", 5),
  scalars=VersionGBDRoundID("20191015_ref_ref_ref_test_omega_max3", 5),
  vehicles_2_plus_4wheels_pc=VersionGBDRoundID(
    "20190104_trunc_20ci_ref_only", 5),
  hiv=VersionGBDRoundID("20190720_hiv_20190104_ref_only", 5),
  shocks=VersionGBDRoundID("20191108_shocks_only_decay_weight_15", 5),
  death=VersionGBDRoundID(
    "20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all", 5),
  population=VersionGBDRoundID("20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all", 5),
  population_mean_ui=VersionGBDRoundID("20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all_combined", 5),
  lex=VersionGBDRoundID("20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all_gbd4_extrap", 5),
  migration=VersionGBDRoundID("20190801_20190607_arima_cap", 5),
  tfr=VersionGBDRoundID("20190806_141418_fix_draw_bound_ccfx_to2110_5_scen", 5),
  tfr_mean_ui=VersionGBDRoundID(
    "20190806_141418_fix_draw_bound_ccfx_to2110_5_scen_combined", 5),
  gdp=VersionGBDRoundID("20191030_15_ref_85_resub_agg_arima_squeeze_shocks_only_decay_wt_15_hiv_all_ordered_gdp", 5)
)

SDG_SCENARIO_VERSIONS = dict(
  met_need=VersionGBDRoundID("20190705_better_sdg", 5),
  education=VersionGBDRoundID("20190705_edu_better_sdg", 5),
  ldi=VersionGBDRoundID("20181205_VL_103_fixed_ref_only", 5),
  asfr=VersionGBDRoundID(
    "20190807_164000_fix_draw_bound_ccfx_sdg_to2110_scen_swapped", 5),
  sdi=VersionGBDRoundID("20190614_alternate_scenarios", 5),
  sevs=VersionGBDRoundID("20181204_post_10pct_caps_ref_only", 5),
  scalars=VersionGBDRoundID("20190608_2100_1kdraw_ref_only", 5),
  vehicles_2_plus_4wheels_pc=VersionGBDRoundID(
    "20190104_trunc_20ci_ref_only", 5),
  hiv=VersionGBDRoundID("20190720_hiv_20190104_ref_only", 5),
  shocks=VersionGBDRoundID("20190606_ldi_fix", 5),
  death=VersionGBDRoundID(
    "20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all", 5),
  population=VersionGBDRoundID("20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all", 5),
  population_mean_ui=VersionGBDRoundID("20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all_combined", 5),
  lex=VersionGBDRoundID("20191126_15_ref_85_99_RELU_1000draw_drawfix_agg_arima_squeeze_shocks_hiv_all_gbd4_extrap", 5),
  migration=VersionGBDRoundID("20190801_20190607_arima_cap", 5),
  tfr=VersionGBDRoundID(
    "20190806_141418_fix_draw_bound_ccfx_to2110_5_scen", 5),
  tfr_mean_ui=VersionGBDRoundID(
    "20190806_141418_fix_draw_bound_ccfx_to2110_5_scen_combined", 5),
  gdp=VersionGBDRoundID("20190808_15_ref_85_agg_gdp", 5)
)

PAST_VERSIONS = dict(
  population=VersionGBDRoundID("20181206_pop_1950_2017", 5),
  lex=VersionGBDRoundID("20191029_fhs_computed_from_20190109_version90_etl_gbd4_all_youth", 5),
  tfr=VersionGBDRoundID("20191114_va84", 5)
)

WPP_VERSIONS = dict(
  population=VersionGBDRoundID("2019", "wpp"),
  population_aggs=VersionGBDRoundID("2019_fhs_agg", "wpp"),
)

WITT_VERSIONS = dict(
  population=VersionGBDRoundID("2018", "wittgenstein"),
  population_aggs=VersionGBDRoundID("2018_fhs_agg", "wittgenstein")
)
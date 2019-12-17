'''
This makes the population pyramids used for the population paper 2019
'''
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from fbd_core import db, YearRange, argparse
from fbd_core.etl import expand_dimensions, resample, Aggregator
from fbd_core.file_interface import FBDPath, open_xr
from matplotlib.backends.backend_pdf import PdfPages

import fhs_2019_population_paper.plug.settings as settings

ALL_AGE_GROUP_IDS = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
SEX_IDS = [1, 2]

PAST = settings.PAST_VERSIONS["population"].version
REF = settings.BASELINE_VERSIONS["population_mean_ui"].version
REF_DRAWS = settings.BASELINE_VERSIONS["population"].version
GBD_ROUND_ID = settings.BASELINE_VERSIONS["population"].gbd_round_id
YEARS = YearRange(1990, 2018, 2100)


def prep_pop_da(past_version, forecast_version, gbd_round_id, years):
    forecast_pop_file = FBDPath(
        f"/{gbd_round_id}/future/population/{forecast_version}/"
        f"population_combined.nc")
    forecast_fhs = open_xr(forecast_pop_file).data.sel(quantile='mean', drop=True)

    past_fhs_file = FBDPath(
        f"/{gbd_round_id}/past/population/{past_version}/population.nc")
    past_fhs = expand_dimensions(
        open_xr(past_fhs_file).data.sel(
            year_id=years.past_years,
            sex_id=forecast_fhs["sex_id"],
            age_group_id=forecast_fhs["age_group_id"],
            location_id=forecast_fhs["location_id"]),
        scenario=forecast_fhs.scenario.values)

    fhs_all_scenarios = xr.concat([past_fhs, forecast_fhs], dim="year_id")

    fhs = fhs_all_scenarios.sel(scenario=[-1, 0, 1])
    alt_sdg = fhs_all_scenarios.sel(scenario=[3])
    alt_99 = fhs_all_scenarios.sel(scenario=[2])

    ages = db.get_ages().query("age_group_id in @ALL_AGE_GROUP_IDS")
    days = ages[["age_group_id", "age_group_days_start", "age_group_days_end"]]
    days["mean_age"] = (days["age_group_days_end"] - (
                days["age_group_days_end"] - days[
            "age_group_days_start"]) / 2) / 365.25
    mean_age = days.set_index("age_group_id")["mean_age"].to_xarray()

    data_fhs = fhs.sel(age_group_id=mean_age["age_group_id"], sex_id=SEX_IDS)
    data_sdg = alt_sdg.sel(age_group_id=mean_age["age_group_id"], sex_id=SEX_IDS)
    data_99 = alt_99.sel(age_group_id=mean_age["age_group_id"], sex_id=SEX_IDS)

    avg_age_fhs = (data_fhs * mean_age).sum("age_group_id") / data_fhs.sum(
        "age_group_id")
    avg_age_sdg = (data_sdg * mean_age).sum("age_group_id") / data_sdg.sum(
        "age_group_id")
    avg_age_99 = (data_99 * mean_age).sum("age_group_id") / data_99.sum(
        "age_group_id")

    ds = data_fhs.rename("population").to_dataset()
    ds_sdg = data_sdg.rename("population").to_dataset()
    ds_99 = data_99.rename("population").to_dataset()

    return avg_age_fhs, avg_age_sdg, avg_age_99, ds, ds_sdg, ds_99


def pop_plot(avg_age_fhs, avg_age_sdg, avg_age_99, ds, ds_sdg, ds_99,
             years, location_id=1):
    location_metadata = db.get_locations_by_max_level(3)

    ages = db.get_ages().query("age_group_id in @ALL_AGE_GROUP_IDS")
    scenarios = [
        {"year": years.past_end, "scenario": 0, "name": years.past_end,
         "color": "black"},
        {"year": years.forecast_end, "scenario": 0, "name": "Reference",
         "color": "steelblue"},
        {"year": years.forecast_end, "scenario": -1,
         "name": "Slower Met Need and Education Pace", "color": "firebrick"},
        {"year": years.forecast_end, "scenario": 1,
         "name": "Faster Met Need and Education Pace", "color": "forestgreen"},
    ]

    alt_scenario_sdg = [{"year": years.forecast_end, "scenario": 3,
                         "name": "SDG Met Need and Education Pace",
                         "color": "#ff7f00"}]

    alt_scenario_99 = [{"year": years.forecast_end, "scenario": 2,
                        "name": "Fastest Met Need and Education Pace",
                        "color": "#984ea3"}]

    gs = plt.GridSpec(1, 2)
    ax_male = plt.subplot(gs[:, 0])
    ax_female = plt.subplot(gs[:, 1])

    fig = ax_female.figure

    # pop plots
    pop_df = ds.sel(
        location_id=location_id, age_group_id=ages["age_group_id"].values
    )["population"].to_dataframe().reset_index().sort_values("age_group_id")

    alt_df_sdg = ds_sdg.sel(
        location_id=location_id, age_group_id=ages["age_group_id"].values
    )["population"].to_dataframe().reset_index().sort_values("age_group_id")

    alt_df_99 = ds_99.sel(
        location_id=location_id, age_group_id=ages["age_group_id"].values
    )["population"].to_dataframe().reset_index().sort_values("age_group_id")

    pop_df["age"] = pop_df["age_group_id"].factorize()[0]
    male_pop = pop_df[pop_df.sex_id == 1]
    male_max_pop = male_pop["population"].max() * 1.01

    female_pop = pop_df[pop_df.sex_id == 1]
    female_max_pop = female_pop["population"].max() * 1.01

    # sdg
    alt_df_sdg["age"] = alt_df_sdg["age_group_id"].factorize()[0]

    # 99
    alt_df_99["age"] = alt_df_99["age_group_id"].factorize()[0]

    # comment or uncomment for fixed/non-fixed axis
    max_pop = max([male_max_pop, female_max_pop])
    # max_pop = pop_df["population"].max() * 1.01

    if max_pop > 1e9:
        max_pop = max_pop / 1e9
        pop_df["population"] = pop_df["population"] / 1e9
        alt_df_sdg["population"] = alt_df_sdg["population"] / 1e9
        alt_df_99["population"] = alt_df_99["population"] / 1e9
        label = "Population (Billions)"
    elif max_pop > 1e6:
        max_pop = max_pop / 1e6
        pop_df["population"] = pop_df["population"] / 1e6
        alt_df_sdg["population"] = alt_df_sdg["population"] / 1e6
        alt_df_99["population"] = alt_df_99["population"] / 1e6
        label = "Population (Millions)"
    elif max_pop > 1e3:
        max_pop = max_pop / 1e3
        pop_df["population"] = pop_df["population"] / 1e3
        alt_df_sdg["population"] = alt_df_sdg["population"] / 1e3
        alt_df_99["population"] = alt_df_99["population"] / 1e3
        label = "Population (Thousands)"
    else:
        label = "Population"

    # male plot
    for c in scenarios:
        df = pop_df.query(
            "sex_id == 1 & scenario == {} & year_id == {}".format(c["scenario"],
                                                                  c["year"]))
        ax_male.step(
            x=df["population"].values.tolist() + [0],
            y=df["age"].values.tolist() + [ages.shape[0]],
            color=c["color"],
            linewidth=2,
            alpha=0.8,
            label=c["name"])
        a = avg_age_fhs.sel(location_id=location_id, sex_id=1,
                            scenario=c["scenario"], year_id=c["year"])
        ax_male.plot(0, a / 5 + 0.5, marker="<", color=c["color"],
                     markersize=20, alpha=0.8)

    # sdg
    for c in alt_scenario_sdg:
        df = alt_df_sdg.query(
            "sex_id == 1 & scenario == {} & year_id == {}".format(c["scenario"],
                                                                  c["year"]))
        ax_male.step(
            x=df["population"].values.tolist() + [0],
            y=df["age"].values.tolist() + [ages.shape[0]],
            color=c["color"],
            linewidth=2,
            alpha=0.8,
            label=c["name"])
        a = avg_age_sdg.sel(location_id=location_id, sex_id=1,
                            scenario=c['scenario'], year_id=c["year"])
        ax_male.plot(0, a / 5 + 0.5, marker="<", color=c["color"],
                     markersize=20, alpha=0.8)

    # 99
    for c in alt_scenario_99:
        df = alt_df_99.query(
            "sex_id == 1 & scenario == {} & year_id == {}".format(c["scenario"],
                                                                  c["year"]))
        ax_male.step(
            x=df["population"].values.tolist() + [0],
            y=df["age"].values.tolist() + [ages.shape[0]],
            color=c["color"],
            linewidth=2,
            alpha=0.8,
            label=c["name"])
        a = avg_age_99.sel(location_id=location_id, sex_id=1,
                           scenario=c['scenario'], year_id=c["year"])
        ax_male.plot(0, a / 5 + 0.5, marker="<", color=c["color"],
                     markersize=20, alpha=0.8)

    ax_male.set_xlim(max_pop + .1 * max_pop, 0)
    ax_male.set_ylim(0, ages.shape[0])
    ax_male.set_yticks(np.arange(ages.shape[0]) + 0.5)
    ax_male.set_yticklabels(ages["age_group_name_short"], fontsize=14)
    ax_male.set_title("Male", fontsize=18)
    ax_male.set_xlabel(label)
    ax_male.legend(frameon=False, loc="upper left")
    sns.despine(ax=ax_male, left=True, right=False)

    # female plot
    for c in scenarios:
        df = pop_df.query(
            "sex_id == 2 & scenario == {} & year_id == {}".format(c["scenario"],
                                                                  c["year"]))
        ax_female.step(x=df["population"].values.tolist() + [0],
                       y=df["age"].values.tolist() + [ages.shape[0]],
                       color=c["color"],
                       linewidth=2,
                       alpha=0.8)
        a = avg_age_fhs.sel(location_id=location_id, sex_id=2,
                            scenario=c["scenario"], year_id=c["year"])
        ax_female.plot(0, a / 5 + 0.5, marker=">", color=c["color"],
                       markersize=20, alpha=0.8)

    # sdg
    for c in alt_scenario_sdg:
        df = alt_df_sdg.query(
            "sex_id == 2 & scenario == {} & year_id == {}".format(c["scenario"],
                                                                  c["year"]))
        ax_female.step(x=df["population"].values.tolist() + [0],
                       y=df["age"].values.tolist() + [ages.shape[0]],
                       color=c["color"],
                       linewidth=2,
                       alpha=0.8)
        a = avg_age_sdg.sel(location_id=location_id, sex_id=2,
                            scenario=c['scenario'], year_id=c["year"])
        ax_female.plot(0, a / 5 + 0.5, marker=">", color=c["color"],
                       markersize=20, alpha=0.8)

    # 99
    for c in alt_scenario_99:
        df = alt_df_99.query(
            "sex_id == 2 & scenario == {} & year_id == {}".format(c["scenario"],
                                                                  c["year"]))
        ax_female.step(x=df["population"].values.tolist() + [0],
                       y=df["age"].values.tolist() + [ages.shape[0]],
                       color=c["color"],
                       linewidth=2,
                       alpha=0.8)
        a = avg_age_99.sel(location_id=location_id, sex_id=2,
                           scenario=c['scenario'], year_id=c["year"])
        ax_female.plot(0, a / 5 + 0.5, marker=">", color=c["color"],
                       markersize=20, alpha=0.8)

    ax_female.set_xlim(0, max_pop + .1 * max_pop)
    ax_female.set_ylim(0, ages.shape[0])
    ax_female.set_yticks(np.arange(ages.shape[0]) + 0.5)
    ax_female.set_yticklabels([])
    ax_female.set_title("Female", fontsize=18)
    ax_female.set_xlabel(label)
    sns.despine(ax=ax_female)

    fig.suptitle(
        location_metadata.query("location_id == @location_id")["location_name"].values[0],
        fontsize=28)

    fig.set_size_inches(14, 8)

    plt.tight_layout()
    plt.subplots_adjust(top=.85)

    return fig


def main(past_version, forecast_version, gbd_round_id, years):
    avg_age_fhs, avg_age_sdg, avg_age_99, ds, ds_sdg, ds_99 = prep_pop_da(
        past_version, forecast_version, gbd_round_id, years)
    plot_file = FBDPath(
        f"/{gbd_round_id}/future/population/{forecast_version}",
        root_dir="plot")
    plot_file.mkdir(exist_ok=True)
    pdf_file = plot_file / "20191217test_figure_7_population_pyramids.pdf"

    location_metadata = db.get_locations_by_max_level(3)

    location_hierarchy = location_metadata.set_index(
        "location_id").to_xarray()["parent_id"]

    with PdfPages(pdf_file) as pdf:
        for l in location_hierarchy["location_id"]:
            fig=pop_plot(
                avg_age_fhs, avg_age_sdg, avg_age_99, ds, ds_sdg, ds_99,
                years, location_id=l)
            pdf.savefig(fig)
            # plt.show()


if __name__ == "__main__":

    main(PAST, REF, GBD_ROUND_ID, YEARS)

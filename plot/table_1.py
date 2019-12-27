from datetime import date
import numpy as np
from os import mkdir
import pandas as pd
import xarray as xr
import xlsxwriter

from db_queries import (get_covariate_estimates, get_location_metadata,
                        get_population)
from fbd_core import argparse, YearRange
from fbd_core.etl import expand_dimensions
from fbd_core.file_interface import FBDPath, open_xr


ALL_AGE_ID = 22
BOTH_SEX_ID = 3
SCENARIOS = [-1, 0, 1, 2, 3]
YEARS = YearRange(1990, 2018, 2100)

CELL_HT = {
    "title": 1,
    "location": 1,
    "stage": 0,
    "data_cols": 2
}

COL_RANGE = {
    "pop":2,
    "tfr":1
}

INDENT_MAP = {
    0: "",
    1: "  ",
    2: "    ",
    3: "      "
}

SCENARIO_MAP = {
    0:"ref", 
    -1:"worse",
    1:"better",
    2:"fastest",
    3:"sdg"
}

COL_NAME_MAP = {
    "lancet_label":"Location",
    "peak_pop_value": "Peak Population (year)",
    "value_2017_pop_ref": "2017",
    "value_2100_pop_ref": "2100 Reference Scenario",
    "value_2100_pop_sdg": "2100 SDG Scenario",
    "value_2017_tfr_ref": "2017",
    "value_2100_tfr_ref": "2100 Reference Scenario",
    "value_2100_tfr_sdg": "2100 SDG Scenario",
}

REVIEW_COL_NAME_MAP = {
    "lancet_label": "Location name",
    "mean_2017_pop_ref": "Population mean 2017",
    "lower_2017_pop_ref": "Population lower 2017",
    "upper_2017_pop_ref": "Population upper 2017",
    "mean_2100_pop_ref": "Reference population mean 2100",
    "lower_2100_pop_ref": "Reference population lower 2100",
    "upper_2100_pop_ref": "Reference population upper 2100",
    "mean_2100_pop_sdg": "SDG population mean 2100",
    "lower_2100_pop_sdg": "SDG population lower 2100",
    "upper_2100_pop_sdg": "SDG population upper 2100",
    "peak_pop": "Peak population",
    "peak_year": "Peak population year",
    "mean_2017_tfr_ref": "TFR mean 2017",
    "lower_2017_tfr_ref": "TFR lower 2017",
    "upper_2017_tfr_ref": "TFR upper 2017",
    "mean_2100_tfr_ref": "Reference TFR mean 2100",
    "lower_2100_tfr_ref": "Reference TFR lower 2100",
    "upper_2100_tfr_ref": "Reference TFR upper 2100",
    "mean_2100_tfr_sdg": "SDG TFR mean 2100",
    "lower_2100_tfr_sdg": "SDG TFR lower 2100",
    "upper_2100_tfr_sdg": "SDG TFR upper 2100"
}


def melt_to_xarray(df):
    """Melts GBD data with 'mean', 'lower', and 'upper' columns to a single
    'quantile' column; converts to xarray dataarray; and adds a scenario
    dimension. 

    Args:
        df (pandas dataframe):
            Dataframe with 'year_id', 'location_id', 'mean', 'lower', and
            'upper' columns.
    Returns:
        da_with_scenario (xarray dataarray):
            Dataarray with 'year_id', 'quantile', 'location_id', and 'scenario'
            dimensions.
    """
    df_long = pd.melt(df,
                      id_vars=["year_id", "location_id"],
                      value_vars=["mean", "lower", "upper"],
                      var_name="quantile")
    
    da = df_long.set_index(
         ["year_id", "quantile", "location_id"]).to_xarray()["value"]
    
    da_with_scenario = expand_dimensions(da, scenario=[0])
    
    return da_with_scenario


def combine_mean_ui(df, df_type="pop"):
    """Takes a dataframe with 'mean', 'lower', and 'upper' columns,
    and returns a dataframe with a 'value' column that has the mean, lower,
    and upper all together. If df_type == "pop", values are converted to
    millions.

    Args:
        df (pandas dataframe):
            Dataframe with 'mean', 'lower', and 'upper' columns.
    Returns:
        df (pandas dataframe):
            Dataframe with 'mean', 'lower', and 'upper' columns and added
            'value' column.
    """
    for col in ["mean", "lower", "upper"]:
        if df_type == "pop":
            df[col] = df[col] / 1000000
        df[col] = df[col].apply(lambda x: round(x, 2))
            
    df["value"] = (df["mean"].astype(str) + "  (" + df["lower"].astype(str) +
        " - " + df["upper"].astype(str) + ")")
    
    return df


def pivot_scenarios(df, prefix, scen_map, df_type="pop"):
    """Takes a dataframe with 'mean', 'lower', 'upper', 'value', and 'scenario'
    columns, and returns a dataframe with wide scenarios. Scenario column names
    are given by:
        prefix + "_" + df_type + "_" + df["scenario"].map(scen_map)

    Args:
        df (pandas dataframe):
            Dataframe with 'mean', 'lower', 'upper', 'value', and 'scenario'
            columns.
    Returns:
        df (pandas dataframe):
            Dataframe with 'mean', 'lower', 'upper', 'value' columns wide by
            scenario.
    """
    df["scenario"] = prefix + "_" + df_type + "_" +\
    df["scenario"].map(scen_map)
    df = df.pivot_table(values=["lower", "mean", "upper" ,"value"],
                        index="location_id",
                        columns="scenario",
                        aggfunc="first").reset_index()
    # This flattens the column levels
    df.columns = ['_'.join(col) for col in df.columns.values if col]
    df.rename(columns={"location_id_":"location_id"}, inplace=True)

    return df


def get_max_pop_year(group):
    """Takes a dataframe (or GroupBy object) with 'mean' and 'year_id' columns,
    and returns a dataframe with the max value in 'mean' and the 'year_id'
    of the max value.

    Args:
        df (pandas dataframe):
            Dataframe with 'year_id' and 'mean' columns.
    Returns:
        df (pandas dataframe):
            Dataframe with 'year_id' and 'mean' columns.
    """
    max_year_val = group.loc[group["mean"].idxmax()][["year_id", "mean"]]

    return max_year_val


def pull_reshape_pop(gbd_round_id, pop_version, location_ids):
    """Pulls year 2017 GBD round 5 populations, converts it an xarray dataarray,
    pulls forecast population, and concatenates the dataarrays. The new array is
    then converted to a pandas dataframe. Peak population and peak population
    year are pulled for each location in the dataframe. All required data are
    then reshaped and merged for downstream table production.

    Args:
        gbd_round_id (int):
            GBD round.
        pop_version (str):
            Forecast populations version.
        location_ids (list):
            List of location IDs to pull from both past and future data.
    Returns:
        pop_final_df (pandas dataframe):
            Dataframe with all required population data, reshaped for downstream
            table production.
    """
    p_end = YEARS.past_end
    f_end = YEARS.forecast_end

    # Get 2017 GBD pops
    pop_2017 = get_population(gbd_round_id=gbd_round_id, age_group_id=22,
                              sex_id=3, location_id=location_ids,
                              status="best", year_id=p_end, with_ui=True)[[
    "year_id", "location_id", "population", "lower", "upper"
    ]].rename(columns={"population": "mean"})
    
    pop_2017_da = melt_to_xarray(pop_2017)
    
    # Get future pops
    pop_fut = open_xr("5/future/population/"
                      f"{pop_version}/population_combined.nc").data
    
    pop_fut_sel = pop_fut.sel(location_id=location_ids, scenario=SCENARIOS,
                              age_group_id=ALL_AGE_ID, sex_id=BOTH_SEX_ID)
    
    # Concat and make quantile wide
    pop_da = xr.concat([pop_2017_da, pop_fut_sel], dim="year_id")
    
    pop_df = pop_da.rename("value").to_dataframe().reset_index()
    pop_df = pop_df.pivot_table(values="value",
                                index=["location_id", "year_id", "age_group_id",
                                       "sex_id", "scenario"],
                                columns="quantile").reset_index()
    
    # Combine value and UI into one column
    pop_with_ui = combine_mean_ui(pop_df)
    
    # Find peak pops and year of peak
    peak_pop_df = pop_with_ui.query("scenario == 0").groupby(
        "location_id").apply(
        get_max_pop_year).reset_index().rename(
        columns={"mean":"peak_pop","year_id":"peak_year"})
    
    peak_pop_df["peak_pop"] = peak_pop_df["peak_pop"].apply(
        lambda x: round(x, 2))
    peak_pop_df["peak_pop_value"] = (peak_pop_df["peak_pop"].astype(str) +
                                     "  (" +
                                     peak_pop_df["peak_year"].astype(
                                        int).astype(str) +
                                     ")")
    
    # Get 2017 and 2100 values
    pop_2017_only = pop_with_ui.query(f"year_id == {p_end} and scenario == 0")
    pop_2100_only = pop_with_ui.query(f"year_id == {f_end}")
    
    pop_2017_wide = pivot_scenarios(pop_2017_only, f"{p_end}", SCENARIO_MAP)
    pop_2100_wide = pivot_scenarios(pop_2100_only, f"{f_end}", SCENARIO_MAP)
    
    # Merge
    pop_final_df = pop_2017_wide.merge(peak_pop_df).merge(pop_2100_wide)
    
    return pop_final_df


def pull_reshape_tfr(gbd_round_id, tfr_version, location_ids):
    """Pulls year 2017 GBD round 5 TFR, converts it an xarray dataarray,
    pulls forecast TFR, and concatenates the dataarrays. The new array is
    then converted to a pandas dataframe. All required data are then reshaped
    and merged for downstream table production.

    Args:
        gbd_round_id (int):
            GBD round.
        tfr_version (str):
            Forecast TFR version.
        location_ids (list):
            List of location IDs to pull from both past and future data.
    Returns:
        tfr_final_df (pandas dataframe):
            Dataframe with all required TFR data, reshaped for downstream table
            production.
    """

    p_end = YEARS.past_end
    f_end = YEARS.forecast_end
    # Get 2017 GBD TFR
    tfr_2017 = get_covariate_estimates(covariate_id=149,
                                       gbd_round_id=gbd_round_id,
                                       location_id=location_ids, year_id=p_end,
                                       status="best")[[
    "year_id", "location_id","mean_value", "lower_value", "upper_value"
    ]].rename(columns={"mean_value":"mean", "lower_value":"lower",
                       "upper_value":"upper"})
    
    tfr_2017_da = melt_to_xarray(tfr_2017)
    
    # Get future TFR
    tfr_fut = open_xr(f"5/future/tfr/{tfr_version}/tfr_combined.nc").data
    
    tfr_fut_sel = tfr_fut.sel(location_id=location_ids, scenario=SCENARIOS,
                              year_id=YEARS.forecast_years)
    
    # Concat and make quantile wide
    tfr_da = xr.concat([tfr_2017_da, tfr_fut_sel], dim="year_id")
    
    tfr_df = tfr_da.to_dataframe().reset_index()
    tfr_df = tfr_df.pivot_table(values="value",
                                index=["location_id", "year_id", "scenario"],
                                columns="quantile").reset_index()
    
    # Combine value and UI into one column
    tfr_df = combine_mean_ui(tfr_df, df_type="tfr")
    
    # Get 2017 and 2100 values
    tfr2017 = tfr_df.query(f"year_id == {p_end} and scenario==0")
    tfr2100 = tfr_df.query(f"year_id == {f_end}")
    tfr2017 = pivot_scenarios(tfr2017, f"{p_end}", SCENARIO_MAP, df_type="tfr")
    tfr2100 = pivot_scenarios(tfr2100, f"{f_end}", SCENARIO_MAP, df_type="tfr")
    
    # Merge
    tfr_final_df = tfr2017.merge(tfr2100)
    
    return tfr_final_df


def convert_to_floating(string):
    """
    Takes a string with a decimal point and converts the decimal point to a
    floating decimal point for lancet style formatting.

    Args:
        string (str):
            A number string with a decimal point.
    Returns:
        str:
            The original string with floating decimal.
    """
    return "".join(["\u00b7" if char=="." else char for char in string])


def get_format_obj(workbook, font_name="Times New Roman", font_size=8,
                   bg_color="#FFFFFF", align=True, bold=False):
    """Utility function to dynamically create cell formatting options.

    Args:
        workbook (xlsxwriter Workbook):
            Parent workbook of the worksheet to which the data is written.
        font_name(str):
            Font of the content.
        font_size(int):
            Font size of the content.
        bg_color(str):
            String representing the HEX code of cell color.
        align(bool):
            If cell content needs to be vertically and horizontally aligned.
        bold (bool):
            If cell content needs to be boldened.
    Returns:
        format_obj (xlsxwriter workbook format object):
            Has specified format properties.
    """

    format_obj = workbook.add_format(
        {
            "font_name": font_name,
            "font_size": font_size
        }
    )
    format_obj.set_border()
    format_obj.set_text_wrap()
    format_obj.set_bg_color(bg_color)
    if bold:
        format_obj.set_bold()
    if align:
        format_obj.set_align("center")
        format_obj.set_align("vcenter")
    return format_obj


def write_header(worksheet, curr_row, cols, data_cols, header_format, stages):
    """Utility function to write the header for each page.

    Args:
        worksheet (Worksheet object):
            Worksheet to which the data is written.
        curr_row (int):
            Starting row number for the header.
        cols (list):
            List of characters representing the columns.
        data_cols (pandas series):
            Columns to be written.
        header_format(xlsxwriter Format object):
            Cell format options for headers.
        stages (list):
            "tfr", "pop" etc.
    Returns:
        int: An integer specifying the row number following the header.
    """

    ### Merge range function takes the locations of the cells to merge, the data
    ### to write and the cell format. A sample input would look like:
    ###      worksheet.merge_range("A0:B1", "Location", cell_format_obj)
    ### The above call will merge 4 cells: A0, A1, B0, B1 and fill it with the
    ### value "Location".  
    
    end_row = curr_row + CELL_HT["location"]
    row_range = cols[0] + str(curr_row) + ":" + cols[0] + str(end_row)
    worksheet.merge_range(row_range, "Location", header_format)
    
    num_pop_cols = sum(map(lambda i: "pop" in  i, data_cols)) - 1
    num_tfr_cols = sum(map(lambda i: "tfr" in  i, data_cols)) - 1

    col_end = 0
    for i, stage in enumerate(stages):
        
        if stage == "pop":
            unit_txt = " (in millions)"
            stage_txt = "Population"
            col_range = num_pop_cols
        else:
            unit_txt = ""
            stage_txt = "Total Fertility Rate"
            col_range = num_tfr_cols
            
        col_st = col_end + 1
        col_end = col_st + col_range
        
        curr_row_copy = curr_row
        end_row = curr_row_copy + CELL_HT["stage"]

        row_range = (
            cols[col_st] + str(curr_row_copy) + ":" +
            cols[col_end] + str(end_row)
        )
        
        col_txt = stage_txt + unit_txt
        worksheet.merge_range(row_range, col_txt, header_format)

        curr_row_copy = end_row + 1
        end_row = curr_row_copy + CELL_HT["stage"]
        
        col_st_copy = col_st
        
        for column in data_cols:
            if stage in column:                
                row_range = cols[col_st_copy] + str(curr_row_copy)
                worksheet.write(row_range, COL_NAME_MAP[column], header_format)
                col_st_copy += 1
            

    return end_row + 1


def write_table(final_df, outfile, stages):
    """Writes the data to an xlsx table.

    Args:
        final_df (pandas dataframe):
            Dataframe with formatted data.
        outfile (FBDPath object):
            Path to store the table.
        stages (list):
            "tfr", "pop" etc.
    """

    workbook = xlsxwriter.Workbook(
        str(outfile), {"constant_memory": False}
    )
    worksheet = workbook.add_worksheet("Table 1")

    header_color = "#F2DCDB"
    white = "#000000"
    black = "#FFFFFF"
    loc_cell_width = 20
    data_cell_width = 15
    column_start = 65

    header_format = get_format_obj(
        workbook, bg_color=header_color, font_size=12, bold=True
    )
    title_format = get_format_obj(
        workbook, bg_color=white, font_size=13, align=False, bold=True
    )
    title_format.set_font_color(black)
    
    # Column length is basically all columns in the dataframe except 'level'
    col_len = final_df.shape[1]-1
    
    data_cols = final_df.drop(["level", "lancet_label"], axis=1).columns.values
    
    cols = list(map(chr, range(column_start, column_start+col_len)))
    worksheet.set_column(cols[0]+":"+cols[0], loc_cell_width)
    worksheet.set_column(cols[1]+":"+cols[-1], data_cell_width)

    # place-holder to manually adjust title as needed
    title = (
        "Title goes here."
    )
    curr_row = 1
    end_row = curr_row + CELL_HT["title"]
    row_range = cols[0] + str(curr_row) + ":" + cols[-1] + str(end_row)
    worksheet.merge_range(row_range, title, title_format)

    curr_row = end_row+1
    page_row_count = 1
    page_breaks = []

    for _, row in final_df.iterrows():
        page_row_count += 1
        
        ### Insert page break after 20 rows.
        if row["level"] == 0 or (page_row_count != 0 and
                                 page_row_count % 20 == 0):
            page_row_count = 0
            page_breaks.append(curr_row - 1)
            curr_row = write_header(
                worksheet, curr_row, cols, data_cols,
                header_format, stages
            )
        end_row = curr_row + CELL_HT["data_cols"]
        col_idx = 0

        if row["level"] < 3:
            loc_fmt_obj = get_format_obj(
                workbook, font_size=11,
                bg_color=header_color, bold=True,
                align=False
            )
            data_fmt_obj = get_format_obj(
                workbook, font_size=11,
                bg_color=header_color, bold=True
            )
        else:
            loc_fmt_obj = get_format_obj(
                workbook, font_size=11, align=False
            )
            data_fmt_obj = get_format_obj(
                workbook, font_size=11
            )

        for col in final_df:
            if col == "level":
                continue

            row_range = (
                cols[col_idx] + str(curr_row) + ":" +
                cols[col_idx] + str(end_row)
            )
            if col == "lancet_label":
                loc_name = INDENT_MAP[row["level"]] + row[col]
                worksheet.merge_range(row_range, loc_name, loc_fmt_obj)
            else:
                worksheet.merge_range(row_range, row[col], data_fmt_obj)

            col_idx += 1
        curr_row = end_row+1

    worksheet.set_h_pagebreaks(page_breaks[1:])
    worksheet.fit_to_pages(1, 0)
    workbook.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pop-version", type=str, required=True,
        help="The version of population to use."
    )
    parser.add_argument("--tfr-version", type=str, required=True,
        help="The version of fertility to use."
    )
    parser.add_argument(
        "--gbd-round-id", type=int, required=True,
        help="The GBD round associated with the data.")
    parser.add_argument(
        "--output-review-table", action="store_true",
        help="Outputs table for reviewers along with Lancet-style table."
        )

    args = parser.parse_args()

    # Define stages (population and TFR)
    stages = ["pop", "tfr"]

    # Get location metadata
    gbd_loc_df = get_location_metadata(gbd_round_id=args.gbd_round_id,
                                       location_set_id=35)
    loc_meta = gbd_loc_df.query("level < 4")[
        ["location_id", "lancet_label", "level","sort_order"]]
    location_ids = loc_meta.location_id.tolist()

    # Make pop and TFR dataframes
    pop_final_df = pull_reshape_pop(
        args.gbd_round_id, args.pop_version, location_ids
        )
    tfr_final_df = pull_reshape_tfr(
        args.gbd_round_id, args.tfr_version, location_ids
        )

    # merge dataframes
    merged_df = loc_meta.merge(
        pop_final_df).merge(
        tfr_final_df,
        on="location_id", how="left").sort_values(by="sort_order")

    # Convert to floating decimal
    data_cols = ["value_2017_pop_ref", "value_2100_pop_ref",
                 "value_2100_pop_sdg", "peak_pop_value",
                 "value_2017_tfr_ref", "value_2100_tfr_ref",
                 "value_2100_tfr_sdg"]

    merged_df.loc[:, data_cols] = merged_df.loc[:, data_cols].applymap(
        lambda x: convert_to_floating(x)
        )

    # Order final dataframe
    final_df = merged_df[["level", "lancet_label"] + data_cols]

    # Write table
    plot_dir = (f"/ihme/forecasting/plot/{args.gbd_round_id}"
                "/future/population/table_1/")
    fname = date.today().strftime("%Y%m%d") + "_table_1.xlsx"
    filepath = plot_dir + fname

    try:
        mkdir(plot_dir)
        print(f"{plot_dir} created.") 
    except FileExistsError:
        print(f"{plot_dir} already exists.")

    write_table(final_df, filepath, stages)

    if args.output_review_table:
        review_fname = date.today().strftime("%Y%m%d") + "_table_1_review.csv"
        review_cols = list(REVIEW_COL_NAME_MAP.keys())
        review_df = merged_df[review_cols].drop_duplicates()
        review_df.rename(columns=REVIEW_COL_NAME_MAP, inplace=True)
        review_df.to_csv(plot_dir + review_fname, index=False)
"""
This program will create a lancet style table for total fertility rates
per country for the years 1990, 2017, 2050, and 2100 for two scenarios
reference and faster(85). Accepts optional reviewer_cols flag
which outputs a basic csv containing mean, lower, and upper data as separate
columns
"""
import datetime

import pandas as pd
import xlsxwriter
import xarray as xr

from db_queries import get_location_metadata
from fbd_core import YearRange, db, argparse
from fbd_core.db import get_locations_by_max_level
from fbd_core.etl import aggregator
from fbd_core.file_interface import FBDPath, open_xr

SCENARIO_MAP = [0, 1]
# Scenario for this particular fertility table is REFERENCE (O) and
# BETTER (1) may need to change depending on the dataset imported or
# desired table

CELL_HT = {
    "title": 3,
    "location": 0,
    "cause": 3,
    "stage": 0,
    "data_cols": 2
}
# dict: Python dictionary for mapping cell heights to their cell types.
# this is required while creating the table in an excel worksheet

INDENT_MAP = {
    0: "",
    1: "  ",
    2: "    ",
    3: "      "
}

# Python dictionary for mapping indentation levels to their corresponding
# cause levels.

YEARS_PAST = [1990, 2017]
YEARS_FUT = [2050, 2100]


def compile_data(tfr_fut_version, tfr_past_version, reviewer_cols):
    """Creates dataframe of total fertility rate data to be converted into
        an excel xlsx files
    Args:
        tfr_fut_version (str):
            the version name of total fertility forecast data to use
        tfr_past_version (str):
            the version name of total fertility past data to use
    Returns:
        final dataframe to save as an excel file
    """
    future_tfr_fbdpath = FBDPath(f"/5/future/tfr/{tfr_fut_version}/"
                                 "tfr_combined.nc")
    past_tfr_fbdpath = FBDPath(f"5/past/tfr/{tfr_past_version}/tfr.nc")

    gbd_loc_df = get_location_metadata(gbd_round_id=5, location_set_id=35)
    gbd_loc_hierarchy = gbd_loc_df.query(
        "level < 4").set_index("location_id").to_xarray().parent_id

    fut_fertility = open_xr(
        future_tfr_fbdpath).data.sel(year_id=YEARS_FUT, scenario=SCENARIO_MAP,
                                     location_id=gbd_loc_hierarchy.location_id)

    past_fertility = open_xr(past_tfr_fbdpath).data.sel(
        year_id=YEARS_PAST,
        location_id=gbd_loc_hierarchy.location_id).expand_dims(scenario=[0, 1])

    past_fertility_mn = past_fertility.mean("draw")

    past_fertility_lims = past_fertility.quantile([0.025, 0.975], dim="draw")
    past_fertility_lims.coords["quantile"] = ["lower", "upper"]

    fut_fertility_df = fut_fertility.to_dataframe(
        name="mean").reset_index().rename(columns={"value": "mean"})

    fut_lims_df = fut_fertility_df.pivot_table(values="mean",
                                               index=["location_id",
                                                      "year_id",
                                                      "scenario"],
                                               columns="quantile"
                                               ).reset_index()

    past_fertility_df = past_fertility_mn.to_dataframe(
        name="mean").reset_index().rename(
            columns={"value": "mean"})
    past_lims_df = past_fertility_lims.to_dataframe(
        "past_lims").reset_index().rename(columns={"value": "mean"})
    past_lims_df = past_lims_df.pivot_table(values="past_lims",
                                            index=["location_id",
                                                   "year_id",
                                                   "scenario"],
                                            columns="quantile").reset_index()
    past_tfr_lims = past_fertility_df.merge(past_lims_df, how="left")
    if reviewer_cols:
        tfr_cols = ["location_id", "scenario",
                    "mean", "lower", "upper",
                    "year_id"]
        val_cols = ["mean", "lower", "upper"]
        past_df_val = add_ui(past_tfr_lims)[tfr_cols]
        fut_df_val = add_ui(fut_lims_df)[tfr_cols]
        combined_tfr = past_df_val.merge(fut_df_val, how="outer",
                                         on=["year_id", "location_id",
                                             "scenario", "mean",
                                             "lower", "upper"])
        combined_tfr_pivot = pivot_scenarios(combined_tfr, val_cols)
        combined_tfr_pivot.columns = [f"{j}_{i}" if j != ""
                                      else f"{i}" for i, j in
                                      combined_tfr_pivot.columns]
    else:
        get_lims(past_tfr_lims)
        get_lims(fut_lims_df)
        tfr_cols = ["location_id", "scenario", "value", "year_id"]
        past_df_val = add_ui(past_tfr_lims)[tfr_cols]
        fut_df_val = add_ui(fut_lims_df)[tfr_cols]

        combined_tfr = past_df_val.merge(fut_df_val, how="outer",
                                         on=["year_id", "location_id",
                                             "scenario", "value"])

        combined_tfr_pivot = pivot_scenarios(combined_tfr, "value")

        combined_tfr_pivot = combined_tfr_pivot[
            ["location_id", "1990_ref", "2017_ref", "2050_ref", "2100_ref",
             "1990_better", "2017_better", "2050_better", "2100_better"]]

    final_df = gbd_loc_df.merge(
       combined_tfr_pivot).sort_values("sort_order")
    return (final_df)


def floating_style(list_nums=None):
    """
    Convert the decimal point in the UI to a lancet style floating decimal for
    both past and future data
    """
    floating_style = []
    for num in list_nums:
        str_num = str(round(num, 2))
        per_idx = str_num.find(".", 0, len(str_num))
        format_num = (str_num[:per_idx] + "\u00b7" +
                      str_num[per_idx+1:per_idx+3])
        floating_style.append(format_num)
    return floating_style


def get_lims(lims_df=None):
    for value in ["mean", "lower", "upper"]:
        value_lims = lims_df[value]
        lims_df[value] = floating_style(value_lims)


def add_ui(df, df_type="mean"):
    """
    Function to add ui to the mean column creating one value ex:
    "2.17 (lowerui - upperui)"
    """
    for col in ["mean", "lower", "upper"]:
        if df_type == "mean":
            df[col] = df[col]
        # df[col] = df[col].apply(lambda x: round(x, 2))

    df["value"] = (df["mean"].astype(str) + "  (" + df["lower"].astype(str)
                   + " - " + df["upper"].astype(str) + ")")
    return df


def pivot_scenarios(df, value):
    """
    Function used to pivot the table from having scenario per row to cols
    (ref and better) concats scenario to col name
    """
    ref_map = {
        0: "ref", 1: "better", 1990: "1990",
        2017: "2017", 2050: "2050", 2100: "2100"
    }

    df["year_scenario"] = (df["year_id"].map(ref_map) + "_"
                           + df["scenario"].map(ref_map))
    df = df.pivot_table(values=value,
                        index=["location_id"],
                        columns=["year_scenario"],
                        aggfunc="first").reset_index()
    return df


def get_format_obj(workbook, font_name="Times New Roman", font_size=8,
                   bg_color="#FFFFFF", align=True, bold=False):
    """Utility function to dynamically create cell formatting options.

    Args:
        workbook (xlsxwriter Workbook): Parent workbook of the
            worksheet to which the data is written.
        font_name(str): Font of the content.
        font_size(int): Font size of the content.
        bg_color(str): String representing the HEX code of cell color.
        align(bool): If cell content needs to be vertically and horizontally
            aligned.
        bold (bool): If cell content needs to be boldened.

    Returns:
        format_obj. xlsxwriter workbook format object with specied format
            properties.
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


def write_header(worksheet, curr_row, cols, data_cols,
                 header_format, stages, col_name_map):
    """Utility function to write the header for each page.

    Args:
        worksheet (Worksheet object): Worksheet to which the data is written.
        curr_row(int): Starting row number for the header.
        cols(list): List of characters representing the columns.
        data_cols(Series): Row containing data to be written.
        header_format(xlsxwriter Format object): Cell format options for
            headers.
        stages (list):  "tfr", "pop" etc.
        years (YearRange): YearRange object. Ex: YearRange(2010, 2017, 2020).
    Returns:
        int: An integer specifying the row number following the header.
    """

    # Merge range function takes the locations of the cells to merge, the data
    # to write and the cell format. For ag. a sample input would look like:
    # worksheet.merge_range("A0:B1", "Location", cell_format_obj)
    # The above call will merge 4 cells: A0, A1, B0, B1 and fill it with
    # the value "Location".

    end_row = curr_row + CELL_HT["location"]
    row_range = cols[0] + str(curr_row) + ":" + cols[0] + str(end_row)
    worksheet.merge_range(row_range, "Location", header_format)

    num_ref_cols = sum(map(lambda i: "ref" in i, data_cols)) - 1
    num_bet_cols = sum(map(lambda i: "better" in i, data_cols)) - 1

    col_end = 0
    for i, stage in enumerate(stages):

        if stage == "ref":
            stage_txt = "Total Fertility Rate - Reference"
            col_range = num_ref_cols
        elif stage == "better":
            stage_txt = ("Total Fertility Rate - Faster Met Need "
                         "and Education Pace")
            col_range = num_bet_cols

        col_st = col_end + 1
        col_end = col_st + col_range

        curr_row_copy = curr_row
        end_row = curr_row_copy + CELL_HT["stage"]

        row_range = (
            cols[col_st] + str(curr_row_copy) + ":" +
            cols[col_end] + str(end_row)
        )

        col_txt = stage_txt
        worksheet.merge_range(row_range, col_txt, header_format)

        curr_row_copy = end_row + 1
        end_row = curr_row_copy + CELL_HT["stage"]

        col_st_copy = col_st

        for column in data_cols:
            if stage in column:
                row_range = cols[col_st_copy] + str(curr_row_copy)
                worksheet.write(row_range, col_name_map[column], header_format)
                col_st_copy += 1

    return end_row + 1


def write_table(final_df, outfile, stages, years, col_name_map):
    """Writes the data to an xlsx table.
    Args:
        final_df (DataFrame): Dataframe with formatted data.
        outfile(FBDPath): Path to store the table.
        stages (list):  "tfr", "pop" etc.
        years (YearRange): YearRange object. Ex: YearRange(2010, 2017, 2020).
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

    # Column length is basically all columns in the dataframe
    # except column `level`
    col_len = final_df.shape[1]-1

    data_cols = final_df.drop(["level", "lancet_label"], axis=1).columns.values

    cols = list(map(chr, range(column_start, column_start+col_len)))
    worksheet.set_column(cols[0]+":"+cols[0], loc_cell_width)
    worksheet.set_column(cols[1]+":"+cols[-1], data_cell_width)

    title = (
        "Table X. Total fertility rate in the reference forecast and faster "
        "met need and education pace scenario: 1990, 2017, 2050 and 2100. Past"
        " estimates are from the Global Burden of Disease (GBD) 2017 study."
        " Estimates are listed as means with 95% uncertainty intervals "
        "in parentheses. Highlighted rows indicate region and super region "
        "results from the GBD location hierarchy."
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

        # Insert page break after 34 rows.
        if row["level"] == 0 or (page_row_count != 0 and
                                 page_row_count % 23 == 0):
            page_row_count = 0
            page_breaks.append(curr_row - 1)
            curr_row = write_header(
                worksheet, curr_row, cols, data_cols,
                header_format, stages, col_name_map
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


def main(tfr_fut_version, tfr_past_version, reviewer_cols):
    """ Generate an excel worksheet with the title outfile ("test" followed by
    date and current time to the second)
    Args:
        tfr_fut_version (str):
            The version of total fertility rate forecast to use
        tfr_past_version (str):
            The version of total fertility rate past data to use
    Output: None.
    Current issue(s): will not write "Location" boxes in final sheet leaving
    blank cells where a salmon colored, bold, text 'Location' should be
    """

    stages = ["ref", "better"]
    year_ids = YearRange(1990, 2018, 2100)

    if reviewer_cols:
        outfile = ("TFR_reviewer_table_%s.csv" %
                   (datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")))
        col_name_map = {
         "lancet_label": "Location",
         "1990_ref_mean": "1990 reference mean",
         "1990_ref_lower": "1990 reference lower",
         "1990_ref_upper": "1990 reference upper",
         "2017_ref_mean": "2017 reference mean",
         "2017_ref_lower": "2017 reference lower",
         "2017_ref_upper": "2017 reference upper",
         "2050_ref_mean": "2050 reference mean",
         "2050_ref_lower": "2050 reference lower",
         "2050_ref_upper": "2050 reference upper",
         "2100_ref_mean": "2100 reference mean",
         "2100_ref_lower": "2100 reference lower",
         "2100_ref_upper": "2100 reference upper",
         "1990_better_mean": "1990 faster met need mean",
         "1990_better_lower": "1990 faster met need lower",
         "1990_better_upper": "1990 faster met need upper",
         "2017_better_mean": "2017 faster met need mean",
         "2017_better_lower": "2017 faster met need lower",
         "2017_better_upper": "2017 faster met need upper",
         "2050_better_mean": "2050 faster met need mean",
         "2050_better_lower": "2050 faster met need lower",
         "2050_better_upper": "2050 faster met need upper",
         "2100_better_mean": "2100 faster met need mean",
         "2100_better_lower": "2100 faster met need lower",
         "2100_better_upper": "2100 faster met need upper"
        }

        cols_to_write = ["lancet_label", "1990_ref_mean",
                         "1990_ref_lower", "1990_ref_upper",
                         "2017_ref_mean", "2017_ref_lower",
                         "2017_ref_upper", "2050_ref_mean",
                         "2050_ref_lower", "2050_ref_upper",
                         "2100_ref_mean", "2100_ref_lower",
                         "2100_ref_upper", "1990_better_mean",
                         "1990_better_lower", "1990_better_upper",
                         "2017_better_mean", "2017_better_lower",
                         "2017_better_upper", "2050_better_mean",
                         "2050_better_lower", "2050_better_upper",
                         "2100_better_mean", "2100_better_lower",
                         "2100_better_upper"]
        df = compile_data(
                tfr_fut_version,
                tfr_past_version,
                reviewer_cols)[cols_to_write].round(2)
        df.rename(columns=col_name_map, inplace=True)
        df = df.set_index("Location")
        df.to_csv(outfile)
    else:
        outfile = ("TFR_combined_table_%s.xlsx" %
                   (datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")))
        # Dictionary used to create worksheet column names from df names
        col_name_map = {
         "lancet_label": "Location",
         "1990_ref": "1990",
         "2017_ref": "2017",
         "2050_ref": "2050",
         "2100_ref": "2100",
         "1990_better": "1990",
         "2017_better": "2017",
         "2050_better": "2050",
         "2100_better": "2100"
        }

        cols_to_write = ["level", "lancet_label", "1990_ref", "2017_ref",
                         "2050_ref", "2100_ref", "1990_better", "2017_better",
                         "2050_better", "2100_better"]

        df = compile_data(tfr_fut_version,
                          tfr_past_version,
                          reviewer_cols)[cols_to_write]
        write_table(df, outfile, stages, year_ids, col_name_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--tfr_fut",
        type=str,
        required=True,
        help="Name of FBD future tfr data file. Taken as name of "
             "directory data file is found in \'.nc\' is not required"
    )
    parser.add_argument(
        "--tfr_past",
        type=str,
        required=True,
        help="Name of FBD past tfr data file. Taken as name of "
             "directory data file is found in \'.nc\' is not required"
    )
    parser.add_argument(
        "--reviewer_cols",
        action="store_true",
        help="Pass if desired output is a csv containin mean, lower, upper as"
             " seperate columns"
    )
    args = parser.parse_args()

    main(tfr_fut_version=args.tfr_fut, tfr_past_version=args.tfr_past,
         reviewer_cols=args.reviewer_cols)

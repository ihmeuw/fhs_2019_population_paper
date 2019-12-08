"""
Life Expectancy Table by country.
Script to create reference table for Life expectancy for males, females,
and both genders for locations (Global, superregions, regions, and countries)
for the years 1990, 2017, 2050, 2100. Calculating and adding the uncertainty
intervals (ui) to the final table figures

example call:
python lex_table.py
--lex_past 20191029_fhs_computed_from_20190109_version90_etl_gbd4_all_youth
--lex_fut 20191030_15_ref_85_resub_agg_arima_squeeze_shocks_only_decay_wt_15_hiv_all_gbd4_extrap
"""

import datetime

import xarray as xr
import pandas as pd
import xlsxwriter

from db_queries import get_location_metadata
from fbd_core import YearRange, argparse
from fbd_core.file_interface import FBDPath, open_xr

CELL_HT = {
    "title": 3,
    "location": 1,
    "stage": 0,
    "data_cols": 2
}
"""dict: Python dictionary for mapping cell heights to their cell types.
    This is required while creating the excel file
"""

INDENT_MAP = {
    0: "",
    1: "  ",
    2: "    ",
    3: "      "
}
"""dict: Python dictionary for mapping indentation levels to their
    corresponding cause levels. Used for formatting the "Cause" column
    in the excel file.
"""


def pivot_on_col(df):
    """
    Args:
        df (pandas dataframe):
            data frame containing columns to pivot from long to wide
    Returns:
        df pivoted to long format with renamed column names
    """
    ref_map = {1990: "1990", 2017: "2017", 2050: "2050", 2100: "2100",
               1: "Male", 2: "Female", 3: "Both"}
    df[value] = df["sex_id"].map(ref_map) + "_" + df["year_id"].map(ref_map)
    df = df.pivot_table(values="value",
                        index=["location_id"],
                        columns=["sex_id"], aggfunc="first").reset_index()
    return df


def add_ui(df):
    """Creates UI column in lancet style with floating point
    Args:
        df (dataframe):
            dataframe containing mean, lower and upper values to create
            floating uncertainty intervals tfr_combined
    Returns:
        df (dataframe):
            dataframe with new "value" column containing a string formated with
            uncertainty interval in the lancet style e.g. 63·2 (63·1 - 63·4)
    """
    for col in ["mean", "lower", "upper"]:
        df[col] = df[col]

    df["value"] = (df["mean"].astype(str) +
                   "  (" + df["lower"].astype(str) + " - "
                   + df["upper"].astype(str)+")")
    return df


def floating_style(list_nums=None):
    floating_style = []
    for num in list_nums:
        str_num = "%.1f" % round(num, 1)
        per_idx = str_num.find(".", 0, len(str_num))
        format_num = (str_num[:per_idx] + "\u00b7" +
                      str_num[per_idx+1:per_idx+2])
        floating_style.append(format_num)
    return floating_style


def get_lims(lims_df=None):
    for value in ["mean", "lower", "upper"]:
        value_lims = lims_df[value]
        lims_df[value] = floating_style(value_lims)


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


def compile_data(past_lex, fut_lex):
    """Creates dataframe of life expectancy data to be converted into an excel
    xlsx files
    Args:
        past_lex (str):
            the version name of the life expectancy past data to use
        fut_lex (str):
            the version name of the life expectancy forecast data to use
    Returns:
        final dataframe to save as an excel sheet
    """
    past_lex_fbdpath = FBDPath(f"/5/past/life_expectancy/{past_lex}/lex.nc")
    fut_lex_fbdpath = FBDPath(f"/5/future/life_expectancy/{fut_lex}/"
                              "lex_agg.nc")
    gbd_loc_df = get_location_metadata(gbd_round_id=5, location_set_id=35)
    gbd_loc_hierarchy = gbd_loc_df.query(
        "level < 4").set_index("location_id").to_xarray().parent_id
    # Setting the dims and coords we want
    sexes = [1, 2, 3]  # all sex gropus
    yearsFut = [2050, 2100]  # Future years
    yearsPast = [1990, 2017]  # Past years
    age = [2]  # Life expectancy from early neonatal age group

    past_life_expec = open_xr(past_lex_fbdpath).data
    past_life_expec = past_life_expec.expand_dims(scenario=[0])
    fut_life_expec = open_xr(fut_lex_fbdpath).data
    past_life = past_life_expec.sel(year_id=yearsPast,
                                    location_id=gbd_loc_hierarchy.location_id,
                                    sex_id=sexes,
                                    age_group_id=age)

    fut_life = fut_life_expec.sel(year_id=yearsFut,
                                  location_id=gbd_loc_hierarchy.location_id,
                                  sex_id=sexes,
                                  scenario=0,
                                  age_group_id=age)
    fut_lex_mn = fut_life.mean("draw")
    past_lex_mn = past_life.mean("draw")

    fut_lex_lims = fut_life.quantile([0.025, 0.975], dim="draw")
    fut_lex_lims.coords["quantile"] = ["lower", "upper"]

    past_lex_lims = past_life.quantile([0.025, 0.975], dim="draw")
    past_lex_lims.coords["quantile"] = ["lower", "upper"]

    fut_lex_df = fut_lex_mn.to_dataframe(
        name="mean").reset_index().rename(columns={"value": "mean"})
    fut_lex_lims_df = fut_lex_lims.to_dataframe().reset_index().rename(
        columns={"ex": "mean"})
    fut_lex_lims_df = fut_lex_lims_df.pivot_table(
        values="mean", index=["location_id", "year_id", "sex_id"],
        columns="quantile").reset_index()

    fut_comb_lex = fut_lex_df.merge(fut_lex_lims_df, how="left")

    past_lex_df = past_lex_mn.to_dataframe(
        name="mean").reset_index().rename(columns={"value": "mean"})
    past_lex_lims_df = past_lex_lims.to_dataframe().reset_index().rename(
        columns={"ex": "mean"})
    past_lex_lims_df = past_lex_lims_df.pivot_table(values="mean",
                                                    index=["location_id",
                                                           "year_id",
                                                           "sex_id"],
                                                    columns="quantile"
                                                    ).reset_index()

    past_comb_lex = past_lex_df.merge(past_lex_lims_df, how="left")
    get_lims(past_comb_lex)
    get_lims(fut_comb_lex)
    lex_cols = ["location_id", "value", "year_id", "sex_id"]
    past_df_val = add_ui(past_comb_lex)[lex_cols]
    fut_df_val = add_ui(fut_comb_lex)[lex_cols]
    combined_lex = past_df_val.merge(
        fut_df_val, how="outer", on=["year_id", "location_id",
                                     "value", "sex_id"])
    pivot_lex = pivot_on_col(combined_lex)
    pivot_lex = pivot_lex[["location_id",
                           "Male_1990",
                           "Male_2017",
                           "Male_2050",
                           "Male_2100",
                           "Female_1990",
                           "Female_2017",
                           "Female_2050",
                           "Female_2100",
                           "Both_1990",
                           "Both_2017",
                           "Both_2050",
                           "Both_2100"]]
    final_df = gbd_loc_df.merge(pivot_lex).sort_values("sort_order")
    return final_df


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
    #      worksheet.merge_range("A0:B1", "Location", cell_format_obj)
    # The above call will merge 4 cells: A0, A1, B0, B1 and fill it with the
    # value "Location".

    end_row = curr_row + CELL_HT["location"]
    row_range = cols[0] + str(curr_row) + ":" + cols[0] + str(end_row)
    worksheet.merge_range(row_range, "Location", header_format)

    num_m_cols = sum(map(lambda i: "Male" in i, data_cols)) - 1
    num_f_cols = sum(map(lambda i: "Female" in i, data_cols)) - 1
    num_b_cols = sum(map(lambda i: "Both" in i, data_cols)) - 1

    col_end = 0
    for i, stage in enumerate(stages):

        if stage == "Male":
            stage_txt = "Males"
            col_range = num_m_cols
        elif stage == "Female":
            stage_txt = "Females"
            col_range = num_f_cols
        elif stage == "Both":
            stage_txt = "Both"
            col_range = num_b_cols

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

    # Column length is basically all columns in the dataframe except
    # column `level`
    col_len = final_df.shape[1]-1

    data_cols = final_df.drop(["level", "lancet_label"], axis=1).columns.values

    cols = list(map(chr, range(column_start, column_start+col_len)))
    worksheet.set_column(cols[0]+":"+cols[0], loc_cell_width)
    worksheet.set_column(cols[1]+":"+cols[-1], data_cell_width)

    title = (
        "Table X. Life expectancy by sex in the reference forecast. Values are"
        " presented as life expectancy at birth, measured in years, with 95% "
        "uncertainty intervals in parentheses. Past estimates are from the "
        "Global Burden of Disease (GBD) 2017 study. Highlighted rows indicate "
        "region and super region results from the GBD location hierarchy."
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

        # Insert page break after 35 rows.
        if row["level"] == 0 or (page_row_count != 0 and
                                 page_row_count % 35 == 0):
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


def main(past_lex, fut_lex):
    """
    Args:
        past_lex (str):
            The version name of past life expectancy data to use
        fut_lex (str):
            the version name of life expectancy forecasts to use
    Returns: None.
    Output: Creates xlsx file in directory script is called from
    """
    outfile = ("LEX_pb35_%s.xlsx"
               % (datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")))
    stages = ["Male", "Female", "Both"]
    year_ids = YearRange(1990, 2018, 2100)

    # Dictionary used to map worksheet column names from df names
    col_name_map = {"lancet_label": "Location",
                    "Male_1990": "1990",
                    "Male_2017": "2017",
                    "Male_2050": "2050",
                    "Male_2100": "2100",
                    "Female_1990": "1990",
                    "Female_2017": "2017",
                    "Female_2050": "2050",
                    "Female_2100": "2100",
                    "Both_1990": "1990",
                    "Both_2017": "2017",
                    "Both_2050": "2050",
                    "Both_2100": "2100"}

    cols_to_write = ["level", "lancet_label", "Male_1990", "Male_2017",
                     "Male_2050", "Male_2100", "Female_1990", "Female_2017",
                     "Female_2050", "Female_2100", "Both_1990", "Both_2017",
                     "Both_2050", "Both_2100"]

    df = compile_data(past_lex, fut_lex)[cols_to_write]
    write_table(df, outfile, stages, year_ids, col_name_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--lex_fut",
        type=str,
        required=True,
        help="Name of FBD future lex data file. Taken as name of "
             "directory data file is found in \'.nc\' is not required"
    )
    parser.add_argument(
        "--lex_past",
        type=str,
        required=True,
        help="Name of FBD past lex data file. Taken as name of "
             "directory data file is found in \'.nc\' is not required"
    )
    args = parser.parse_args()

    main(past_lex=args.lex_past, fut_lex=args.lex_fut)

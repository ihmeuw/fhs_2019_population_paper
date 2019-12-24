"""
This code creates appendix life tables in lancet formatted excel sheets for all
forecasting locations. A single xlsx file is created for each location and
given the filename "sort-order_loc-name_year" that is put in the output
directory. Using a vba script to merge all resulting excel files you can remove
the row of "-" included for formatting purposes then print as pdf to get the
final pdf table result. I've included the vba script used to merge
all xlsx files in this program as a comment at the bottom with brief
instructions for use
Author Sam Farmer
"""


import pandas as pd
import numpy as np
import xlsxwriter
import os
from fbd_core import YearRange, argparse
from db_queries import get_location_metadata as go


CELL_HT = {
    "title": 3,
    "location": 1,
    "stage": 0,
    "data_cols": 2
}
# dict: Python dictionary for mapping cell heights to their cell types.
# This is required while creating the table

INDENT_MAP = {
    0: "",
    1: "  ",
    2: "    ",
    3: "      "
}

gbd_loc_df = go(gbd_round_id=5, location_set_id=39)[["sort_order",
                                                     "lancet_label",
                                                    "location_id"]].rename(
                                                        {"lancet_label":
                                                         "Location"}, axis=1)


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


def write_header(worksheet, curr_row, cols, data_cols, header_format,
                 stages, col_name_map):
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

    # Merge range function takes the locations of the cells to merge, the
    # data to write and the cell format. For ag. a sample input would look
    # like:     worksheet.merge_range("A0:B1", "Location", cell_format_obj)
    # The above call will merge 4 cells: A0, A1, B0, B1 and fill it with the
    # value "Location".
    end_row = curr_row + CELL_HT["location"]
    row_range = cols[0] + str(curr_row) + ":" + cols[0] + str(end_row)
    worksheet.merge_range(row_range, "Age Group", header_format)

    num_male_cols = sum(map(lambda i: "Male" in i, data_cols)) - 1
    num_female_cols = sum(map(lambda i: "Female" in i, data_cols)) - 1

    col_end = 0
    for i, stage in enumerate(stages):

        if stage == "Male":
            stage_txt = "Male"
            col_range = num_male_cols
        else:
            stage_txt = "Female"
            col_range = num_female_cols

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

    location = str(final_df["Location"][1])
    year = str(int(final_df["Year"][1]))
    data_cols = final_df.drop(["level"], axis=1).columns.values
    final_df = final_df.drop(["Year", "Location"], axis=1)
    col_len = final_df.shape[1]-1

    excel_cols = list(map(chr, range(column_start, column_start+col_len)))
    worksheet.set_column(excel_cols[0]+":"+excel_cols[0], loc_cell_width)
    worksheet.set_column(excel_cols[1]+":"+excel_cols[-1], data_cell_width)

    title = (
        f"Table 14: {location} {year} life table, by age and sex. "
        "mx=mortality rate, ax=mean person-years lived in an age "
        "interval among those who die in that age interval,  "
        "lx=number of persons left alive at age x, nLx=person-years "
        "lived between age x and x+n, ex=life expectancy at age x."
    )
    curr_row = 1
    end_row = curr_row + CELL_HT["title"]
    row_range = (excel_cols[0] + str(curr_row) + ":" +
                 excel_cols[-1] + str(end_row))
    worksheet.merge_range(row_range, title, title_format)

    curr_row = end_row+1
    page_row_count = 1
    page_breaks = []

    for i, row in final_df.iterrows():
        page_row_count += 1

        # Insert page break after 75 rows.
        if row["level"] == 0 or (page_row_count != 0 and
                                 page_row_count % 75 == 0):
            page_row_count = 0
            page_breaks.append(curr_row - 1)
            curr_row = write_header(
                worksheet, curr_row, excel_cols, data_cols,
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
                excel_cols[col_idx] + str(curr_row) + ":" +
                excel_cols[col_idx] + str(end_row)
            )
            if col == "Location":
                loc_name = INDENT_MAP[row["level"]] + row[col]
                worksheet.merge_range(row_range, loc_name, loc_fmt_obj)
            else:
                worksheet.merge_range(row_range, row[col], data_fmt_obj)

            col_idx += 1
        curr_row = end_row+1

    worksheet.set_h_pagebreaks(page_breaks[1:])
    worksheet.set_landscape()
    worksheet.fit_to_pages(1, 0)
    workbook.close()


def sort_age_groups(df):
    """Creates categorical orginization for age groups in df
    Args:
        df (dataframe): dataframe containing Age Group column
    returns:
        df (dataframe): dataframe sorted by age group
    """
    df["Age Group"] = pd.Categorical(df["Age Group"],
                                     ("Early Neonatal", "Late Neonatal",
                                      "Post Neonatal", "1 to 4", "5 to 9",
                                      "10 to 14", "15 to 19", "20 to 24",
                                      "25 to 29", "30 to 34", "35 to 39",
                                      "40 to 44", "45 to 49", "50 to 54",
                                      "55 to 59", "60 to 64", "65 to 69",
                                      "70 to 74", "75 to 79", "80 to 84",
                                      "85 to 89", "90 to 94", "95 plus"))
    return (df.sort_values("Age Group"))


def blank_row(df):
    """
    This function inserts a row of "-" into the first row of
    the dataframe. This is required to have the column labels
    show up in the in the worksheet that is generated. Can be removed once
    tables are completed.
    Args:
        df (dataframe): dataframe to be altered
    Returns:
        df (dataframe): dataframe with first row containing only "-"
    """
    df.loc[-1] = (pd.Series("-"))
    df.index = df.index + 1
    return (df.sort_index().fillna("-"))


def make_table(passed_locs_df, outfile):
    """
    Args:
        passed_locs_df (dataframe): dataframe containing the per location life
        table information
        outfile (str): String name of the excel file to be saved
    Returns: None
    """
    stages = ["Male", "Female"]
    year_ids = YearRange(1990, 2018, 2100)

    # Dictionary used to create worksheet column names from df names
    col_name_map = {
        "Male-mx": "mx",
        "Male-ax": "ax",
        "Male-lx": "lx",
        "Male-nLx": "nLx",
        "Male-ex": "ex",
        "Female-mx": "mx",
        "Female-ax": "ax",
        "Female-lx": "lx",
        "Female-nLx": "nLx",
        "Female-ex": "ex"
        }

    final_df = passed_locs_df
    write_table(final_df, outfile, stages, year_ids, col_name_map)


def main(directory):
    """
    Args:
        directory (str): directory containing separate lifetables per location,
        and year e.g. (1_global_2017)
    Returns: None
    """
    if not os.path.isdir("output"):
        os.makedirs("output", exist_ok=True)
    for subdirs, dirs, files in os.walk(csvs):
        for filename in files:
            if os.path.isfile(csvs+filename) and filename[0] != ".":
                file_split = filename.split("_")
                country = file_split[1]
                loc_id = int(file_split[2])
                year = file_split[3].split(".")[0]
                data = pd.read_csv(csvs + filename)
                data["level"] = 3
                data = sort_age_groups(data)
                data = sort_age_groups(data).reset_index()
                data = blank_row(data).drop("index", axis=1)
                data.iat[0, 13] = 0
                sort_order = int(gbd_loc_df[
                    gbd_loc_df["location_id"] == loc_id]["sort_order"])
                outfile = (f"output/{sort_order}_{country}_{year}
                           _table.xlsx")
                make_table(data, outfile)


if __name__ == "__main__":
    main("/snfs1/temp/steuben/timetable_popresub/")


"""
The below vba script prompts the user to choose which excel files they wish to
copy in the directory that the xlsm is placed in and inserts them as a new
sheet in the workbook. To use, copy and past all code following the first
'Sub' to the 'End Sub' line into an xlsm file's visual basic editor under the
developer tab then select it from the macro option in the developer tab.

Sub MergeExcelFiles()
    Dim fnameList, fnameCurFile As Variant
    Dim countFiles, countSheets As Integer
    Dim wksCurSheet As Worksheet
    Dim wbkCurBook, wbkSrcBook As Workbook

    fnameList = Application.GetOpenFilename(FileFilter:="Microsoft Excel Workbooks (*.xls;*.xlsx;*.xlsm),*.xls;*.xlsx;*.xlsm", Title:="Choose Excel files to merge", MultiSelect:=True)

    If (vbBoolean <> VarType(fnameList)) Then

        If (UBound(fnameList) > 0) Then
            countFiles = 0
            countSheets = 0

            Application.ScreenUpdating = False
            Application.Calculation = xlCalculationManual

            Set wbkCurBook = ActiveWorkbook

            For Each fnameCurFile In fnameList
                countFiles = countFiles + 1

                Set wbkSrcBook = Workbooks.Open(Filename:=fnameCurFile)

                For Each wksCurSheet In wbkSrcBook.Sheets
                    countSheets = countSheets + 1
                    wksCurSheet.Copy after:=wbkCurBook.Sheets(wbkCurBook.Sheets.Count)
                Next

                wbkSrcBook.Close SaveChanges:=False

            Next

            Application.ScreenUpdating = True
            Application.Calculation = xlCalculationAutomatic

            MsgBox "Processed " & countFiles & " files" & vbCrLf & "Merged " & countSheets & " worksheets", Title:="Merge Excel files"
        End If

    Else
        MsgBox "No files selected", Title:="Merge Excel files"
    End If
End Sub

"""

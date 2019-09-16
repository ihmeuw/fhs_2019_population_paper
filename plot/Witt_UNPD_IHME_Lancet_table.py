"""
This code creates a lancet style table for comparison between UNPD, Wittgenstein, and IHME forcasts of Total
fertility rate, and population in millions by country. It can create a superregions only table or all regions table.

It takes in 8 command line arguments: 4 population datasets (--fbd-pop, --witt-pop, --wpp-pop, --wpp-pop-agg),
3 Fertilty rate datasets(--fbd-tfr, --witt-tfr, --wpp-tfr), and whether the final table should contain
superregions only or all regions ('y' 'n').

Example usage: python Witt_UNPD_IHME_Lancet_table_edits.py --fbd-pop population_combined --fbd-tfr tfr_combined
--wpp-pop 2019_fhs_agg_allage_bothsex_only --wpp-tfr tfr --wpp-pop-agg population --witt-pop population_ssp2
--witt-tfr tfr --supers-only n

written by Sam Farmer and Julian Chalek
"""

import pandas as pd
import numpy as np
import xlsxwriter

import xarray as xr
from db_queries import get_location_metadata
from fbd_core import YearRange, db
from fbd_core.file_interface import FBDPath, open_xr
from fbd_core.etl import aggregator
from fbd_core import argparse
from datetime import datetime


ext_year = 2095

##  Height for different cell types
CELL_HT = {
    'title': 3,
    'location': 1,
    'stage': 0,
    'data_cols': 2
}

    # dict: Python dictionary for mapping indentation levels to their
    # corresponding cause levels. Used for formatting the 'Cause' column
    # in the table.
INDENT_MAP = {
    0: '',
    1: '  ',
    2: '    ',
    3: '      '
}
# Query gbd shared tables and get locations
GBD_LOC_DF = get_location_metadata(gbd_round_id=5, location_set_id=35)

# Function used to find and programmatically add in those locations that are in the GBD database but not in
# UNPD and WITT data
def check_locs_array(df):
    return df['location_id'].isin(GBD_LOC_DF['location_id'])


# Convert the decimal point in the UI to a lancet style floating single decimal for both past and future data
def floating_style (list_nums):
    floating_style = []
    for num in list_nums:
        str_num = str(num)
        per_idx = str_num.find('.', 0, len(str_num))
        format_num = str_num[:per_idx] + '\u00b7' + str_num[per_idx+1:per_idx+3]
        floating_style.append(format_num)
    return floating_style

def compile_data(fbd_pop_version, fbd_tfr_version, wpp_pop_version,
        wpp_tfr_version, witt_pop_version, witt_tfr_version,
        wpp_pop_agg_version, supers_only):



    # Paths to relevant data
    # pop data
    fbdpoppath = FBDPath(f'/5/future/population/20190808_15_ref_85_agg_combined/{fbd_pop_version}.nc')
    wittpoppath = FBDPath(f'/wittgenstein/future/population/2018_with_under5/{witt_pop_version}.nc')
    wpppoppath = FBDPath(f'/wpp/future/population/2019_fhs_agg/{wpp_pop_version}.nc')
    wpp_pop_path_agg = FBDPath(f'/wpp/future/population/2019/{wpp_pop_agg_version}.nc')

    # tfr data
    future_tfr_path = FBDPath(f'/5/future/tfr/20190806_141418_fix_draw_bound_ccfx_to2110_combined/{fbd_tfr_version}.nc')
    wpptfrpath = FBDPath(f'/wpp/future/tfr/2019/{wpp_tfr_version}.nc')
    witttfrpath = FBDPath(f'/wittgenstein/future/tfr/2018/{witt_tfr_version}.nc')

    if supers_only.lower() == 'y':
        gbd_loc_hierarchy = GBD_LOC_DF.query('level < 2').set_index("location_id").to_xarray().parent_id
        final_gbd_locs_df = GBD_LOC_DF.query('level < 2')
    else:
        gbd_loc_hierarchy = GBD_LOC_DF.query('level < 4').set_index("location_id").to_xarray().parent_id
        final_gbd_locs_df = GBD_LOC_DF.query('level < 4')

    # Load in the external pop datasets
    wpp_pop_xr = open_xr(wpppoppath).data
    witt_pop_xr = open_xr(wittpoppath).data

    # Load in the tfr data
    wpp_tfr_xr = open_xr(wpptfrpath).data
    witt_tfr_xr = open_xr(witttfrpath).data
    fut_fertility = open_xr(future_tfr_path).data

    # Checking the locations for all data so they can be combined later on with all the same locations
    wpp_pop_df = wpp_pop_xr.to_dataframe().reset_index()
    witt_pop_df = witt_pop_xr.to_dataframe().reset_index()

    wpp_pop_df = wpp_pop_df['location_id'].drop_duplicates()
    witt_pop_df = witt_pop_df['location_id'].drop_duplicates()

    wpp_tfr_df = wpp_tfr_xr.to_dataframe().reset_index()
    witt_tfr_df = witt_tfr_xr.to_dataframe().reset_index()

    wpp_tfr_locs = wpp_tfr_df['location_id'].drop_duplicates()
    witt_tfr_locs = witt_tfr_df['location_id'].drop_duplicates()
    missing_locs_witt = witt_tfr_df[~check_locs_array(witt_tfr_df)]
    missing_locs_witt = missing_locs_witt.drop_duplicates('location_id')

    removed_locs_witt_tfr = witt_tfr_df[check_locs_array(witt_tfr_df)]
    removed_locs_witt_tfr = removed_locs_witt_tfr[~(removed_locs_witt_tfr['location_id']==354)]
    removed_locs_witt_tfr = removed_locs_witt_tfr[~(removed_locs_witt_tfr['location_id']==361)]

    witt_tfr_with_locsxr = removed_locs_witt_tfr.set_index(['location_id','year_id']).to_xarray()['rate']

    # Calculate aggs for WITT pop data
    locs = db.get_locations_by_max_level(3)
    hierarchy = locs[["location_id", "parent_id"]].set_index("location_id").to_xarray().parent_id

    agg_witt_pop = aggregator.Aggregator(witt_pop_xr)
    agg_witt_pop.aggregate_locations(loc_hierarchy=hierarchy, data=witt_pop_xr)
    witt_pop_da = agg_witt_pop.pop
    witt_pop_allage = witt_pop_da.sel(age_group_id=22, sex_id=3)

    wpp_pop_xr_aggs = open_xr(wpp_pop_path_agg).data

    # Calculate the UNPD TFR aggregates
    wpp_pop_2 = wpp_pop_xr_aggs.sel(sex_id=2)
    wpp_pop_2 = wpp_pop_2.rename(dict(age_group_name="age_group_id"))
    age_group_ids = [1] + list(range(6, 21)) + [30, 31, 32, 235]
    wpp_pop_2["age_group_id"] = age_group_ids
    wpp_pop_fert = wpp_pop_2.sel(age_group_id = range(7, 15 + 1)).sum("age_group_id")
    agg = aggregator.Aggregator(wpp_pop_fert)
    wpp_tfr_agg = agg.aggregate_locations(hierarchy, data=wpp_tfr_xr).rate

    # Calculate the WITT TFR aggregates
    witt_pop_2 = witt_pop_da.sel(sex_id=2)
    witt_pop_fert = witt_pop_2.sel(age_group_id = range(7, 15 + 1), year_id=range(1990, 2100, 5)).sum("age_group_id")
    agg = aggregator.Aggregator(witt_pop_fert)
    witt_tfr_agg = agg.aggregate_locations(hierarchy, data=witt_tfr_with_locsxr).rate.drop("sex_id").squeeze()

    # condense down to only the data we are interested in
    witt_tfr_100 = witt_tfr_agg.sel(year_id=ext_year).drop("year_id").squeeze().rename("witt_tfr").to_dataframe().reset_index()
    wpp_tfr_100 = wpp_tfr_agg.sel(year_id=ext_year).drop(['year_id', 'sex_id']).squeeze().rename('unpd_tfr').to_dataframe().reset_index()
    ihme_tfr_100 = fut_fertility.sel(year_id=2100, scenario=0, quantile="mean").rename("ihme_tfr").drop(
        ["year_id", "scenario", "quantile"]).squeeze().to_dataframe().reset_index()


    witt_pop_100 = witt_pop_allage.sel(year_id=2100).drop(
        ["year_id", "sex_id", "age_group_id"]).squeeze().rename("witt_pop").to_dataframe().reset_index()
    wpp_pop_100 = wpp_pop_xr.sel(year_id=2100).drop(
        ["year_id", "sex_id", "age_group_id"]).squeeze().rename("unpd_pop").to_dataframe().reset_index().drop('scenario', axis=1)
    ihme_pop_100 = open_xr(fbdpoppath).data.sel(age_group_id=22, sex_id=3, scenario=0, year_id=2100, quantile="mean").drop(
        ["age_group_id", "sex_id", "scenario", "year_id", "quantile"]).squeeze().rename(
        "ihme_pop").to_dataframe().reset_index()


    ihme_pop_100['ihme_pop_int'] = ihme_pop_100['ihme_pop'].astype(int)
    witt_pop_100['witt_pop_int'] = witt_pop_100['witt_pop'].astype(int)
    wpp_pop_100['unpd_pop_int'] = wpp_pop_100['unpd_pop'].astype(int)

    witt_tfr_100['witt_tfr_round'] = witt_tfr_100.witt_tfr.round(2)
    wpp_tfr_100['unpd_tfr_round'] = wpp_tfr_100.unpd_tfr.round(2)
    ihme_tfr_100['ihme_tfr_round'] = ihme_tfr_100.ihme_tfr.round(2)

    # create new column of strings that are the pop data for each dataset put into comma format eg 100,000
    witt_pop_100['witt_pop_comma'] = pd.to_numeric(witt_pop_100['witt_pop'].fillna(0), errors='coerce')
    witt_pop_100['witt_pop_comma'] = witt_pop_100['witt_pop_comma'].map('{:,.0f}'.format)

    wpp_pop_100['unpd_pop_comma'] = pd.to_numeric(wpp_pop_100['unpd_pop'].fillna(0), errors='coerce')
    wpp_pop_100['unpd_pop_comma'] = wpp_pop_100['unpd_pop_comma'].map('{:,.0f}'.format)

    ihme_pop_100['ihme_pop_comma'] = pd.to_numeric(ihme_pop_100['ihme_pop'], errors='coerce')
    ihme_pop_100['ihme_pop_comma'] = ihme_pop_100['ihme_pop_comma'].map('{:,.0f}'.format)

    final_df = ihme_pop_100.merge(wpp_pop_100, how="left").merge(witt_pop_100, how="left").merge(
        ihme_tfr_100, how="left").merge(wpp_tfr_100, how="left").merge(witt_tfr_100, how="left")

    final_df = final_df.merge(final_gbd_locs_df[["location_id", "lancet_label", "sort_order","level"]]).sort_values("sort_order").fillna("-")


    final_df = final_df[["lancet_label", "ihme_pop_comma", "unpd_pop_comma", "witt_pop_comma",
                         "ihme_tfr_round", "unpd_tfr_round", "witt_tfr_round", "level"]]
    return (final_df)

def get_format_obj(
    workbook, font_name='Times New Roman', font_size=8,
    bg_color='#FFFFFF', align=True, bold=False):
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


def write_header(worksheet, curr_row, cols, data_cols, header_format,stages, col_name_map):
    """Utility function to write the header for each page.

    Args:
        worksheet (Worksheet object): Worksheet to which the data is written.
        curr_row(int): Starting row number for the header.
        cols(list): List of characters representing the columns.
        data_cols(Series): Row containing data to be written.
        header_format(xlsxwriter Format object): Cell format options for
            headers.
        stages (list):  'tfr', 'pop' etc.
        years (YearRange): YearRange object. Ex: YearRange(2010, 2017, 2020).
    Returns:
        int: An integer specifying the row number following the header.
    """

    ### Merge range function takes the locations of the cells to merge, the data
    ### to write and the cell format. For ag. a sample input would look like:
    ###      worksheet.merge_range('A0:B1', 'Location', cell_format_obj)
    ### The above call will merge 4 cells: A0, A1, B0, B1 and fill it with the value 'Location'.

    end_row = curr_row + CELL_HT['location']
    row_range = cols[0] + str(curr_row) + ':' + cols[0] + str(end_row)
    worksheet.merge_range(row_range, 'Location', header_format)

    num_pop_cols = sum(map(lambda i: 'pop' in  i, data_cols)) - 1
    num_tfr_cols = sum(map(lambda i: 'tfr' in  i, data_cols)) - 1
    #num_b_cols = sum(map(lambda i: 'Both' in  i, data_cols)) - 1

    col_end = 0
    for i, stage in enumerate(stages):

        if stage == 'pop':
            stage_txt = 'Population'
            col_range = num_pop_cols
        else:
            stage_txt = 'Total Fertilty Rates'
            col_range = num_tfr_cols

        col_st = col_end + 1
        col_end = col_st + col_range

        curr_row_copy = curr_row
        end_row = curr_row_copy + CELL_HT['stage']

        row_range = (
            cols[col_st] + str(curr_row_copy) + ':' +
            cols[col_end] + str(end_row)
        )

        col_txt = stage_txt
        worksheet.merge_range(row_range, col_txt, header_format)

        curr_row_copy = end_row + 1
        end_row = curr_row_copy + CELL_HT['stage']

        col_st_copy = col_st

        for column in data_cols:
            if stage in column:
                row_range = cols[col_st_copy] + str(curr_row_copy)
                worksheet.write(row_range, col_name_map[column], header_format)
                col_st_copy += 1


    return end_row + 1


def write_table(final_df, outfile, stages, years, col_name_map, supers_only):
    """Writes the data to an xlsx table.
    Args:
        final_df (DataFrame): Dataframe with formatted data.
        outfile(FBDPath): Path to store the table.
        stages (list):  'tfr', 'pop' etc.
        years (YearRange): YearRange object. Ex: YearRange(2010, 2017, 2020).
    """

    workbook = xlsxwriter.Workbook(
        str(outfile), {'constant_memory': False}
    )
    worksheet = workbook.add_worksheet('Table 1')

    header_color = '#F2DCDB'
    white = '#000000'
    black = '#FFFFFF'
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

    ### Column length is basically all columns in the dataframe except column `level`
    col_len = final_df.shape[1]-1

    data_cols = final_df.drop(['level', 'lancet_label'], axis=1).columns.values

    cols = list(map(chr, range(column_start, column_start+col_len)))
    worksheet.set_column(cols[0]+':'+cols[0], loc_cell_width)
    worksheet.set_column(cols[1]+':'+cols[-1], data_cell_width)

    title = (
        'Table X.'
    )
    curr_row = 1
    end_row = curr_row + CELL_HT['title']
    row_range = cols[0] + str(curr_row) + ':' + cols[-1] + str(end_row)
    worksheet.merge_range(row_range, title, title_format)

    curr_row = end_row+1
    page_row_count = 1
    page_breaks = []

    for _, row in final_df.iterrows():
        page_row_count += 1

        ### Insert page break after 33 rows.
        if row['level'] == 0 or (page_row_count != 0 and
                                 page_row_count % 33 == 0):
            page_row_count = 0
            page_breaks.append(curr_row - 1)
            curr_row = write_header(
                worksheet, curr_row, cols, data_cols,
                header_format, stages, col_name_map
            )
        end_row = curr_row + CELL_HT['data_cols']
        col_idx = 0
        if supers_only.lower() == 'y': # Check if supers_only to change formatting for columns. If True superregion columns are
                        # formatted as white background,
            if row['level'] > 3:
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
        else:
            if row['level'] < 3:
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
            if col == 'level':
                continue

            row_range = (
                cols[col_idx] + str(curr_row) + ':' +
                cols[col_idx] + str(end_row)
            )
            if col == 'lancet_label':
                loc_name = INDENT_MAP[row['level']] + row[col]
                worksheet.merge_range(row_range, loc_name, loc_fmt_obj)
            else:
                worksheet.merge_range(row_range, row[col], data_fmt_obj)

            col_idx += 1
        curr_row = end_row+1

    worksheet.set_h_pagebreaks(page_breaks[1:])
    worksheet.fit_to_pages(1, 0)
    workbook.close()


def main(fbd_pop_version, fbd_tfr_version, wpp_pop_version,
        wpp_tfr_version, witt_pop_version, witt_tfr_version,
        wpp_pop_agg_version, supers_only):

        final_df = compile_data(fbd_pop_version, fbd_tfr_version, wpp_pop_version,
                wpp_tfr_version, witt_pop_version, witt_tfr_version,
                wpp_pop_agg_version, supers_only)

        # Generate an excel worksheet with the title outfile ('Combined_table' followed by page brek size date and current
        # time to the second)

        outfile = (f'Combined_table_pb33_lancet_label_%s.xlsx' % (datetime.now().strftime('%Y-%m-%d_%H%M%S')))
        stages = ['pop','tfr']
        year_ids = YearRange(1990, 2018, 2100)

        # Dictionary used to create worksheet column names from final_df names
        col_name_map = {
        'lancet_label':'Location',
         'ihme_pop_comma': 'IHME Reference 2100',
         'unpd_pop_comma': 'UNPD Medium Variant 2100',
         'witt_pop_comma': 'Wittgenstein Reference 2100',
         'ihme_tfr_round': 'IHME Reference 2100',
         'unpd_tfr_round': 'UNPD Medium Variant 2100',
         'witt_tfr_round':'Wittgenstein SSP2 2100'
        }

        write_table(final_df, outfile, stages, year_ids, col_name_map, supers_only)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--fbd-pop",
        type=str,
        required=True,
        help="File pathing to future FBD population data. Taken as name of data file \'.nc\' is not required"
    )
    parser.add_argument(
        "--fbd-tfr",
        type=str,
        required=True,
        help="File pathing to FBD future tfr data. Taken as name of data file \'.nc\' is not required"
    )
    parser.add_argument(
        "--wpp-pop",
        type=str,
        required=True,
        help="File pathing for WPP future pop data. Taken as name of data file \'.nc\' is not required"
    )
    parser.add_argument(
        "--wpp-tfr",
        type=str,
        required=True,
        help="File pathing for WPP future tfr data. Taken as name of data file \'.nc\' is not required"
    )
    parser.add_argument(
        "--wpp-pop-agg",
        type=str,
        help=("File pathing for WPP population data. Note at time of code writing this had to be a different file due to aggregates having to be"
        " run on WPP pop file and only available population dataset had aggregates already run on it using IHME pop file. May not be needed in future. Taken as parent_folder/data.nc")
    )
    parser.add_argument(
        "--witt-pop",
        type=str,
        required=True,
        help="File pathing for WITT future pop data. Taken as name of data file \'.nc\' is not required"
    )
    parser.add_argument(
        "--witt-tfr",
        type=str,
        required=True,
        help="File pathing for WITT future tfr data. Taken as name of data file \'.nc\' is not required"
    )

    parser.add_argument(
        "--supers-only",
        type=str,
        required=True,
        help=("\'Y\' if desired table includes only global and superregions. \'N\' if desired table includes all regions")
    )

    args = parser.parse_args()

    main(fbd_pop_version=args.fbd_pop, fbd_tfr_version=args.fbd_tfr, wpp_pop_version=args.wpp_pop,
            wpp_tfr_version=args.wpp_tfr, witt_pop_version=args.witt_pop, witt_tfr_version=args.witt_tfr,
            wpp_pop_agg_version=args.wpp_pop_agg, supers_only=args.supers_only)

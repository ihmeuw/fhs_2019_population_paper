"""
This code creates a lancet style table for comparison between UNPD, Wittgenstein, and IHME forcasts of Total
fertility rate, and population in millions by country. It can create superregions only or subnationals.

It takes in 4 population datasets (1 IHME, 1 WITT, 2 UNPD) and 3 Fertilty rate datasets(IHME, WITT, UNPD).

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
from datetime import datetime

supers_only = True #set whether the table is only looking at the 7 superregions only (True) or country level (False)

ext_year=2095
# for both WITT and UNPD datasets the data is given in a 5 year range (e.g. data for 1990-1995 would
# be 1995) this variable is used to select the correct year for both TFR datasets. Since pop data was calculated
#  by us we have remapped to our year standard and can use 2100 for selecting pop data

##  Height for different cell types
CELL_HT = {
    'title': 3,
    'location': 1,
    'stage': 0,
    'data_cols': 2
}
# dict: Python dictionary for mapping cell heights to their cell types.
# This is required while creating the table

INDENT_MAP = {
    0: '',
    1: '  ',
    2: '    ',
    3: '      '
}
# dict: Python dictionary for mapping indentation levels to their
# corresponding cause levels. Used for formatting the 'Cause' column
# in the table.

# Paths to relevant data

# pop data
fbdpoppath = ('/ihme/forecasting/data/5/future/population/20190808_15_ref_85_agg_combined/population_combined.nc')
wpppoppath = ('/ihme/forecasting/data/wpp/future/population/2019_with_under5_our_aggs/2019_fhs_agg_allage_bothsex_only.nc')
wittpoppath = ('/ihme/forecasting/data/wittgenstein/future/population/2018_with_under5/population_ssp2.nc')
wpp_pop_path_agg = ('/ihme/forecasting/data/wpp/future/population/2019/population.nc')

# tfr data
wpptfrpath = ('/ihme/forecasting/data/wpp/future/tfr/2019/tfr.nc')
witttfrpath = ('/ihme/forecasting/data/wittgenstein/future/tfr/2018/tfr.nc')
future_tfr_path = ('/ihme/forecasting/data/5/future/tfr/20190806_141418_fix_draw_bound_ccfx_to2110_combined/tfr_combined.nc')

# Query gbd shared tables and get locations
gbd_loc_df = get_location_metadata(gbd_round_id=5, location_set_id=35)

if supers_only:
    gbd_loc_hierarchy = gbd_loc_df.query('level < 2').set_index("location_id").to_xarray().parent_id
    final_gbd_locs_df = gbd_loc_df.query('level < 2')
else:
    gbd_loc_hierarchy = gbd_loc_df.query('level < 4').set_index("location_id").to_xarray().parent_id
    final_gbd_locs_df = gbd_loc_df.query('level < 4')

# Load in the external pop datasets
wpp_pop_xr = xr.open_dataset(wpppoppath)
witt_pop_xr = xr.open_dataset(wittpoppath)

# Load in the tfr data
wpp_tfr_xr = xr.open_dataset(wpptfrpath)
witt_tfr_xr = xr.open_dataset(witttfrpath)
fut_fertility = xr.open_dataset(future_tfr_path)

# Checking the locations for all data so they can be combined later on with all the same locations needed
wpp_pop_df = wpp_pop_xr.to_dataframe().reset_index()
witt_pop_df = witt_pop_xr.to_dataframe().reset_index()

wpp_pop_locs = wpp_pop_df['location_id'].drop_duplicates()
witt_pop_locs = witt_pop_df['location_id'].drop_duplicates()

wpp_tfr_df = wpp_tfr_xr.to_dataframe().reset_index()
witt_tfr_df = witt_tfr_xr.to_dataframe().reset_index()

wpp_tfr_locs = wpp_tfr_df['location_id'].drop_duplicates()
witt_tfr_locs = witt_tfr_df['location_id'].drop_duplicates()

# Function used to find and programmatically add in those locations that are in the GBD database but not in
# UNPD and WITT data
def check_locs_array(df):
    return df['location_id'].isin(gbd_loc_df['location_id'])

missing_locs = witt_tfr_df[~check_locs_array(witt_tfr_df)]
missing_locs = missing_locs.drop_duplicates('location_id')

removed_locs_witt_tfr = witt_tfr_df[check_locs_array(witt_tfr_df)]
removed_locs_witt_tfr = removed_locs_witt_tfr[~(removed_locs_witt_tfr['location_id']==354)]
removed_locs_witt_tfr = removed_locs_witt_tfr[~(removed_locs_witt_tfr['location_id']==361)]

witt_tfr_with_locsxr = removed_locs_witt_tfr.set_index(['location_id','year_id']).to_xarray()['rate']

# Calculate aggs for pop data
#agg_wpp_pop = aggregator.Aggregator(wpp_pop_xr)
locs = db.get_locations_by_max_level(3)
hierarchy = locs[["location_id", "parent_id"]].set_index("location_id").to_xarray().parent_id

agg_witt_pop = aggregator.Aggregator(witt_pop_xr)
agg_witt_pop.aggregate_locations(loc_hierarchy=hierarchy, data=witt_pop_xr)
witt_pop_da = agg_witt_pop.pop["population"]
witt_pop_allage = witt_pop_da.sel(age_group_id=22, sex_id=3)

wpp_pop_xr_aggs = xr.open_dataset(wpp_pop_path_agg)

# Calculate the UNPD TFR aggregates
wpp_pop_2 = wpp_pop_xr_aggs.sel(sex_id=2)
wpp_pop_2 = wpp_pop_2.rename(dict(age_group_name="age_group_id"))
age_group_ids = [1] + list(range(6, 21)) + [30, 31, 32, 235]
wpp_pop_2["age_group_id"] = age_group_ids
wpp_pop_fert = wpp_pop_2.sel(age_group_id = range(7, 15 + 1)).sum("age_group_id")["pop"]
agg = aggregator.Aggregator(wpp_pop_fert)
locs = db.get_locations_by_max_level(3)
hierarchy = locs[["location_id", "parent_id"]].set_index("location_id").to_xarray().parent_id
wpp_tfr = wpp_tfr_xr["value"]
wpp_tfr_agg = agg.aggregate_locations(hierarchy, data=wpp_tfr).rate

# Calculate the WITT TFR aggregates
witt_pop_2 = witt_pop_da.sel(sex_id=2)
witt_pop_fert = witt_pop_2.sel(age_group_id = range(7, 15 + 1), year_id=range(1990, 2100, 5)).sum("age_group_id")
agg = aggregator.Aggregator(witt_pop_fert)
locs = db.get_locations_by_max_level(3)
hierarchy = locs[["location_id", "parent_id"]].set_index("location_id").to_xarray().parent_id
witt_tfr_agg = agg.aggregate_locations(hierarchy, data=witt_tfr_with_locsxr).rate.drop("sex_id").squeeze()

# condense down to only the data we are interested in
witt_tfr_100 = witt_tfr_agg.sel(year_id=ext_year).drop("year_id").squeeze().rename("witt_tfr").to_dataframe().reset_index()
wpp_tfr_100 = wpp_tfr_agg.sel(year_id=ext_year).drop(['year_id', 'sex_id']).squeeze().rename('unpd_tfr').to_dataframe().reset_index()
ihme_tfr_100 = fut_fertility.sel(year_id=2100, scenario=0, quantile="mean")["value"].rename("ihme_tfr").drop(
    ["year_id", "scenario", "quantile"]).squeeze().to_dataframe().reset_index()


witt_pop_100 = witt_pop_allage.sel(year_id=2100).drop(
    ["year_id", "sex_id", "age_group_id"]).squeeze().rename("witt_pop").to_dataframe().reset_index()
wpp_pop_100 = wpp_pop_xr.sel(year_id=2100).drop(
    ["year_id", "sex_id", "age_group_id"]).squeeze().rename({'population':"unpd_pop"}).to_dataframe().reset_index().drop('scenario', axis=1)
ihme_pop_100 = xr.open_dataset(fbdpoppath).sel(age_group_id=22, sex_id=3, scenario=0, year_id=2100, quantile="mean").drop(
    ["age_group_id", "sex_id", "scenario", "year_id", "quantile"])["value"].squeeze().rename(
    "ihme_pop").to_dataframe().reset_index()



ihme_pop_100['ihme_pop_int'] = ihme_pop_100['ihme_pop'].astype(int)
witt_pop_100['witt_pop_int'] = witt_pop_100['witt_pop'].astype(int)
wpp_pop_100['unpd_pop_int'] = wpp_pop_100['unpd_pop'].astype(int)



# Convert the decimal point in the UI to a lancet style floating single decimal for both past and future data
def floating_style (list_nums = None):
    floating_style = []
    for num in list_nums:
        str_num = str(num)
        per_idx = str_num.find('.', 0, len(str_num))
        format_num = str_num[:per_idx] + '\u00b7' + str_num[per_idx+1:per_idx+3]
        floating_style.append(format_num)
    return floating_style

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
            #unit_txt = '' Left in for future in case a unit text needs to be put in after the col name
            stage_txt = 'Population'
            col_range = num_pop_cols
        else:
            #unit_txt = '' Left in for future in case a unit text needs to be put in after the col name
            stage_txt = 'Total Fertilty Rates'
            col_range = num_tfr_cols
        #elif stage == 'Both' :
            #unit_txt = '' Left in for future in case a unit text needs to be put in after the col name
            #stage_txt = 'Both'
            #col_range = num_b_cols

        col_st = col_end + 1
        col_end = col_st + col_range

        curr_row_copy = curr_row
        end_row = curr_row_copy + CELL_HT['stage']

        row_range = (
            cols[col_st] + str(curr_row_copy) + ':' +
            cols[col_end] + str(end_row)
        )

        col_txt = stage_txt #+ unit_txt Left in for future in case a unit text needs to be put in after the col name
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


def write_table(final_df, outfile, stages, years, col_name_map):
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

        ### Insert page break after 34 rows.
        if row['level'] == 0 or (page_row_count != 0 and
                                 page_row_count % 30 == 0):
            page_row_count = 0
            page_breaks.append(curr_row - 1)
            curr_row = write_header(
                worksheet, curr_row, cols, data_cols,
                header_format, stages, col_name_map
            )
        end_row = curr_row + CELL_HT['data_cols']
        col_idx = 0
        if supers_only: # Check if supers_only to change formatting for columns. If True superregion columns are
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

# Generate an excel worksheet with the title outfile ('Combined_table' followed by page brek size date and current
# time to the second)

outfile = ('Combined_table_pb30_lancet_label_%s.xlsx' % (datetime.now().strftime('%Y-%m-%d_%H%M%S')))
stages = ['pop','tfr']
year_ids = YearRange(1990, 2018, 2100)

# Dictionary used to create worksheet column names from df names
col_name_map = {
'lancet_label':'Location',
 'ihme_pop_comma': 'IHME Reference 2100',
 'unpd_pop_comma': 'UNPD Medium Variant 2100',
 'witt_pop_comma': 'Wittgenstein Reference 2100',
 'ihme_tfr_round': 'IHME Reference 2100',
 'unpd_tfr_round': 'UNPD Medium Variant 2100',
 'witt_tfr_round':'Wittgenstein SSP2 2100'
}

write_table(final_df, outfile, stages, year_ids, col_name_map)

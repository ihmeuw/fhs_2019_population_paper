'''This makes the arrow diagram showing the change in GDP ranking over the
specified years.
'''
import logging

import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
import xarray as xr

from db_queries import get_population, get_location_metadata
from fbd_core import argparse
from fbd_core import db
from fbd_core.etl import df_to_xr, resample
from fbd_core.file_interface import FBDPath, open_xr
from get_draws.api import get_draws

import fhs_2019_population_paper.plug.settings as settings

GDP = settings.BASELINE_VERSIONS["gdp"].version
GBD_ROUND_ID = settings.BASELINE_VERSIONS["population"].gbd_round_id
YEAR_LIST = [2017, 2030, 2050, 2100]

FILL = {'High-income': [179/256,226/256,205/256],
        'Southeast Asia, East Asia, and Oceania': [253/256,205/256,172/256],
        'South Asia': [203/256,213/256,232/256],
        'Latin America and Caribbean': [244/256,202/256,228/256],
        'Central Europe, Eastern Europe, and Central Asia': [230/256,245/256,201/256],
        'North Africa and Middle East': [255/256,242/256,174/256],
        'Sub-Saharan Africa': [241/256,226/256,204/256]}


def load_data(version):
    gdp_path = FBDPath(f"{GBD_ROUND_ID}/future/gdp/{version}/gdp.nc")
    gdp_da = open_xr(gdp_path).data.sel(scenario=0)
    gdp_da.name = "gdp"
    gdp_df = gdp_da.to_dataframe().reset_index()

    return gdp_df


def prep_data(gdp_df, year_list):
    location_metadata = db.get_locations_by_max_level(3)

    location_dict = location_metadata.set_index("location_id")[
        "location_name"].to_dict()

    gdp = gdp_df.filter(["location_id", "year_id", "gdp"])
    largest_by_year = pd.DataFrame()
    for year in year_list:
        gdp_year = gdp[gdp.year_id == year] \
            .sort_values(by=['gdp'], ascending=False) \
            .reset_index(drop=True)
        gdp_year["rank"] = gdp_year.index + 1
        largest_by_year = largest_by_year.append(gdp_year)

    largest_by_year.location_id = largest_by_year.location_id.map(location_dict)

    largest_df = largest_by_year.pivot_table(values="gdp",
                                             index=["location_id", "rank"],
                                             columns="year_id") \
        .reset_index()
    for year in year_list:
        largest_df[f"rank_{year}"] = largest_df.dropna(subset=[year])['rank']
    largest_df = largest_df.fillna(0)
    ranked_df = largest_df.groupby("location_id") \
        .sum() \
        .filter(like="rank_") \
        .reset_index()

    df = ranked_df.replace(0.0, np.nan) \
        .sort_values(by="rank_2017") \
        .reset_index(drop=True)

    region_dict = location_metadata.set_index("location_name")[
        "super_region_name"].to_dict()

    df['region'] = df.location_id.map(region_dict)

    ranked_data = pd.DataFrame()
    t = df.replace(0, np.nan)
    for year in year_list:
        tt = t.sort_values(by=f'rank_{year}')['location_id']
        ranked_data[f'rank_{year}'] = tt.values

    return df, ranked_data


def main(gdp_version):
    gdp = load_data(gdp_version)
    data, ranked_data = prep_data(gdp, YEAR_LIST)

    plot_dir = FBDPath(f"/{GBD_ROUND_ID}/future/gdp/{gdp_version}/",
                       root_dir='plot')
    plot_dir.mkdir(parents=True, exist_ok=True)

    location_metadata = db.get_locations_by_max_level(3)

    region_dict = location_metadata.set_index("location_name")[
        "super_region_name"].to_dict()

    title = 'Top 25 Nations by Total GDP'
    plot_file = plot_dir / "table_1_2017_arrow_diagram.pdf"

    c = canvas.Canvas(str(plot_file), pagesize=(792.0, 612.0))

    # text size and style
    titletextsize = 12
    headertextsize = 10
    textsize = 8
    textgap = textsize * 2.0

    # # write title
    titley = 625

    row1 = titley - (2.0 * textgap)
    row1_and_ahalf = titley - (3.0 * textgap)
    row2 = row1 - (2.0 * textgap)

    c.setFont("Helvetica-Bold", titletextsize)
    c.drawString(
        315,
        titley,
        "{title}".format(title=title))

    # write column headers
    c.setFont("Helvetica-Bold", textsize)
    year2017_columnwidth = 100
    gap = 500
    year2030_columnwidth = 100
    year2050_columnwidth = 100
    year2100_columnwidth = 100

    # set columns widths (counting from left to right)
    year2017_column = 70 + 3

    year2030_column = (
            year2017_column + year2017_columnwidth + 80
    )
    year2050_column = (
            year2030_column + year2030_columnwidth + 80
    )
    year2100_column = (
            year2050_column + year2050_columnwidth + 80
    )

    c.setFont("Helvetica-Bold", headertextsize)

    # name columns
    textobject_year2017 = c.beginText(year2017_column + 40, row1_and_ahalf)
    for line in ["", f"{2017}"]:
        textobject_year2017.textLine(line)
    c.drawText(textobject_year2017)

    textobject_year2030 = c.beginText(
        year2030_column + 40, row1_and_ahalf)
    for line in ["", f"{2030}"]:
        textobject_year2030.textLine(line)
    c.drawText(textobject_year2030)

    textobject_year2050 = c.beginText(
        year2050_column + 40, row1_and_ahalf)
    for line in ["", f"{2050}"]:
        textobject_year2050.textLine(line)
    c.drawText(textobject_year2050)

    textobject_year2100 = c.beginText(
        year2100_column + 40, row1_and_ahalf)
    for line in ["", f"{2100}"]:
        textobject_year2100.textLine(line)
    c.drawText(textobject_year2100)

    # unknown territory

    total_iter = 1

    # country position (after top 25)
    countryposition_2017 = 26
    countryposition_2030 = 26
    countryposition_2050 = 26
    countryposition_2100 = 26

    lineposition_2017 = 26
    lineposition_2030 = 26
    lineposition_2050 = 26
    lineposition_2100 = 26

    for index in ranked_data['rank_2017'].unique():
        row_data = ranked_data.query('rank_2017 == @index').reset_index().iloc[
            0]

        rank2017 = row_data['index'] + 1
        label2017 = row_data['rank_2017']

        rank2030 = row_data['index'] + 1
        label2030 = row_data['rank_2030']

        rank2050 = row_data['index'] + 1
        label2050 = row_data['rank_2050']

        rank2100 = row_data['index'] + 1
        label2100 = row_data['rank_2100']

        c.setFont("Helvetica", textsize)

        # determine rank change
        this_rank = label2017

        line_start_2017 = data.query(
            "location_id == @this_rank")['rank_2017'].values[0]
        line_end_2030 = data.query(
            "location_id == @this_rank")['rank_2030'].values[0]
        line_end_2050 = data.query(
            "location_id == @this_rank")['rank_2050'].values[0]
        line_end_2100 = data.query(
            "location_id == @this_rank")['rank_2100'].values[0]

        # draw rectangles
        if total_iter < 26:
            # line style
            c.setDash(1, 0)
            # stroke colour
            c.setStrokeColorRGB(0, 0, 0)

            # 2017
            region = region_dict[label2017]
            c.setFillColorRGB(
                FILL[region][0],
                FILL[region][1],
                FILL[region][2])

            c.rect(year2017_column,
                   row2 - (total_iter * textgap) - 2.5,
                   year2017_columnwidth,
                   textsize * 2.0,
                   stroke=1,
                   fill=1)
            # 2030
            region = region_dict[label2030]
            c.setFillColorRGB(
                FILL[region][0],
                FILL[region][1],
                FILL[region][2])

            c.rect(year2030_column,
                   row2 - (total_iter * textgap) - 2.5,
                   year2030_columnwidth,
                   textsize * 2.0,
                   stroke=1,
                   fill=1)
            # 2050
            region = region_dict[label2050]
            c.setFillColorRGB(
                FILL[region][0],
                FILL[region][1],
                FILL[region][2])

            c.rect(year2050_column,
                   row2 - (total_iter * textgap) - 2.5,
                   year2050_columnwidth,
                   textsize * 2.0,
                   stroke=1,
                   fill=1)
            # 2100
            region = region_dict[label2100]
            c.setFillColorRGB(
                FILL[region][0],
                FILL[region][1],
                FILL[region][2])

            c.rect(year2100_column,
                   row2 - (total_iter * textgap) - 2.5,
                   year2100_columnwidth,
                   textsize * 2.0,
                   stroke=1,
                   fill=1)

        # draw country names

        c.setStrokeColorRGB(0, 0, 0)
        c.setFillColorRGB(0, 0, 0)
        c.setStrokeAlpha(1)
        c.setFillAlpha(1)

        if (line_start_2017 > 25 and line_end_2030 < 26):
            c.drawString(
                year2017_column + 20,
                row2 - (countryposition_2017 * textgap) + 2.5,
                f"{int(rank2017)} {label2017}")
            countryposition_2017 += 1

        if ((line_start_2017 < 26) and (line_end_2030 > 25)):
            c.drawString(
                year2030_column + 20,
                row2 - (countryposition_2030 * textgap) + 2.5,
                f"{int(line_end_2030)} {label2017}")
            countryposition_2030 += 1

        if ((line_end_2030 > 25) and (line_end_2050 < 26)):
            label = data.query(
                "rank_2050 == @line_end_2050")['location_id'].values[0]
            c.drawString(
                year2030_column + 20,
                row2 - (countryposition_2030 * textgap) + 2.5,
                f"{int(line_end_2030)} {label}")
            countryposition_2030 += 1

        if ((line_end_2030 < 26) and (line_end_2050 > 25)):
            label = data.query(
                "rank_2030 == @line_end_2030")['location_id'].values[0]
            c.drawString(
                year2050_column + 20,
                row2 - (countryposition_2050 * textgap) + 2.5,
                f"{int(line_end_2050)} {label}")
            countryposition_2050 += 1

        if ((line_end_2050 > 25) and (line_end_2100 < 26)):
            label = data.query(
                "rank_2100 == @line_end_2100")['location_id'].values[0]
            c.drawString(
                year2050_column + 20,
                row2 - (countryposition_2050 * textgap) + 2.5,
                f"{int(line_end_2050)} {label}")
            countryposition_2050 += 1

        if ((line_end_2050 < 26) and (line_end_2100 > 25)):
            label = data.query(
                "rank_2050 == @line_end_2050")['location_id'].values[0]
            c.drawString(
                year2100_column + 20,
                row2 - (countryposition_2100 * textgap) + 2.5,
                f"{int(line_end_2100)} {label}")
            countryposition_2100 += 1

        if total_iter < 26:
            c.drawString(
                year2017_column + 20,
                row2 - (total_iter * textgap) + 2.5,
                f"{rank2017} {label2017}")
            c.drawString(
                year2030_column + 20,
                row2 - (total_iter * textgap) + 2.5,
                f"{rank2030} {label2030}")
            c.drawString(
                year2050_column + 20,
                row2 - (total_iter * textgap) + 2.5,
                f"{rank2050} {label2050}")
            c.drawString(
                year2100_column + 20,
                row2 - (total_iter * textgap) + 2.5,
                f"{rank2100} {label2100}")

        # determine line type and draw

        c.setStrokeColorRGB(0, 0, 0)

        if line_start_2017 > line_end_2030:
            c.setDash(1, 0)
        else:
            c.setDash(3, 1)
        if (line_start_2017 > 25) and (line_end_2030 < 26):
            c.line(year2017_column + year2017_columnwidth,
                   row2 - (lineposition_2017 * textgap) + (0.33 * textsize),
                   year2030_column,
                   row2 - (line_end_2030 * textgap) + (0.33 * textsize))
            lineposition_2017 += 1
        elif (line_start_2017 < 26) and (line_end_2030 > 25):
            c.line(year2017_column + year2017_columnwidth,
                   row2 - (line_start_2017 * textgap) + (0.33 * textsize),
                   year2030_column,
                   row2 - (lineposition_2030 * textgap) + (0.33 * textsize))
            lineposition_2030 += 1
        elif (line_start_2017 < 26) or (line_end_2030 < 26):
            c.line(year2017_column + year2017_columnwidth,
                   row2 - (line_start_2017 * textgap) + (0.33 * textsize),
                   year2030_column,
                   row2 - (line_end_2030 * textgap) + (0.33 * textsize))

        # col2-3

        if line_end_2030 > line_end_2050:
            c.setDash(1, 0)
        else:
            c.setDash(3, 1)
        if (line_end_2030 > 25) and (line_end_2050 < 26):
            c.line(year2030_column + year2030_columnwidth,
                   row2 - (lineposition_2030 * textgap) + (0.33 * textsize),
                   year2050_column,
                   row2 - (line_end_2050 * textgap) + (0.33 * textsize))
            lineposition_2030 += 1
        elif ((line_end_2030 < 26) and (line_end_2050 > 25)):
            c.line(year2030_column + year2030_columnwidth,
                   row2 - (line_end_2030 * textgap) + (0.33 * textsize),
                   year2050_column,
                   row2 - (lineposition_2050 * textgap) + (0.33 * textsize))
            lineposition_2050 += 1
        elif (line_end_2030 < 26) or (line_end_2050 < 26):
            c.line(year2030_column + year2030_columnwidth,
                   row2 - (line_end_2030 * textgap) + (0.33 * textsize),
                   year2050_column,
                   row2 - (line_end_2050 * textgap) + (0.33 * textsize))

        # col3-4

        if line_end_2050 > line_end_2100:
            c.setDash(1, 0)
        else:
            c.setDash(3, 1)
        if ((line_end_2050 > 25) and (line_end_2100 < 26)):
            c.line(year2050_column + year2050_columnwidth,
                   row2 - (lineposition_2050 * textgap) + (0.33 * textsize),
                   year2100_column,
                   row2 - (line_end_2100 * textgap) + (0.33 * textsize))
            lineposition_2050 += 1
        elif ((line_end_2050 < 26) and (line_end_2100 > 25)):
            c.line(year2050_column + year2050_columnwidth,
                   row2 - (line_end_2050 * textgap) + (0.33 * textsize),
                   year2100_column,
                   row2 - (lineposition_2100 * textgap) + (0.33 * textsize))
            lineposition_2100 += 1
        elif (line_end_2050 < 26) or (line_end_2100 < 26):
            c.line(year2050_column + year2050_columnwidth,
                   row2 - (line_end_2050 * textgap) + (0.33 * textsize),
                   year2100_column,
                   row2 - (line_end_2100 * textgap) + (0.33 * textsize))

        # iterate
        total_iter = total_iter + 1

    # 2017
    rect_loc = 31

    region = "High-income"
    c.setFillColorRGB(
        FILL[region][0],
        FILL[region][1],
        FILL[region][2])

    c.rect(year2017_column,
           row2 - (rect_loc * textgap) - 2.5,
           year2017_columnwidth + 20,
           textsize * 2.0,
           stroke=1,
           fill=1)

    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeAlpha(1)
    c.setFillAlpha(1)

    c.drawString(
        year2017_column + 10,
        row2 - (rect_loc * textgap) + 2.5,
        "High-income")

    region = "Southeast Asia, East Asia, and Oceania"
    c.setFillColorRGB(
        FILL[region][0],
        FILL[region][1],
        FILL[region][2])

    c.rect(year2030_column - 60,
           row2 - (rect_loc * textgap) - 2.5,
           year2017_columnwidth + 90,
           textsize * 2.0,
           stroke=1,
           fill=1)

    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeAlpha(1)
    c.setFillAlpha(1)

    c.drawString(
        year2030_column - 50,
        row2 - (rect_loc * textgap) + 2.5,
        f"{region}")

    region = "South Asia"
    c.setFillColorRGB(
        FILL[region][0],
        FILL[region][1],
        FILL[region][2])

    c.rect(year2017_column,
           row2 - ((rect_loc + 1) * textgap) - 2.5,
           year2017_columnwidth + 20,
           textsize * 2.0,
           stroke=1,
           fill=1)

    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeAlpha(1)
    c.setFillAlpha(1)

    c.drawString(
        year2017_column + 10,
        row2 - ((rect_loc + 1) * textgap) + 2.5,
        "South Asia")

    region = "Latin America and Caribbean"
    c.setFillColorRGB(
        FILL[region][0],
        FILL[region][1],
        FILL[region][2])

    c.rect(year2030_column - 60,
           row2 - ((rect_loc + 1) * textgap) - 2.5,
           year2017_columnwidth + 90,
           textsize * 2.0,
           stroke=1,
           fill=1)

    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeAlpha(1)
    c.setFillAlpha(1)

    c.drawString(
        year2030_column - 50,
        row2 - ((rect_loc + 1) * textgap) + 2.5,
        "Latin America and Caribbean")

    region = "Central Europe, Eastern Europe, and Central Asia"
    c.setFillColorRGB(
        FILL[region][0],
        FILL[region][1],
        FILL[region][2])

    c.rect(year2030_column - 60,
           row2 - ((rect_loc + 2) * textgap) - 2.5,
           year2017_columnwidth + 90,
           textsize * 2.0,
           stroke=1,
           fill=1)

    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeAlpha(1)
    c.setFillAlpha(1)

    c.drawString(
        year2030_column - 50,
        row2 - ((rect_loc + 2) * textgap) + 2.5,
        "Central Europe, Eastern Europe, and Central Asia")

    region = "North Africa and Middle East"
    c.setFillColorRGB(
        FILL[region][0],
        FILL[region][1],
        FILL[region][2])

    c.rect(year2017_column,
           row2 - ((rect_loc + 2) * textgap) - 2.5,
           year2017_columnwidth + 20,
           textsize * 2.0,
           stroke=1,
           fill=1)

    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeAlpha(1)
    c.setFillAlpha(1)

    c.drawString(
        year2017_column + 10,
        row2 - ((rect_loc + 2) * textgap) + 2.5,
        "North Africa and Middle East")

    region = "Sub-Saharan Africa"
    c.setFillColorRGB(
        FILL[region][0],
        FILL[region][1],
        FILL[region][2])

    c.rect(year2017_column,
           row2 - ((rect_loc + 3) * textgap) - 2.5,
           year2017_columnwidth + 20,
           textsize * 2.0,
           stroke=1,
           fill=1)

    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeAlpha(1)
    c.setFillAlpha(1)

    c.drawString(
        year2017_column + 10,
        row2 - ((rect_loc + 3) * textgap) + 2.5,
        "Sub-Saharan Africa")

    c.save()


if __name__ == "__main__":
    main(GDP)
import argparse
import os
import string
import pandas as pd
import xarray as xr
from fbd_core import db

from fbd_core.file_interface import FBDPath, stage_versions


def save_netcdf(data, filepath):
    """
    Saves an input xarray object to a specified filepath, making parent
    directories if necessary

    :param xarray.DataArray or xarray.Dataset: data to save
    :param FBDPath filepath: location (should be a netcdf) to save the data
    """
    filepath.PFN().parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(str(filepath))


def write_cod_forecast(data, acause, model, gbd_round_id, addendum=''):
    """
    Save a cod model and assert that it has the appropriate dimensions.

    :param data: dataarray
        Dataarray with cod mortality rate.
    :param acause: str
        valid acause for cod outcome.
    :param model: str, default None
        model to load, defaults to best
    :param gbd_round_id: int
        gbd round that the forecasts were run from (e.g., 3 for 2015)
    :param : str
        valid acause for cod outcome.
    """
    keys = ['year_id', 'age_group_id', 'location_id', 'sex_id', 'draw']
    missing = [k for k in keys if k not in list(data.coords.keys())]
    assert len(missing) == 0, "Data is missing dimensions: {}".format(missing)
    save_netcdf(data, FILEPATH)


def write_cod_betas(model_params, acause, model, gbd_round_id, addendum=''):
    """
    Write the betas from a gk cod model run.

    :param model_params: Dataset
        Xarray Dataset with covariate and draw information.
    :param acause: str
        The string of the acause to place saved results in correct location.
    :param model: str
        The string of the model to place saved results in correct location.
    :param gbd_round_id: int
        gbd round the forecasts were run based on (e.g., 3 for 2015)
    :param addendum: str
        Addition information to include in save file name.
    """
    model_params *= xr.DataArray([1], dims=["acause"], coords=[[acause]])
    save_netcdf(model_params, FILEPATH)


def check_output(model, gbd_round_id):
    """
    Check the output of a model run and return any missing models.

    :param model: str
         The model to check results for
    :param gbd_round_id: int
         gbd round that the outputs are based on (e.g., 3 for 2015)
    :return: data.frame
         Data frame with failed model results.
    """
    cause_df = db.get_cod_model_causes(long_on_sex=True)
    fail_df = pd.DataFrame(dict(cause_id=[], acause=[], sex_id=[]))
    for i, x in cause_df.iterrows():
        sexn = "_male" if x["sex_id"] == 1 else "_female"
        f_ext = "{c}{s}.nc".format(c=x["acause"], s=sexn)
        file_path = FILEPATH / f_ext
        if not file_path.isfile():
            fail_df = fail_df.append(x)
    fail_df.cause_id = fail_df.cause_id.astype(int)
    fail_df.sex_id = fail_df.sex_id.astype(int)
    return fail_df


def beta_excel_map(model, par, cov, gbd_round_id):
    """
    Create an excel sheet of a gk cod models beta.

    :param model: str
        COD model to pull results from.
    :param par: str
        The paramerter level to pull such as "beta_global", "beta_age",
        "gamma_region_age"
    :param cov: str
        The covariate to extract.
    :param gbd_round_id: int
        GBD round the models were based off of (e.g., 3 for 2015)
    """
    beta_df_list = [xr.open_dataset(str(x))
                    for x in FILEPATH.iterdir()
                    if str(x).endswith(".nc")]
    beta = list()
    for s in beta_df_list:
        try:
            beta += [s[par].loc[dict(cov=cov)].drop("cov")]
        except KeyError:
            pass
    dim1concat = [[], []]

    for i, s in enumerate([1, 2]):
        for b in beta:
            if b.sex_id[0] == s:
                dim1concat[i].append(b)

    betaX = xr.concat([xr.concat(x, dim="acause") for x in dim1concat],
                      dim="sex_id").mean("draw")
    if "age_group_id" not in betaX.dims:
        betaX = betaX * xr.DataArray([1], dims=["age_group_id"], coords=[[22]])
    betaX = betaX.transpose(*["acause", "age_group_id", "sex_id"])
    betaXsubm = betaX.loc[{"sex_id": 1}].drop("sex_id")
    betaXsubf = betaX.loc[{"sex_id": 2}].drop("sex_id")
    betamale = pd.DataFrame(betaXsubm.values)
    betamale.index = betaX.acause.values
    betamale.columns = betaX.age_group_id.values
    betafemale = pd.DataFrame(betaXsubf.values)
    betafemale.index = betaX.acause.values
    betafemale.columns = betaX.age_group_id.values
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    excel_file = FILEPATH / 'betavals_{p}_{c}.xlsx'.format(p=par, c=cov)
    sheet_name1 = 'Female'
    sheet_name2 = 'Male'

    writer = pd.ExcelWriter(str(excel_file), engine='xlsxwriter')
    betafemale.to_excel(writer, sheet_name=sheet_name1)
    betamale.to_excel(writer, sheet_name=sheet_name2)

    worksheet1 = writer.sheets[sheet_name1]
    worksheet2 = writer.sheets[sheet_name2]

    # Apply a conditional format to the cell range.
    endx = betafemale.shape[0] + 1
    endy = betafemale.shape[1]
    letterstart = 'B'
    letterend = string.uppercase[endy]
    cond = '{ls}2:{le}{end}'.format(ls=letterstart, le=letterend, end=endx)
    worksheet1.conditional_format(cond, {'type': '3_color_scale'})
    worksheet2.conditional_format(cond, {'type': '3_color_scale'})

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def list_cod_models(gbd_round_id):
    """
    List all the cod models in the cod model folder.

    :param gbd_round_id: int
        GBD round id to get cod models from (e.g., 3 for 2015)
    :return: cod model list
    """
    return sorted(stage_versions(FILEPATH))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Group betas in to escel file.")
    parser.add_argument("-p", "--par", type=str, required=True,
                        help=("Paramter level to extract such as "
                              "'beta_age', 'beta_region_age', 'beta_global'"))
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Model Version to pull results from.")
    parser.add_argument("-c", "--cov", type=str, required=True,
                        help="Covariate to extract and write to excel sheet.")
    parser.add_argument("--gbd-round-id", type=int, required=True,
                        help="GBD round that the model was based on (e.g., 3 \
                        for 2015)")

    args = parser.parse_args()
    beta_excel_map(args.model, args.par, args.cov, args.gbd_round_id)

import xarray as xr
from fbd_core.file_interface import FBDPath

def get_rid_of_draws_scenarios_and_sex_id(file, drop_draw=True, drop_scenario=False, select_age_group_ids = None, select_years = None, select_lid = None):
    """If you want to get rid of draws, then the mean is taken across draws.
    If you want to get rid of scenarios, then the reference scenario is sliced into.
    Sex is always removed by slicing into female data. Reads in a .nc file.
    Returns a pandas dataframe.
    """
    data = xr.open_dataarray(file)
    data
    if select_age_group_ids is None:
        select_age_group_ids = range(8,15)
    if select_years is None:
        select_years = data.year_id.values
    if select_lid is None:
        select_lid = data.location_id.values
    data = data.loc[dict(age_group_id = select_age_group_ids, year_id = select_years, location_id = select_lid)]
    if drop_scenario and "scenario" in data.dims:
        data = data.loc[dict(scenario=0)]
    if "sex_id" in data.dims:
        data = data.loc[dict(sex_id=2)]
    if drop_draw and "draw" in data.dims:
        data = data.mean("draw")
    data=data.to_dataframe().reset_index()

    return data



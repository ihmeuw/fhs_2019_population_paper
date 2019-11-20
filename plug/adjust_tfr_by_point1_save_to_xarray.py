import xarray as xr
from fbd_core.etl import df_to_xr
from fbd_core.file_interface import FBDPath, save_xr

def output_to_xarray(gbd_round, out, version_out):
    asfr_path = FBDPath("/{gri}/future/asfr/{version}".format(
        gri=gbd_round, version=version_out))
    dims = ['location_id', 'year_id', 'scenario', 'age_group_id', 'sex_id', 'draw']
    out_xr = df_to_xr(out, dims = dims)
    save_xr(out_xr,
        fbdpath = asfr_path / "asfr.nc",
        metric="rate",
        space="identity",
        version="version",
        model="asfr_adjusted_to_tfr_plus_point1_if_below2")

import numpy as np
import xarray as xr
from pandas import to_numeric

from sgs_tools.io.um import restrict_ds


def data_ingest_MONC_on_single_grid(
    fname,
    requested_fields: list[str] = ["u", "v", "w", "theta"],
):
    """read and pre-process MONC data

    :param fname_pattern: MONC NetCDF diagnostic file to read. can be a glob pattern. (should belong to the same simulation)
    :param  requested_fields: list of fields to read and pre-process using sgs_tools naming convention. Defaults to ['u', 'v', 'w', 'theta']
    """
    ds = xr.open_mfdataset(fname, chunks={}, parallel=True)

    # parse metadata
    metadata = ds["options_database"].load().data
    metadata = dict(np.char.decode(metadata))
    for k, v in metadata.items():
        if v in ["true", "false"]:
            v = v == "true"
        metadata[k] = to_numeric(v, errors="ignore")
    metadata = dict(sorted(metadata.items()))
    del ds["options_database"]

    ds = ds.squeeze()

    base_field_dict = {"th": "theta", "p": "P"}
    coord_dict = {"zn": "z_theta"}

    ds = ds.squeeze()
    # change variable names
    ds = ds.rename(base_field_dict)

    # standardize coordinate names
    ds = ds.rename(coord_dict)
    ds["x"] = ds["x"] * metadata["dxx"]
    ds["y"] = ds["y"] * metadata["dyy"]

    # interpolate theta to vel grid
    ds["theta_interp"] = (
        ds["theta"]
        .rename({"z_theta": "z"})
        .interp(z=ds["w"].z, method="linear", assume_sorted=True)
    )
    del ds["theta"]
    ds = ds.rename({"theta_interp": "theta"})
    ds = restrict_ds(ds, requested_fields)
    for coord in ds.coords:
        ds[coord].attrs.update({"units": "m"})
    return metadata, ds

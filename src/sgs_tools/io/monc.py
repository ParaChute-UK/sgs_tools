import numpy as np
import xarray as xr
from pandas import to_numeric


def data_ingest_MONC_on_single_grid(fname):
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

    for coord in ds.coords:
        ds[coord].attrs.update({"units": "m"})
    return metadata, ds

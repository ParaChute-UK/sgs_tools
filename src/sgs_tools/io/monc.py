from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from pandas import to_numeric

from sgs_tools.io.um import restrict_ds

base_field_dict = {"th": "theta", "p": "P"}

coord_dict = {"zn": "z_theta"}


def data_ingest_MONC_on_single_grid(
    fname_pattern,
    requested_fields: list[str] = ["u", "v", "w", "theta"],
    chunks: Any = "auto",
):
    """read and pre-process MONC data

    :param fname_pattern: MONC NetCDF diagnostic file to read. can be a glob pattern. (should belong to the same simulation)
    :param  requested_fields: list of fields to read and pre-process using sgs_tools naming convention.
    """
    fname = list(
        Path(fname_pattern.root).glob(
            str(Path(*fname_pattern.parts[fname_pattern.is_absolute() :]))
        )
    )

    ds = xr.open_mfdataset(fname, chunks=chunks, parallel=True)

    # parse metadata
    metadata = ds["options_database"].load().data
    metadata = dict(np.char.decode(metadata))
    for k, v in metadata.items():
        if v in ["true", "false"]:
            metadata[k] = v == "true"
        else:
            metadata[k] = to_numeric(v, errors="ignore")  # type: ignore
    metadata = dict(sorted(metadata.items()))
    del ds["options_database"]

    ds = ds.squeeze()
    # rename to sgs_tools naming convention
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
    ds, _ = restrict_ds(ds, requested_fields)
    for coord in ds.coords:
        ds[coord].attrs.update({"units": "m"})
    return metadata, ds

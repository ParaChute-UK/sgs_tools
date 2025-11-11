from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from pandas import to_numeric

from sgs_tools.geometry.staggered_grid import interpolate_to_grid
from sgs_tools.io.read_util import restrict_ds, standardize_varnames

base_field_dict = {"th": "theta", "p": "P"}

coord_dict = {"zn": "z_theta"}


def data_ingest_MONC(
    fname_pattern,
    requested_fields: list[str] = ["u", "v", "w", "theta"],
    chunks: Any = "auto",
):
    """read and pre-process MONC data using sgs_tools naming convention.
    Any unknown fields will retain their original names.

    :param fname_pattern: MONC NetCDF diagnostic file to read. can be a glob pattern. (should belong to the same simulation)
    :param  requested_fields: list of fields to retain in ds, if falsy will retain all.
    :param chunks: chunking of datasets "auto" or a dictionary of {coordinate: chunks}.
    :return: metadata dictionary, xarray Dataset of fields.
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
    ds = standardize_varnames(ds, base_field_dict)

    # standardize coordinate names
    ds = ds.rename(coord_dict)
    ds["x"] = ds["x"] * metadata["dxx"]
    ds["y"] = ds["y"] * metadata["dyy"]

    for coord in ds.coords:
        ds[coord].attrs.update({"units": "m"})
    if requested_fields:
        ds, _ = restrict_ds(ds, requested_fields)
    assert len(ds) > 0, "None of the requested fields are available"
    return metadata, ds


def data_ingest_MONC_on_single_grid(
    fname_pattern: Path,
    requested_fields: list[str] = ["u", "v", "w", "theta"],
    chunks: Any = "auto",
) -> xr.Dataset:
    """read pre-process MONC data and interpolate to a cell-centred grid
    Any unknown fields will retain their original names.

    :param fname_pattern: MONC NetCDF diagnostic file(s) to read. will be interpreted as a glob pattern. (should belong to the same simulation)
    :param  requested_fields: list of fields to retain in ds, if falsy will retain all.
    :param chunks: chunking of datasets "auto" or a dictionary of {coordinate: chunks}.
    """
    # read, constrain fields, unify grids
    metadata, simulation = data_ingest_MONC(fname_pattern, requested_fields, chunks)

    # interpolate all vars to a cell-centred grid
    centre_dims = ["x", "y", "z"]
    simulation = interpolate_to_grid(simulation, centre_dims, drop_coords=True)
    return metadata, simulation

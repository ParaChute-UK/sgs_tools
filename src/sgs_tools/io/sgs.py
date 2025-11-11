from pathlib import Path
from typing import Any, Dict

import xarray as xr

from sgs_tools.io.read_util import (
    parse_fname_pattern,
    restrict_ds,
    standardize_varnames,
)

degenerate_naming_convention: Dict[str, str] = {}


def data_ingest_SGS(
    fname_pattern: Path | str,
    requested_fields: list[str] = ["u", "v", "w", "theta"],
    chunks: Any = "auto",
):
    """read and pre-process local-convention (sgs_tools) NetCDF data using sgs_tools naming convention.
    Will not rename any fields, assume they are in local convention.

    :param fname_pattern: NetCDF diagnostic file to read. can be a glob pattern. (should belong to the same simulation)
    :param requested_fields: list of fields to retain in ds, if falsy will retain all.
    :param chunks: chunking for data
    """
    # parse filename (glob, ~, etc.)
    fname = parse_fname_pattern(fname_pattern)
    # open file(s)
    ds = xr.open_mfdataset(fname, chunks=chunks, parallel=True, engine="h5netcdf")
    # rename to sgs_tools naming convention
    ds = standardize_varnames(ds, degenerate_naming_convention)

    if requested_fields:
        ds, _ = restrict_ds(ds, requested_fields)
    return ds

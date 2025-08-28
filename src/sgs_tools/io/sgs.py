from pathlib import Path
from typing import Any

import xarray as xr

from sgs_tools.io.um import restrict_ds


def data_ingest_SGS(
    fname_pattern,
    requested_fields: list[str] = ["u", "v", "w", "theta"],
    chunks: Any = "auto",
):
    """read and pre-process local-convention (sgs_tools) NetCDF data

    :param fname_pattern: NetCDF diagnostic file to read. can be a glob pattern. (should belong to the same simulation)
    :param  requested_fields: list of fields to read and pre-process using sgs_tools naming convention.
    :param chunks: chunking for data
    """
    fname = list(
        Path(fname_pattern.root).glob(
            str(Path(*fname_pattern.parts[fname_pattern.is_absolute() :]))
        )
    )

    ds = xr.open_mfdataset(fname, chunks=chunks, parallel=True, engine="h5netcdf")
    ds = restrict_ds(ds, requested_fields)

    return ds

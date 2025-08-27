from pathlib import Path

import numpy as np
import xarray as xr

from sgs_tools.io.monc import data_ingest_MONC_on_single_grid
from sgs_tools.io.sgs import data_ingest_SGS
from sgs_tools.io.um import data_ingest_UM_on_single_grid


def read(
    input_files: Path, input_format: str, requested_fields: list[str], **kwargs
) -> xr.Dataset:
    if input_format == "sgs":
        simulation = data_ingest_SGS(
            input_files,
            requested_fields=requested_fields,
        )
        assert "h_resolution" in simulation.attrs, "missing attribute 'h_resolution'."
    elif input_format == "um":
        simulation = data_ingest_UM_on_single_grid(
            input_files, requested_fields=requested_fields, res=kwargs["resolution"]
        )
        simulation.attrs["h_resolution"] = kwargs["resolution"]
    elif input_format == "monc":
        meta, simulation = data_ingest_MONC_on_single_grid(
            input_files,
            requested_fields=requested_fields,
        )
        # overwrite resolution
        assert np.isclose(meta["dxx"], meta["dyy"])
        simulation.attrs["h_resolution"] = meta["dxx"]
    else:
        raise ValueError(f"Unsupported input format {input_format}")
    return simulation

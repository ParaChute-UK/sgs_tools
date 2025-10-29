from pathlib import Path

import numpy as np
import xarray as xr

from sgs_tools.io.monc import data_ingest_MONC_on_single_grid
from sgs_tools.io.sgs import data_ingest_SGS
from sgs_tools.io.um import data_ingest_UM_on_single_grid


def read(
    input_files: Path, input_format: str, requested_fields: list[str], **kwargs
) -> xr.Dataset:
    """
    Read simulation data from input files and return an xarray Dataset.

    :param input_files: Path to the input file(s) containing simulation data.
    :param input_format: Format of the input data. Supported formats are:
        ``sgs``, ``um``, ``monc``

    :param requested_fields: List of variable names to extract from the input data.
    :param kwargs: Additional keyword arguments depending on the input format.
      The ``um`` format, requires ``resolution`` (float) specifying horizontal grid spacing.

    :return: xarray Dataset containing the requested fields and metadata, including
      the horizontal resolution stored in ``attrs["h_resolution"]``.

    .. note::
        - For ``monc`` format, resolution is inferred from metadata and assumed isotropic in x and y
        - For ``um`` format, resolution must be explicitly provided via `kwargs`.
        - For ``sgs`` format, if h_resolution is not a dataset attribute it is guessed by the spacing in "x" and "y" coordinates
    """

    if input_format == "sgs":
        simulation = data_ingest_SGS(
            input_files,
            requested_fields=requested_fields,
        )
        if "h_resolution" not in simulation.attrs:
            dx = simulation.coords["x"].diff(dim="x")
            dy = simulation.coords["y"].diff(dim="y")
            assert dx.std().item() < 1e-10
            assert dy.std().item() < 1e-10

            assert np.isclose(dx[0], dy[0])
            simulation.attrs["h_resolution"] = dx[0].item()

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

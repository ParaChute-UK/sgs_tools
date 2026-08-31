from pathlib import Path

import numpy as np
import xarray as xr

from sgs_tools.io.monc import data_ingest_MONC, data_ingest_MONC_on_single_grid
from sgs_tools.io.sgs import data_ingest_SGS
from sgs_tools.io.um import data_ingest_UM, data_ingest_UM_on_single_grid


def read(
    input_files: Path,
    input_format: str,
    requested_fields: list[str],
    interp_grid: bool = True,
    **kwargs,
) -> xr.Dataset:
    """
    Read simulation data from input files and return an xarray Dataset.

    :param input_files: Path to the input file(s) containing simulation data.
    :param input_format: Format of the input data. Supported formats are:
        ``sgs``, ``um_ideal``, ``um_real``, ``monc``

    :param requested_fields: List of variable names to extract from the input data.
    :param kwargs: Additional keyword arguments depending on the input format.
      The ``um_ideal`` format requires ``resolution`` (float)
      specifying horizontal grid spacing. It also accepts
      ``field_names_dict`` dict[str,str] prescribing a field-names lookup table.
      The ``um_real`` format accepts ``field_names_dict`` only;
      ``resolution`` is not required because coordinate values are read from the file.


    :return: xarray Dataset containing the requested fields and metadata, including
      the horizontal resolution stored in ``attrs["h_resolution"]``.

    .. note::
        - For ``monc`` format, resolution is inferred from metadata and assumed
            isotropic in x and y
        - For ``um_ideal`` format, resolution must be explicitly provided via `kwargs`.
        - For ``um_real`` format, resolution is not required and
            ``attrs["h_resolution"]`` is set to ``None``.
        - For ``sgs`` format, if h_resolution is not a dataset attribute
            it is guessed by the spacing in "x" and "y" coordinates
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

    elif "um" in input_format:
        if input_format == "um_ideal":
            assert kwargs.get("resolution"), "missing resolution reqiured for um_ideal"
            res = kwargs["resolution"]
        elif input_format == "um_real":
            res = None
        else:
            raise ValueError(
                f"Unknown um format: {input_format}, accept 'um_real' and 'um_ideal'"
            )

        if interp_grid:
            simulation = data_ingest_UM_on_single_grid(
                input_files,
                requested_fields=requested_fields,
                res=res,
                field_names_dict=kwargs.get("field_names_dict"),
            )
        else:
            simulation = data_ingest_UM(
                input_files,
                requested_fields=requested_fields,
                res=res,
                field_names_dict=kwargs.get("field_names_dict"),
            )
        simulation.attrs["h_resolution"] = res

    elif input_format == "monc":
        if interp_grid:
            meta, simulation = data_ingest_MONC_on_single_grid(
                input_files,
                requested_fields=requested_fields,
            )
        else:
            meta, simulation = data_ingest_MONC(
                input_files,
                requested_fields=requested_fields,
            )
        # overwrite resolution
        assert np.isclose(meta["dxx"], meta["dyy"])
        simulation.attrs["h_resolution"] = meta["dxx"]
    else:
        raise ValueError(f"Unsupported input format {input_format}")
    return simulation

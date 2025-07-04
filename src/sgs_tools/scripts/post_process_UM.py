from pathlib import Path
from typing import Any, Dict, Sequence

import dask
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from sgs_tools.diagnostics.directional_profile import directional_profile
from sgs_tools.diagnostics.spectra import spectra_1d_radial
from sgs_tools.io.netcdf_writer import NetCDFWriter

from sgs_tools.util.timer import timer

base_fields = ["u", "v", "w", "theta"]
v_profile_fields_out = [
    "vel",
    "theta",
    "s",
    "vert_heat_flux",
    "vert_mom_flux",
    "fluct_ke",
    "tke",
    "smag_visc_m",
    "smag_visc_h",
    "smag_visc_m_vert",
    "smag_visc_h_vert",
    "csDelta",
    "cs_theta",
    "cs_diag",
    "cs_theta_diag",
]

v_profile_fields_in = [
    # base
    "u",
    "v",
    "w",
    "theta",
    # sgs
    "tke",
    "csDelta",
    "cs",
    "cs_theta",
    "cs_1",
    "cs_2",
    "cs_3",
    "cs_theta_1",
    "cs_theta_2",
    "cs_theta_3",
    # diffusivities
    "s",
    "smag_visc_m",
    "smag_visc_h",
    "smag_visc_m_vert",
    "smag_visc_h_vert",
    # stability
    # "Richardson",
]

power_spectra_fields = ["u", "v", "w", "theta"]
cross_spectra_fields = [
    ("u", "w"),
    ("v", "w"),
    ("u", "v"),
    ("theta", "w"),
]

all_fields = (
    set(base_fields)
    .union(v_profile_fields_in)
    .union(power_spectra_fields)
)

v_prof_name = "post_proc_vert_profiles.nc"
spectra_name = "post_proc_spectra.nc"


def parse_args(arguments: Sequence[str] | None = None) -> Dict[str, Any]:
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    from sgs_tools.scripts.arg_parsers import add_dask_group, add_input_group

    parser = ArgumentParser(
        description="""Post process a simulation and save result as NetCDF files
                """,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    add_input_group(parser)

    output = parser.add_argument_group("Output datasets on disk")
    output.add_argument(
        "output_path",
        type=Path,
        help="""output directory, where to store post-processed results
                will create any missing intermediate directories""",
    )
    output.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="""behaviour if diagnostic file already exists. default: skip""",
    )
    analysis = parser.add_argument_group("Analysis")

    analysis.add_argument(
        "--vertical_profiles",
        action="store_true",
        help="""Vertical profiles""",
    )

    analysis.add_argument(
        "--horizontal_spectra",
        action="store_true",
        help="""Horizontal power spectra and cross spectra
            """,
    )

    add_dask_group(parser)

    # parse arguments into a dictionary
    args = vars(parser.parse_args(arguments))

    # add any pottentially missing file extension
    assert args["output_path"].is_dir()

    # parse negative values in the [t,z]_range
    if args["t_range"][0] < 0:
        args["t_range"][0] = -np.inf
    if args["t_range"][1] < 0:
        args["t_range"][1] = np.inf

    args["t_range="] = np.sort(args["t_range"])

    if args["z_range"][0] < 0:
        args["z_range"][0] = -np.inf
    if args["z_range"][1] < 0:
        args["z_range"][1] = np.inf
    args["z_range="] = np.sort(args["z_range"])

    return args


def read(
    fname: Path,
    resolution: float,
    requested_fields: list[str],
    t_range: Sequence[float],
    z_range: Sequence[float],
) -> xr.Dataset:
    from sgs_tools.io.um import data_ingest_UM_on_single_grid

    simulation = data_ingest_UM_on_single_grid(
        fname,
        resolution,
        requested_fields=requested_fields,
    )
    simulation = data_slice(simulation, t_range, z_range)

    return simulation


def data_slice(
    ds: xr.Dataset, t_range: Sequence[float], z_range: Sequence[float]
) -> xr.Dataset:
    """restrit ds to the intervals inside [t,z]_range.
       Restrict a set of standard coordinate names for t and z.
    :param ds: input dataset/dataarray
    :param t_range: time interval
    :param t_range: verical interval
    """

    for z in "z", "z_rho", "z_theta":
        if z in ds:
            zslice = ds[z].where((ds[z] >= z_range[0]) * (ds[z] <= z_range[1]))
            z0 = zslice.argmin(...)[z].item()
            z1 = zslice.argmax(...)[z].item() + 1

            ds = ds.isel({z: slice(z0, z1)})
    for t in "t", "t_0":
        if t in ds:
            tslice = ds[t].where((ds[t] >= t_range[0]) * (ds[t] <= t_range[1]))
            t0 = tslice.argmin(...)[t].item()
            t1 = tslice.argmax(...)[t].item() + 1
            ds = ds.isel({t: slice(t0, t1)})
    return ds


# create simple post-processing fields
def post_process_fields(simulation: xr.Dataset) -> xr.Dataset:
    from sgs_tools.geometry.tensor_algebra import Frobenius_norm
    from sgs_tools.physics.fields import (
        Fluct_TKE,
        compose_vector_components_on_grid,
        strain_from_vel,
        vertical_heat_flux,
    )

    simulation["vel"] = compose_vector_components_on_grid(
        [simulation["u"], simulation["v"], simulation["w"]],
        target_dims=["x", "y", "z"],
    )

    simulation["vert_heat_flux"] = vertical_heat_flux(
        simulation["w"], simulation["theta"], ["x", "y"]
    )
    simulation["Sij"] = strain_from_vel(
        simulation["vel"],
        space_dims=["x", "y", "z"],
        vec_dim="c1",
        new_dim="c2",
        make_traceless=True,
    )
    simulation["s"] = Frobenius_norm(simulation["Sij"], ["c1", "c2"])

    simulation["vert_mom_flux"] = simulation["vel"] * simulation["w"]

    simulation["fluct_ke"] = Fluct_TKE(
        simulation["u"], simulation["v"], simulation["w"], ["x", "y", "z"], ["x", "y"]
    )

    if all([diag in simulation for diag in ["cs_1", "cs_2", "cs_3"]]):
        simulation["cs_diag"] = compose_vector_components_on_grid(
            [simulation["cs_1"], simulation["cs_2"], simulation["cs_3"]],
            target_dims=[],
            vector_dim="c1",
        )
    else:
        print(sorted(simulation))
        print("Skipping missing inputs for cs_diag")

    if all([diag in simulation for diag in ["cs_theta_1", "cs_theta_2", "cs_theta_3"]]):
        simulation["cs_theta_diag"] = compose_vector_components_on_grid(
            [
                simulation["cs_theta_1"],
                simulation["cs_theta_2"],
                simulation["cs_theta_3"],
            ],
            target_dims=[],
            vector_dim="c1",
        )
    else:
        print(sorted(simulation))
        print("Skipping missing inputs for cs_theta_diag")

    return simulation


def main(args: Dict[str, Any]) -> None:
    simulation = read(
        args["input_files"],
        args["h_resolution"],
        list(all_fields),
        args["t_range"],
        args["z_range"],
    )
    simulation = post_process_fields(simulation)

    writer = NetCDFWriter(overwrite=args["overwrite_existing"])
    output_dir = args["output_path"]

    hdims = ["x", "y"]

    if args["vertical_profiles"]:
        with timer("Vertical profiles", "s"):
            f_pr = [f for f in v_profile_fields_out if f in simulation]
            f_missing = [f for f in v_profile_fields_out if f not in simulation]
            if f_missing:
                print(f"Missing vertical profile fields {f_missing}")
            # don't overwrite but skip existing filters/scales
            output_path = output_dir / v_prof_name
            if writer.check_filename(output_path) and not writer.overwrite:
                print(f"Warning: Skip existing file {output_path}.")
            else:
                profile = directional_profile(
                    simulation[f_pr], hdims, ["mean", "std", "median"]
                )
                # rechunk for IO optimisation??
                # have to do explicit rechunking because UM date-time coordinate is an object
                # profile = profile.chunk({})
                writer.write(profile, output_path)

    if args["horizontal_spectra"]:
        with timer("Horizontal spectra", "s", "Horizontal spectra"):
            cross_fields_list = set([f for fl in cross_spectra_fields for f in fl])
            spec_fields = set(power_spectra_fields).union(cross_fields_list)

            output_path = output_dir / spectra_name
            if writer.check_filename(output_path) and not writer.overwrite:
                print(f"Warning: Skip existing file {output_path}.")
            else:
                spec_ds = spectra_1d_radial(
                    simulation[spec_fields],
                    hdims,
                    power_spectra_fields,
                    cross_spectra_fields,
                    radial_smooth_factor=1,
                )

                # rechunk for IO optimisation ??
                # have to do explicit rechunking because UM date-time coordinate is an object
                # spec_ds = spec_ds.chunk({dim: "auto" for dim in ["x", "y", "z"] if dim in spec_ds.dims})
                writer.write(spec_ds, output_path)


if __name__ == "__main__":
    args = parse_args()
    with timer("Total execution time", "min"):
        main(args)

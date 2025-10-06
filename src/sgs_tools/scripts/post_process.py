import warnings
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import xarray as xr

from sgs_tools.diagnostics.anisotropy import anisotropy_analysis
from sgs_tools.diagnostics.directional_profile import directional_profile
from sgs_tools.diagnostics.spectra import spectra_1d_radial
from sgs_tools.io.netcdf_writer import NetCDFWriter
from sgs_tools.io.read import read
from sgs_tools.util.gitinfo import get_git_state, print_git_state, write_git_diff_file
from sgs_tools.util.timer import timer

v_profile_fields_out = [
    "vel",
    "vel_horiz",
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
v_profiles_stats = ["mean", "std", "median"]
v_prof_name = "post_proc_vert_profiles.nc"


power_spectra_fields = ["u", "v", "w", "theta"]
cross_spectra_fields = [
    ("u", "w"),
    ("v", "w"),
    ("u", "v"),
    ("theta", "w"),
]
spectra_name = "post_proc_spectra.nc"


anisotropy_fields = ["u", "v", "w"]
box_delta_scales = [2, 4, 8, 16]
box_meter_scales = [800, 400, 200, 100]
box_domain_scales = [1, 0.5, 0.25]
gauss_scales = [2, 4]
filter_shapes = ["gauss", "box", "coarse"]

anisotropy_name = r"post_proc_anisotropy"


def parse_args(arguments: Sequence[str] | None = None) -> Dict[str, Any]:
    from argparse import (
        ArgumentDefaultsHelpFormatter,
        ArgumentParser,
        ArgumentTypeError,
    )

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
        help="""
        output directory, where to store post-processed results
        will create any missing intermediate directories""",
    )
    output.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="""behaviour if diagnostic file already exists. default: skip""",
    )

    parser.add_argument(
        "--hdims",
        default=["x", "y"],
        nargs="+",
        help="""List of horizontal dimension names.""",
    )

    vprof = parser.add_argument_group("Vertical Profiles")

    vprof.add_argument(
        "--vertical_profiles",
        action="store_true",
        help="""Swtitch for computing vertical profiles.""",
    )

    vprof.add_argument(
        "--vprofile_fields",
        nargs="+",
        default=v_profile_fields_out,
        type=str,
        help="""List of fields whose vertical profile to compute.""",
    )

    vprof.add_argument(
        "--vprofile_stats",
        nargs="+",
        default=v_profiles_stats,
        type=str,
        help="""Statistics to generate vertical profiles with.""",
    )

    vprof.add_argument(
        "--vprofile_fname_out",
        default=v_prof_name,
        type=str,
        help="""Filename where to save the generated vertical profile. relative to output_path""",
    )

    spectra = parser.add_argument_group("Horizontal spectra")

    spectra.add_argument(
        "--horizontal_spectra",
        action="store_true",
        help="""Horizontal power spectra and cross spectra.""",
    )

    spectra.add_argument(
        "--power_spectra_fields",
        nargs="+",
        default=power_spectra_fields,
        type=str,
        help="""Fields whose power spectra to compute.""",
    )

    def tuple_from_comma_str(s):
        tup = tuple(s.split(","))
        if len(tup) != 2:
            raise ArgumentTypeError(
                f"Expected only 2 fields for each comma-separated cross-spectrum, got {len(tup)}"
            )
        return tup

    spectra.add_argument(
        "--cross_spectra_fields",
        nargs="+",
        default=cross_spectra_fields,
        type=tuple_from_comma_str,
        help="""
        Fields whose cross spectra to compute.
        Use spaces to between cross spectra and commas between fields in each cross-spectrim, e.g.
        u,v v,w""",
    )

    spectra.add_argument(
        "--radial_smooth_factor",
        default=1,
        type=int,
        help="""Radial binning of radial horizontal spectrum in units of delta_kx spacings""",
    )

    spectra.add_argument(
        "--radial_truncation",
        action="store_true",
        help="""
        Truncation of radial horizontal spectrum.
        If True will disregard wavenumbers above the maximum linear wavenumber.
        """,
    )

    spectra.add_argument(
        "--hspectra_fname_out",
        default=spectra_name,
        type=str,
        help="""Filename where to save the generated horisontal spectra. relative to output_path""",
    )

    anisotropy = parser.add_argument_group("Anisotropy diagnostics")

    anisotropy.add_argument(
        "--anisotropy",
        action="store_true",
        help="""Anisotropy of velocity strain and stress""",
    )

    anisotropy.add_argument(
        "--box_domain_scales",
        nargs="+",
        default=box_domain_scales,
        type=float,
        help="""
        Anisotropy box filter and coarse-graining scales in fraction of the horizontal domain size.
        Will round to nearest integer number of horizontal grid cells.
        Will combine all box scales and ignore entries which are less than `2 delta` apart.
        """,
    )

    anisotropy.add_argument(
        "--box_meter_scales",
        nargs="+",
        default=box_meter_scales,
        type=float,
        help="""
        Anisotropy box filter and coarse-graining scales in meters.
        Will round to nearest integer number of horizontal grid cells.
        Will combine all box scales and ignore entries which are less than `2 delta` apart
        """,
    )

    anisotropy.add_argument(
        "--box_delta_scales",
        nargs="+",
        default=box_delta_scales,
        type=int,
        help="""
        Anisotropy box filter and coarse-graining scales in units of horizontal grid spacing `delta`.
        Will combine all box scales and ignore entries which are less than `2 delta` apart.
        """,
    )
    anisotropy.add_argument(
        "--gauss_scales",
        nargs="+",
        default=gauss_scales,
        type=int,
        help="""Anisotropy Gaussian filter scales in  units of horizontal grid spacing. Support 2 and 4""",
    )
    anisotropy.add_argument(
        "--filter_shapes",
        nargs="+",
        default=filter_shapes,
        type=str,
        help=f"""Anisotropy filter shapes. Support any of {filter_shapes}""",
    )

    spectra.add_argument(
        "--aniso_fname_out",
        default=anisotropy_name,
        type=Path,
        help="""
        **Core** filename where to save the generated anisotropy eigen values. relative to output_path.
        Will append the filter label. Will set an '.nc' extension (whether given or not).
        """,
    )

    add_dask_group(parser)

    # parse arguments into a dictionary
    args = vars(parser.parse_args(arguments))

    # check io group consistency
    if args["input_format"] == "um":
        assert args["h_resolution"] > 0, (
            "Missing required a positive h_resolution for UM datasets"
        )
    assert args["output_path"].is_dir()

    # parse negative values in the [t,z]_range
    if args["t_range"][0] < 0:
        args["t_range"][0] = -np.inf
    if args["t_range"][1] < 0:
        args["t_range"][1] = np.inf

    args["t_range="] = sorted(args["t_range"])

    if args["z_range"][0] < 0:
        args["z_range"][0] = -np.inf
    if args["z_range"][1] < 0:
        args["z_range"][1] = np.inf
    args["z_range="] = sorted(args["z_range"])

    return args


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
        vector_dim="c1",
    )
    # horizontal wind
    simulation["vel_horiz"] = (simulation["vel"].sel(c1=[1, 2]) ** 2).sum("c1") ** 0.5

    if all([diag in simulation for diag in ["theta", "w"]]):
        simulation["vert_heat_flux"] = vertical_heat_flux(
            simulation["w"], simulation["theta"], ["x", "y"]
        )
    else:
        print(
            "Skipping missing inputs for cs_theta_diag: "
            "['cs_theta_1', 'cs_theta_2', 'cs_theta_3']"
        )
        print("Available fields:", sorted(simulation, key=str))

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
        print("Skipping missing inputs for cs_diag: ['cs_1', 'cs_2', 'cs_3']")
        print("Available fields:", sorted(simulation, key=str))

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
        print(
            "Skipping missing inputs for cs_theta_diag: "
            "['cs_theta_1', 'cs_theta_2', 'cs_theta_3']"
        )
        print("Available fields:", sorted(simulation, key=str))

    return simulation


# select filters based on number of points and horizontal spacing
def choose_filter_set(
    hminsize: int,  # number of grid points in horizontal direction
    dx: float,  # horizontal grid spacing
    box_delta_scales: Sequence[int] = [
        2,
        4,
        8,
        16,
    ],  # effective resolution/decorelation scales
    box_meter_scales: Sequence[float] = [
        800.0,
        400.0,
        200.0,
        100.0,
    ],  # sub-km grey zone horizontal resolutions
    box_domain_scales: Sequence[float] = [0.25, 0.5, 1],  # domain unit scales
    gauss_scales: Sequence[float] = [2, 4],  # Gaussian filter scales
    filter_shapes: Sequence[str] = ["gauss", "box", "coarse"],  # filter shapes
    hdims: Sequence[str] = ["x", "y"],
):
    from sgs_tools.sgs.coarse_grain import CoarseGrain
    from sgs_tools.sgs.filter import (
        Filter,
        box_kernel,
        weight_gauss_3d,
        weight_gauss_5d,
    )

    # dictionary of filters
    filter_dic: Dict[str, Filter | CoarseGrain] = {}

    # Box-cart scales
    meter_scales = [int(res / dx) for res in box_meter_scales]
    domain_scales = [int(hminsize * x) for x in box_domain_scales]
    box_scales: list[float] = []
    # only allows scales at least 2 (delta_<x>) apart
    # order of loop determines precedence
    for x in set(domain_scales + meter_scales + list(box_delta_scales)):
        if x > 1 and x <= hminsize:
            if not box_scales or np.min(np.abs([x - y for y in box_scales])) >= 2:
                box_scales += [x]

    # sort from fast to slow to compute coarse grain(small to large scales)
    box_scales = sorted(box_scales)
    assert max(box_scales) <= hminsize, (
        "Unsupported box_scales greater than domain size."
    )
    assert min(box_scales) > 1, "Unsupported box_scales less than 0."

    # Coarse-graining filters
    if "coarse" in filter_shapes:
        filter_dic = filter_dic | {
            f"Coarse{scale}delta": CoarseGrain({x: int(scale) for x in hdims})
            for scale in box_scales
        }

    # Box filters
    # stencil size is (filter_scale + 1)delta under finite-difference data interpretation
    if "box" in filter_shapes:
        box_scales = [x for x in box_scales if x <= hminsize // 10]
        filter_dic = filter_dic | {
            f"Box{scale}delta": Filter(
                box_kernel([int(scale) + 1 for x in hdims]), hdims
            )
            for scale in box_scales
        }

    if "gauss" in filter_shapes:
        # Gausssian filters
        for s in gauss_scales:
            if s == 2:
                filter_dic[f"Gauss{2}delta"] = Filter(
                    weight_gauss_3d, filter_dims=hdims
                )
            elif s == 4:
                filter_dic[f"Gauss{4}delta"] = Filter(
                    weight_gauss_5d, filter_dims=hdims
                )
            else:
                warnings.warn(
                    f"Skipping unsupported Gauss scale {s}. Support only 2 and 4."
                )

    return filter_dic


def run(args: Dict[str, Any]) -> None:
    spectra_fields_list = set(
        [f for fl in args["cross_spectra_fields"] for f in fl]
        + args["power_spectra_fields"]
    )

    all_fields = (
        set()
        .union(v_profile_fields_in)
        .union(spectra_fields_list)
        .union(anisotropy_fields)
    )

    simulation = read(
        args["input_files"],
        args["input_format"],
        list(all_fields),
        resolution=args["h_resolution"],
    )
    # slice to the requested sub-domain
    simulation = data_slice(simulation, args["t_range"], args["z_range"])
    simulation = post_process_fields(simulation)

    writer = NetCDFWriter(overwrite=args["overwrite_existing"])
    output_dir = args["output_path"]

    hdims = args["hdims"]

    # get repo state and setup as attributes of netcdf
    git_info = get_git_state(2)
    git_attrs = {"git_commit": git_info["Commit"]}
    if git_info.get("Changes"):
        git_attrs["git_diff_file"] = write_git_diff_file(args["output_path"])

    if args["vertical_profiles"]:
        with timer("Vertical profiles", "s"):
            f_pr = [f for f in args["vprofile_fields"] if f in simulation]
            f_missing = [f for f in args["vprofile_fields"] if f not in simulation]
            if f_missing:
                print(f"Missing vertical profile fields {f_missing}")
            # don't overwrite but skip existing filters/scales
            output_path = output_dir / args["vprofile_fname_out"]
            if writer.check_filename(output_path) and not writer.overwrite:
                print(f"Warning: Skip existing file {output_path}.")
            else:
                profile = directional_profile(
                    simulation[f_pr], hdims, args["vprofile_stats"]
                )
                with timer(f"write {output_path}", "s"):
                    # rechunk for IO optimisation??
                    # have to do explicit rechunking because UM date-time coordinate is an object
                    profile = profile.chunk({"z": "auto"})
                    profile.attrs.update(git_attrs)
                    writer.write(profile, output_path)

    if args["horizontal_spectra"]:
        with timer("Horizontal spectra", "s", "Horizontal spectra"):
            pspec_fields = [f for f in args["power_spectra_fields"] if f in simulation]
            cspec_fields = [
                s
                for s in args["cross_spectra_fields"]
                if s[0] in simulation and s[1] in simulation
            ]
            cross_fields_set = set([f for fl in cspec_fields for f in fl])
            spec_fields = cross_fields_set.union(pspec_fields)
            # get the missing fields from the power-spectra fields
            # since they contain all the cross-spectra fields
            spec_f_missing = set(args["power_spectra_fields"]) - set(pspec_fields)
            if spec_f_missing:
                print(f"Missing spectra fields {spec_f_missing}")

            output_path = output_dir / args["hspectra_fname_out"]
            if writer.check_filename(output_path) and not writer.overwrite:
                print(f"Warning: Skip existing file {output_path}.")
            else:
                spec_ds = spectra_1d_radial(
                    simulation[spec_fields],
                    hdims,
                    pspec_fields,
                    cspec_fields,
                    radial_smooth_factor=args["radial_smooth_factor"],
                    radial_truncation=args["radial_truncation"],
                )

                with timer(f"write {output_path}", "s"):
                    # rechunk for IO optimisation ??
                    # have to do explicit rechunking because UM date-time coordinate is an object
                    spec_ds = spec_ds.chunk(
                        {dim: "auto" for dim in ["x", "y", "z"] if dim in spec_ds.dims}
                    )
                    spec_ds.attrs.update(git_attrs)
                    writer.write(spec_ds, output_path)

    if args["anisotropy"]:
        with timer("Anisotropy", "s", "Anisotropy"):
            # anisotropy diagnostic

            # min horizontal number of cells
            hminsize = min([simulation[x].size for x in hdims])
            # assume horizontally square cells
            dx = (simulation[hdims[0]][1] - simulation[hdims[0]][0]).item()

            # set up the a dictionary of filters
            filter_dic = choose_filter_set(
                hminsize=hminsize,
                dx=dx,
                box_delta_scales=args["box_delta_scales"],
                box_meter_scales=args["box_meter_scales"],
                box_domain_scales=args["box_domain_scales"],
                gauss_scales=args["gauss_scales"],
                filter_shapes=args["filter_shapes"],
                hdims=hdims,
            )

            print(f"Filters: {filter_dic.keys()}")

            # rechunk velocity -- unify filtering and vector dimensions
            vel = simulation["vel"].chunk({x: -1 for x in hdims + ["c1"]}).persist()
            # run analysis
            for filt_lbl, filt in filter_dic.items():
                with timer(
                    f"{filt_lbl}:{filt.scales()}", "s", f"{filt_lbl}:{filt.scales()}"
                ):
                    output_path = (
                        output_dir / f"{args['aniso_fname_out'].stem}_{filt_lbl}"
                    ).with_suffix(".nc")
                    # don't over-write but skip existing filters/scales
                    if not writer.overwrite and writer.check_filename(output_path):
                        print(f"Warning: Skip existing file {output_path}.")
                    else:
                        evals = anisotropy_analysis(vel, filt)
                        evals = evals.expand_dims(
                            {"filter": xr.DataArray([filt_lbl], dims=["filter"])}
                        )
                        with timer(f"write {output_path}", "s"):
                            # rechunk for IO optimisation
                            # have to do explicit rechunking because UM date-time coordinate is an object
                            evals.chunk(
                                {
                                    dim: "auto"
                                    for dim in ["x", "y", "z", "c1", "c2"]
                                    if dim in evals.dims
                                }
                            )
                            evals.attrs.update(git_attrs)
                            writer.write(evals, output_path)


def main() -> None:
    args = parse_args()
    if args["version"]:
        print_git_state(args["version"])
        exit()
    with timer("Total execution time", "min"):
        run(args)


if __name__ == "__main__":
    main()

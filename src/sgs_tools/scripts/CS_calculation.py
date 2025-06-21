from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from numpy import inf
from sgs_tools.io.monc import data_ingest_MONC_on_single_grid
from sgs_tools.io.um import data_ingest_UM_on_single_grid
from sgs_tools.scripts.arg_parsers import (
    add_dask_group,
    add_input_group,
    add_plotting_group,
)
from sgs_tools.scripts.CS_calculation_genmodel import (
    compute_cs,
    data_slice,
    gather_model_inputs,
    make_filter,
    model_name_map,
    model_selection,
    plot,
)
from sgs_tools.util.path_utils import add_extension
from sgs_tools.util.timer import timer


def parser() -> dict[str, Any]:
    parser = ArgumentParser(
        description="""Compute dynamic Smagorinsky coefficients as function
                        of scale from UM NetCDF output and store them in
                        a NetCDF files""",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    fname = add_input_group(parser)
    fname.add_argument(
        "output_file",
        type=Path,
        help="""output path, will create/overwrite existing file and
                create any missing intermediate directories""",
    )
    fname.add_argument("--input_format", type=str, choices=["um", "monc"], default="um")

    add_plotting_group(parser)
    add_dask_group(parser)

    filter = parser.add_argument_group("Filter parameters")
    filter.add_argument(
        "--filter_type",
        type=str,
        default="box",
        choices=["box", "gaussian"],
        help="Shape of filter kernel to use for scale separation.",
    )

    filter.add_argument(
        "--filter_scales",
        type=int,
        default=(2,),
        nargs="+",
        help="Scales to perform filter at, in number of cells. "
        "If a single value is given, it will be used for all `regularize_filter_scales`. "
        "Otherwise, must provide as many values as for `regularize_filter_scales`",
    )

    filter.add_argument(
        "--regularize_filter_type",
        type=str,
        default="box",
        choices=["box", "gaussian"],
        help="Shape of filter kernel used for coefficient regularization.",
    )

    filter.add_argument(
        "--regularize_filter_scales",
        type=int,
        default=(2,),
        nargs="+",
        help="Scales to perform regularization at, in number of cells. "
        "If a single value is given, it will be used for all `filter_scale`. "
        "Otherwise, must provide as many values as for `filter_scale`",
    )

    # parse arguments into a dictionary
    args = vars(parser.parse_args())

    # parameter consistency checks
    if args["filter_type"] == "gaussian":
        assert all(
            [x in [2, 4] for x in args["filter_scales"]]
        ), "Gaussian filters only support scales 2 and 4 for now..."

    # singleton filter_scales or regularize_filter_scales means broadcast against the other
    if len(args["filter_scales"]) == 1:
        args["filter_scales"] = args["filter_scales"] * len(
            args["regularize_filter_scales"]
        )

    if len(args["regularize_filter_scales"]) == 1:
        args["regularize_filter_scales"] = args["regularize_filter_scales"] * len(
            args["filter_scales"]
        )

    assert len(args["filter_scales"]) == len(args["regularize_filter_scales"])

    # add any pottentially missing file extension
    args["output_file"] = add_extension(args["output_file"], ".nc")

    # parse negative values in the [t,z]_range
    if args["t_range"][0] < 0:
        args["t_range"][0] = -inf
    if args["t_range"][1] < 0:
        args["t_range"][1] = inf

    if args["z_range"][0] < 0:
        args["z_range"][0] = -inf
    if args["z_range"][1] < 0:
        args["z_range"][1] = inf
    return args


def read(args: dict[str, Any]) -> xr.Dataset:
    # read UM stash files
    if args["input_format"] == "um":
        simulation = data_ingest_UM_on_single_grid(
            args["input_files"],
            args["h_resolution"],
            requested_fields=args["requested_fields"],
        )
    elif args["input_format"] == "monc":
        # read MONC files
        meta, simulation = data_ingest_MONC_on_single_grid(
            args["input_files"],
            requested_fields=args["requested_fields"],
        )
        # overwrite resolution
        assert np.isclose(meta["dxx"], meta["dyy"])
        args["h_resolution"] = meta["dxx"]
    simulation = data_slice(simulation, args["t_range"], args["z_range"])
    simulation = simulation.chunk(
        {
            "z": args["z_chunk_size"],
            # "z_theta": args["z_chunk_size"],
            "t_0": args["t_chunk_size"],
        }
    )
    return simulation


def main() -> None:
    # read and pre-process simulation
    with timer("Arguments", "ms"):
        args = parser()
    print(args)
    args["requested_fields"] = ["u", "v", "w", "theta"]
    # read UM stasth files: data
    with timer("Read Dataset", "s"):
        simulation = read(args)

    # check scales make sense
    nhoriz = min(simulation["x"].shape[0], simulation["y"].shape[0])
    for scale in args["filter_scales"]:
        assert scale in range(
            1, nhoriz
        ), f"scale {scale} must be less than horizontal number of grid cells {nhoriz}"
    for scale in args["regularize_filter_scales"]:
        assert (
            scale in range(1, nhoriz)
        ), f"regularization_scale {scale} must be less than horizontal number of grid cells {nhoriz}"

    with timer("Extract grid-based fields", "s"):
        simulation = gather_model_inputs(simulation)
        output = simulation["vel", "sij", "grad_theta"]

    with timer("Setup filtering operators"):
        test_filters = []
        regularization_filters = []
        for scale, regularization_scale in zip(
            args["filter_scales"], args["regularize_filter_scales"]
        ):
            test_filters.append(make_filter(args["filter_type"], scale, ["x", "y"]))
            regularization_filters.append(
                make_filter(
                    args["regularize_filter_type"], regularization_scale, ["x", "y"]
                )
            )
    for m in "Smag_vel", "Smag_vel_diag", "Smag_theta", "Smag_theta_diag":
        # setup dynamic model
        with timer(f"Coeff calculation SETUP for {model_name_map[m]} model", "s"):
            dynamic_model = model_selection(m, simulation, args["h_resolution"])

            coeff = compute_cs(
                dynamic_model,
                test_filters,
                regularization_filters,
            )
            # for multi-coefficient models
            if "cdim" in coeff.dims:
                coeff = coeff.rename({"cdim": "c1"})
        out_fname = args["output_file"].with_stem(
            f'{model_name_map[m]}_{args["output_file"].stem}'
        )

        # trigger computation
        with timer(f"Coeff calculation compute for {model_name_map[m]} model", "s"):
            with ProgressBar():
                coeff.compute()
        # write to disk
        with timer(f"Coeff calculation write for {model_name_map[m]} model", "s"):
            with ProgressBar():
                coeff.to_netcdf(
                    out_fname,
                    mode="w",
                    compute=True,
                    unlimited_dims=["scale"],
                    engine="h5netcdf",
                )

    # plot horizontal mean profiles
    if args["plot_show"] or args["plot_path"] is not None:
        try:
            with timer("Plotting", "s"):
                plot(args)
        except:
            print("Failed in generating plots")

    # interactive plotting out of time
    if args["plot_show"]:
        plt.show()

    with timer("Write to disk", "s"):
        with ProgressBar():
            output.to_netcdf(
                args["output_file"],
                mode="w",
                compute=True,
                unlimited_dims=["scale"],
                engine="h5netcdf",
            )


if __name__ == "__main__":
    with timer("Total execution time", "min"):
        main()

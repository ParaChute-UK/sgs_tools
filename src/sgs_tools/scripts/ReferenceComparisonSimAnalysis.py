from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from numpy import array, inf

from sgs_tools.scripts.arg_parsers import (
    add_dask_group,
    add_plotting_group,
    add_version_group,
    parse_json_or_file,
)
from sgs_tools.scripts.BasicComparisonSimAnalysis import (
    io,
    plot,
    prof_fields,
    slice_fields,
)
from sgs_tools.scripts.cli_helpers import print_args_dict, print_header
from sgs_tools.scripts.plotting import configure_matplotlib_backend
from sgs_tools.util.timer import timer

default_plotting_style = [
    {
        "label": "target",
        "linestyle": "--",
        "color": "C1",
        "linewidth": 1,
        "marker": "",
    },
    {
        "label": "reference",
        "linestyle": "-",
        "color": "k",
        "linewidth": 1,
        "marker": "",
    },
]


def parse_args(arguments: Sequence[str] | None = None) -> dict[str, Any]:
    parser = ArgumentParser(
        description="""
                    Create (and optionally save) standard diagnostic plots for
                    a dry atmospheric boundary layer UM simulation
                    Best-suited to one-parameter suite of simulations,
                    but can handle several varying parameters through plot_style_file
                """,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    add_version_group(parser)
    fname = parser.add_argument_group("I/O datasets on disk")
    fname.add_argument(
        "target",
        type=Path,
        help="""
            Location of target simulation outputs -- UM NetCDF diagnostic files.
            Recognizes glob patterns and walks directory trees,
            e.g. './my_file_p[br]*nc'
            Can have multiple files, but only one glob pattern.
            (All files in a glob pattern should belong to the simulation). """,
    )

    fname.add_argument(
        "reference",
        type=Path,
        help="""
            Location of reference simulation outputs -- UM NetCDF diagnostic files.
            Recognizes glob patterns and walks directory trees,
            e.g. './my_file_p[br]*nc'.
            Can have multiple files, but only one glob pattern.
            (All files in a glob pattern should belong to the simulation). """,
    )

    fname.add_argument(
        "input_format",
        type=str,
        choices=["um", "monc", "sgs"],
        help="Type of 'input_files'. Only support different NetCDF flavours from "
        "various production codes. 'sgs' refers to files produced by sgs_tools. "
        "All simulations must have the same format",
    )

    fname.add_argument(
        "--h_resolution",
        type=float,
        nargs="+",
        default=[0],
        help="""
        horizontal resolution in meters.
        *ONLY* used for UM ideal simulations
        (will use to overwrite horizontal coordinates).
        If a single resolution is given, assume it applies to all input files.
        Else, must give as many resolutions as inpu_file glob patterns.
        """,
    )

    fname.add_argument(
        "--times",
        type=float,
        nargs="*",
        default=[],
        help="""
              times at which to perform the analysis;
              in code coordinates; will find nearest available match.
              default (which is empty) means the full data range at 1h intervals.
             """,
    )

    fname.add_argument(
        "--z_range",
        type=float,
        nargs=2,
        default=[-1, -1],
        help="vertical interval to consider, in code coordinates, "
        "negative values are interpreted as take the min/max respectively",
    )

    plotting = add_plotting_group(parser)
    plotting.add_argument(
        "--plot_styles",
        type=parse_json_or_file,
        default=None,
        help="""
                JSON configuration describing a list of plot styles and decorations
                matched sequentially to the target and reference.
                Can pass as a json-compatible string, but better a path to a JSON file.
                See plot_config_template.json for a template.
                If absent, will use ``default_plotting_style``.
            """,
    )

    plotting.add_argument(
        "--hor_slice_levels",
        type=float,
        nargs="*",
        default=[],
        help="""
                Vertical height at which to plot horizontal slices.
                If not given will omit these plots.
            """,
    )

    plotting.add_argument(
        "--skip_vert_profiles",
        action="store_true",
        help="skip vertical profiles from plotting",
    )

    plotting.add_argument(
        "--skip_clouds",
        action="store_true",
        help="skip cloud plot",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="""More granular command line output.""",
    )
    add_dask_group(parser)

    # parse arguments into a dictionary
    args = vars(parser.parse_args(arguments))

    # collated input files. Order matters -- should match order in plotting_styles
    args["input_files"] = [args["target"], args["reference"]]
    if len(args["h_resolution"]) == 1:
        args["h_resolution"] = [args["h_resolution"][0]] * len(args["input_files"])
    else:
        if args["input_format"] == "um":
            assert len(args["h_resolution"]) == len(args["input_files"])

    # initial validation
    assert args["plot_show"] or args["plot_path"], (
        "require at least one of 'plot_show' or  'plot_path'"
    )

    # parse plotting style
    if args["plot_styles"] is None:
        plot_styles = default_plotting_style
    else:
        plot_styles = args["plot_styles"]
        # ensure we have enough plotting styles
        assert len(plot_styles) == len(args["input_files"])

    args["plot_map"] = plot_styles

    # parse negative values in the [t,z]_range
    args["times"] = array(args["times"])
    assert all(args["times"] >= 0)

    if args["z_range"][0] < 0:
        args["z_range"][0] = -inf
    if args["z_range"][1] < 0:
        args["z_range"][1] = inf
    assert all(
        args["z_range"][0] <= z <= args["z_range"][1] for z in args["hor_slice_levels"]
    ), (
        f"hor_slice_levels {args['hor_slice_levels']} aren't "
        f"contained in z_range {args['z_range']}"
    )
    return args


def main(arguments: Sequence[str] | None = None) -> None:
    with timer("Total execution time", "min"):
        with timer("Arguments", "ms"):
            # needs to happen before any plotting
            configure_matplotlib_backend(arguments)
            args = parse_args(arguments)
            print_header("ref_comparison")
            print_args_dict(args)

        ds_collection, field_plot_map = io(args)

        # make plots
        with timer("Make plots", "s"):
            plot(ds_collection, args, slice_fields, prof_fields, field_plot_map)

        with timer("Make error plots", "s"):
            for f in field_plot_map:
                field_plot_map[f] = field_plot_map[f].with_args(cmap="RdBu_r")
            # error
            err_collection = {
                "difference": ds_collection["target"] - ds_collection["reference"]
            }
            args["plot_map"] = [
                {
                    "label": "difference",
                    "linestyle": "-",
                    "color": "k",
                    "linewidth": 1,
                    "marker": "x",
                },
            ]
            if args["plot_path"]:
                args["plot_path"] = args["plot_path"] / "difference"
            plot(err_collection, args, slice_fields, prof_fields, field_plot_map)

            err_collection = {
                "rel_difference": 2
                * (ds_collection["target"] - ds_collection["reference"])
                / (ds_collection["target"] + ds_collection["reference"])
            }

            args["plot_map"] = [
                {
                    "label": "rel_difference",
                    "linestyle": "-",
                    "color": "k",
                    "linewidth": 1,
                    "marker": "x",
                },
            ]
            if args["plot_path"]:
                args["plot_path"] = args["plot_path"].parent / "rel_difference"
            plot(err_collection, args, slice_fields, prof_fields, field_plot_map)


if __name__ == "__main__":
    main()

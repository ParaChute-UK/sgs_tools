from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any

from numpy import array, inf

from sgs_tools.scripts.arg_parsers import add_dask_group, add_plotting_group
from sgs_tools.scripts.BasicComparisonSimAnalysis import (
    io,
    plot,
    prof_fields,
    slice_fields,
)
from sgs_tools.util.timer import timer

plotting_styles = [
    {
        "label": "target",
        "linestyle": "--",
        "color": "C1",
        "linewidth": 1,
        "marker": "x",
    },
    {
        "label": "reference",
        "linestyle": "-",
        "color": "k",
        "linewidth": 1,
        "marker": "o",
    },
]


def parse_args() -> dict[str, Any]:
    parser = ArgumentParser(
        description="""
                    Create (and optionally save) standard diagnostic plots for
                    a dry atmospheric boundary layer UM simulation
                    Best-suited to one-parameter suite of simulations,
                    but can handle several varying parameters through plot_style_file
                """,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    fname = parser.add_argument_group("I/O datasets on disk")
    fname.add_argument(
        "target",
        type=Path,
        help="""
            Location of target simulation outputs -- UM NetCDF diagnostic files.
            Recognizes glob patterns and walks directory trees, e.g. './my_file_p[br]*nc'
            Can have multiple files, but only one glob pattern.
            (All files in a glob pattern should belong to the simulation). """,
    )

    fname.add_argument(
        "reference",
        type=Path,
        help="""
            Location of reference simulation outputs -- UM NetCDF diagnostic files.
            Recognizes glob patterns and walks directory trees, e.g. './my_file_p[br]*nc'
            Can have multiple files, but only one glob pattern.
            (All files in a glob pattern should belong to the simulation). """,
    )

    fname.add_argument(
        "h_resolution",
        type=float,
        nargs=1,
        help="""
                horizontal resolution (will use to overwrite horizontal coordinates).
                must apply to both reference and target.
                **NB** works for ideal simulations""",
    )

    fname.add_argument(
        "--times",
        type=float,
        nargs="*",
        default=[],
        help="""
              times at which to perform the analysis;
              in code coordinates; will find nearest available match.
              default (which is empty) means the full data range
             """,
    )

    fname.add_argument(
        "--z_range",
        type=float,
        nargs=2,
        default=[-1, -1],
        help="vertical interval to consider, in code coordinates, negative values are interpreted as take the min/max respectively",
    )

    plotting = add_plotting_group(parser)
    plotting.add_argument(
        "--hor_slice_levels",
        type=float,
        nargs="+",
        default=[],
        help="""Vertical height at which to plot horizontal slices
            """,
    )

    plotting.add_argument(
        "--skip_vert_profiles",
        action="store_true",
        help="skip vertical profiles from plotting",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="""More granular command line output.""",
    )
    add_dask_group(parser)

    # parse arguments into a dictionary
    args = vars(parser.parse_args())

    # collated input files. Order matters -- should match order in plotting_styles
    args["input_files"] = [args["target"], args["reference"]]
    # duplicate resolution
    args["h_resolution"] = [args["h_resolution"][0], args["h_resolution"][0]]

    # initial validation
    assert args["plot_show"] or args["plot_path"], (
        "require at least one of 'plot_show' or  'plot_path'"
    )

    # add a hardcoded plotting style
    args["plot_map"] = plotting_styles

    # parse negative values in the [t,z]_range
    args["times"] = array(args["times"])
    assert all(args["times"] >= 0)

    if args["z_range"][0] < 0:
        args["z_range"][0] = -inf
    if args["z_range"][1] < 0:
        args["z_range"][1] = inf
    return args


def main():
    with timer("Total execution time", "min"):
        with timer("Arguments", "ms"):
            args = parse_args()

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

from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path


def add_input_group(parser: ArgumentParser) -> _ArgumentGroup:
    fname = parser.add_argument_group("I/O datasets on disk")
    fname.add_argument(
        "input_files",
        type=Path,
        help=""" location of UM NetCDF diagnostic file(s). Recognizes glob patterns and walks directory trees, e.g. './my_file_p[br]*nc'
              (All files in the pattern should belong to the same simulation). """,
    )

    fname.add_argument(
        "h_resolution",
        type=float,
        help="horizontal resolution (will use to overwrite horizontal coordinates). **NB** works for ideal simulations",
    )

    fname.add_argument(
        "--t_range",
        type=float,
        nargs=2,
        default=[-1, -1],
        help="time interval to consider, in code coordinates, negative value are interpreted as infinity",
    )

    fname.add_argument(
        "--z_range",
        type=float,
        nargs=2,
        default=[-1, -1],
        help="vertical interval to consider, in code coordinates, negative values are interpreted as take the min/max respectively",
    )

    return fname


def add_plotting_group(parser: ArgumentParser) -> _ArgumentGroup:
    plotting = parser.add_argument_group("Plotting parameters")

    plotting.add_argument(
        "--plot_show",
        action="store_true",
        help="flag to display generated plots",
    )

    plotting.add_argument(
        "--plot_path",
        type=Path,
        default=None,
        help="output directory, for storing generated plots",
    )

    return plotting


def add_dask_group(parser: ArgumentParser) -> _ArgumentGroup:
    dask = parser.add_argument_group("Dask parameters")

    dask.add_argument(
        "--z_chunk_size",
        type=int,
        default=None,
        help="""
      Size of dask array chunks in the vertical direction. Should divide the total number of levels.
      Smaller size leads to smaller memory footprint, but may penalize walltime.
      NB:The default value has not been optimised.""",
    )

    dask.add_argument(
        "--t_chunk_size",
        type=int,
        default=None,
        help="""
      Size of dask array chunks in the time direction. Should divide the total number time snapshots.
      Smaller size leads to smaller memory footprint, but may penalize walltime.
      NB:The default value has not been optimised.""",
    )

    return dask

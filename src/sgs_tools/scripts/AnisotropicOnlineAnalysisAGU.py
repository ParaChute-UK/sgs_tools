import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy import arange, array, inf, sqrt
from pint import UnitRegistry
from sgs_tools.io.um import data_ingest_UM

# from sgs_tools.plotting.collection_plots import (
#     plot_horizontal_slice_tseries,
#     plot_vertical_prof_time_slice_compare_sims_slice,
# )
from sgs_tools.plotting.field_plot_map import (
    anisotropic_plot_map,
    debug_field_plot_map,
    field_plot_kwargs,
    field_plot_map,
)
from sgs_tools.scripts.arg_parsers import add_dask_group, add_plotting_group
from sgs_tools.util.timer import timer

plotting_styles = [
    {
        "label": "ds_diag",
        "linestyle": "-",
        "color": "C1",
        "linewidth": 1,
        "marker": "x",
    },
    {
        "label": "ds_iso",
        "linestyle": "--",
        "color": "k",
        "linewidth": 1,
        "marker": "o",
    },
    {
        "label": "ds_smag",
        "linestyle": ":",
        "color": "k",
        "linewidth": 1,
        "marker": "o",
    },
]


slice_fields = (  # "u", "v", "w", "theta", "s",
    "cs",
    "cs_theta",
    "cs_vert",
    "cs_horiz",
    "ctheta_vert",
    "ctheta_horiz",
)
prof_fields = (
    # # base
    # "u",
    # "v",
    # "w",
    # "theta",
    # #turbulent fluxes
    # "u'w'",
    # "v'w'",
    # "u'v'",
    # "vertical_heat_flux",
    # #energetics
    # "w^2",
    # "tke",
    # length scales
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
    # "s",
    "smag_visc_m",
    "smag_visc_h",
    "smag_visc_m_vert",
    "smag_visc_h_vert",
    # # stability
    # "Richardson",
)


verbose = False


def parse_args() -> dict[str, Any]:
    parser = ArgumentParser(
        description="""Create (and optionally save) standard diagnostic plots for
                    a dry atmospheric boundary layer UM simulation
                    Best-suited to one-parameter suite of simulations,
                    but can handle several varying parameters through plot_style_file
                """,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    fname = parser.add_argument_group("I/O datasets on disk")
    fname.add_argument(
        "ds_diag",
        type=Path,
        help=""" Location of diagonal CS simulation outputs -- UM NetCDF diagnostic files.
            Recognizes glob patterns and walks directory trees, e.g. './my_file_p[br]*nc'
            Can have multiple files in, but only one glob pattern.
            (All files in a glob pattern should belong to the same simulation). """,
    )

    fname.add_argument(
        "ds_iso",
        type=Path,
        help=""" Location of isotropic CS simulation outputs -- UM NetCDF diagnostic files.
            Recognizes glob patterns and walks directory trees, e.g. './my_file_p[br]*nc'
            Can have multiple files in, but only one glob pattern.
            (All files in a glob pattern should belong to the same simulation). """,
    )

    fname.add_argument(
        "ds_smag",
        type=Path,
        help=""" Location of default Smagorinsky simulation outputs -- UM NetCDF diagnostic files.
            Recognizes glob patterns and walks directory trees, e.g. './my_file_p[br]*nc'
            Can have multiple files in, but only one glob pattern.
            (All files in a glob pattern should belong to the same simulation). """,
    )

    fname.add_argument(
        "h_resolution",
        type=float,
        nargs="+",
        help="""horizontal resolution (will use to overwrite horizontal coordinates).
                If a single resolution is given, assume it applies to all input files.
                Else, must give as many resolutions as inpu_file glob patterns.
              **NB** works for ideal simulations""",
    )

    fname.add_argument(
        "--times",
        type=float,
        nargs="*",
        default=[],
        help="""times at which to perform the analysis;
              in code coordinates; will find nearest available match.
              default (which is empty) means the full data range at 1h intervals.
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
        nargs="*",
        default=[],
        help="""Vertical height at which to plot horizontal slices.
                If not given will omit these plots.
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
    args["input_files"] = [args["ds_diag"], args["ds_iso"], args["ds_smag"]]

    # check input args
    if len(args["h_resolution"]) == 1:
        args["h_resolution"] = [args["h_resolution"][0]] * len(args["input_files"])
    else:
        assert len(args["h_resolution"]) == len(args["input_files"])

    # initial validation
    assert (
        args["plot_show"] or args["plot_path"]
    ), "require at least one of 'plot_show' or  'plot_path'"

    # parse plotting style
    args["plot_map"] = plotting_styles

    # parse negative values in the [t,z]_range
    args["times"] = array(args["times"])
    assert all(args["times"] >= 0)

    if args["z_range"][0] < 0:
        args["z_range"][0] = -inf
    if args["z_range"][1] < 0:
        args["z_range"][1] = inf
    assert all(
        [
            args["z_range"][0] <= z <= args["z_range"][1]
            for z in args["hor_slice_levels"]
        ]
    ), f"hor_slice_levels {args['hor_slice_levels']} aren't contained in z_range {args['z_range']}"
    return args


def add_offline_fields(
    ds: xr.Dataset,
) -> list[xr.Dataset, Dict[str, field_plot_kwargs]]:
    # add offline fields
    offline_field_map = {}

    # extract cs from mixing length
    try:
        delta_x = ds["x_centre"].diff("x_centre")
        assert (
            delta_x[0] == delta_x[1:]
        ).all(), "non-homogeneous horizontal resolution"
        ds["cs"] = ds["csDelta"] / delta_x[0].item()
    except KeyError:
        print("Can't add cs with missing ingredients")

    # anisotropic mixing lengths
    try:
        ds["cs_h"] = ds["cs"]
        offline_field_map["cs_h"] = offline_field_map["cs"]
    except:
        print("Can't add offline 'cs_h' in dataset")

    # NOTE: these two rel on ds being pre-interpolated to a single grid (no stagerring)
    # because the *_vert viscosities are natively on the rho-grid
    try:
        # sqrt(c_v^2 Delta^2 |S|  / c_h^2 Delta^2 |S| )*c_h
        ds["cs_v"] = sqrt(ds["smag_visc_m_vert"] / ds["smag_visc_m"]) * ds["cs_h"]
        offline_field_map["cs_v"] = offline_field_map["cs"]
    except:
        print("Can't add offline 'cs_v' in dataset")

    try:
        # sqrt(c_v^2 Delta^2 |S| Pr / c_h^2 Delta^2 |S| Pr )*c_h
        ds["cs_theta_v"] = sqrt(ds["smag_visc_h_vert"] / ds["smag_visc_h"]) * ds["cs_h"]
        offline_field_map["cs_theta_v"] = offline_field_map["cs_theta"]
    except:
        print("Can't add offline 'cs_theta_v' in dataset")

    # # diagonal cs
    # try:
    #   ds["cs_diag"] = xr.DataArray([ds[f"cs_1"], ds[f"cs_2"], ds[f"cs_3"]], dims = 'c1',
    #                               coords= {'c1':[1,2,3]})
    # except KeyError:
    #     print("Can't add cs_diag with missing ingredients")

    # # diagonal cs
    # try:
    #   ds["cs_theta_diag"] = xr.DataArray([ds[f"cs_theta_1"], ds[f"cs_theta_2"], ds[f"cs_theta_3"]], dims = 'c1',
    #                               coords= {'c1':[1,2,3]})
    # except KeyError:
    #     print("Can't add cs_diag with missing ingredients")
    return ds, offline_field_map


def preprocess_dataset(
    ds: xr.Dataset, args: ArgumentParser
) -> list[xr.Dataset, Dict[str, field_plot_kwargs]]:
    """preprocess data:
       fix coordinates, take time/z constraints, add offline fields
    :param ds: xarray dataset to be modified:
    :param args: ArgumentParser from which to extract parametes;
                 Will replace with kwards, once the dust is settled!
    """

    # add offline field calculators
    # Note: do this first to protect non-local offline fields
    # shouldn't be a memory burden for lazy caculations.
    ds, offline_field_map = add_offline_fields(ds)

    # drop initial conditions to unify time dimension
    if "t_0" in ds:
        if "t" in ds:
            ds = ds.sel(t_0=ds["t"].data).drop_vars("t_0")
        ds = ds.rename({"t_0": "t"})

    # take z-range
    for z in "z", "z_rho", "z_theta", "z_face", "z_centre", "thlev_bl_zsea_theta":
        if z in ds:
            zslice = (args["z_range"][0] <= ds[z]) * (ds[z] <= args["z_range"][1])
            ds = ds.sel({z: zslice})

    # take time slice
    if args["times"].size == 0:
        times = arange(0, ds["t"].max(), 60)
    else:
        times = args["times"]
    ds = ds.sel({"t": times}, method="nearest").drop_duplicates(dim="t")

    # drop top level to align tke to z_theta
    if "thlev_bl_zsea_theta" in ds:
        if "z_theta" in ds:
            ds = ds.sel(z_theta=ds["thlev_bl_zsea_theta"].data).drop_vars("z_theta")
            ds = ds.rename({"thlev_bl_zsea_theta": "z_theta"})

    return ds, offline_field_map


def io(args) -> None:
    # read UM stasth files: data
    ds_collection: Mapping[str, xr.Dataset] = {}
    with timer("Read Dataset", "s"):
        for f, res, plot_map in zip(
            args["input_files"], args["h_resolution"], args["plot_map"]
        ):
            print(f'{plot_map["label"]}: {f}')
            ds = data_ingest_UM(
                f,
                res,
                requested_fields=slice_fields + prof_fields,
            )
            if len(ds) == 0:
                continue
            # preprocess data:
            #     fix coordinates, take time/z constraints, add offline fields
            ds, offline_field_map = preprocess_dataset(ds, args)

            # add chunking for better memory management
            # will not chunk along z because staggering makes it annoying
            # for simple plotting it should be ok
            ds = ds.chunk({"t": args["t_chunk_size"]})

            # store with a label from the plotting map
            ds_collection[plot_map["label"]] = ds

        field_plot_map.update(debug_field_plot_map)
        field_plot_map.update(offline_field_map)
        field_plot_map.update(anisotropic_plot_map)

    return ds_collection, field_plot_map


def vert_profile_reduction(
    da: xr.DataArray,
    reduction: Callable | str,
    reduction_dims: Iterable[str],
) -> xr.DataArray:
    if reduction == "mean":
        data = da.mean(reduction_dims).squeeze()
    elif reduction == "var":
        data = da.var(reduction_dims).squeeze()
    elif reduction == "std":
        data = da.std(reduction_dims).squeeze()
    elif reduction == "median":
        data = da.median(reduction_dims).squeeze()
    else:
        data = reduction(da, reduction_dims).squeeze()
    return data


def plot1(ds_diag: xr.Dataset, ds_iso: xr.DataArray, ds_smag: xr.DataArray, plot_map):
    # data reduction
    da_collection = {}
    linecolor = {}
    linestyle = {}
    # cs1, cs2, cs3
    for comp in range(1, 4):
        field = f"cs_{comp}"
        da_collection[field] = vert_profile_reduction(
            ds_diag[field], "mean", plot_map[field].hcoords
        )
        linecolor[field] = f"C{comp}"
        linestyle[field] = "-"
    # isotropic cs
    field = "cs"
    da_collection["cs_iso"] = vert_profile_reduction(
        ds_iso[field], "mean", plot_map[field].hcoords
    )
    plot_map["cs_iso"] = plot_map["cs"].with_args(label=r"$Cs_{iso}$")
    linecolor["cs_iso"] = "k"
    linestyle["cs_iso"] = ":"
    # Smagorinsky reference cs
    field = "cs"
    da_collection["cs_smag"] = vert_profile_reduction(
        ds_smag[field], "mean", plot_map[field].hcoords
    )
    plot_map["cs_smag"] = plot_map["cs"].with_args(label=r"$Cs_{Smag}$")
    linecolor["cs_smag"] = "k"
    linestyle["cs_smag"] = "--"
    # plot
    tcoord = "t"
    times = None
    for k in da_collection:
        if times is not None:
            assert np.allclose(times, da_collection[k][tcoord])
        else:
            times = da_collection[k][tcoord].data
        assert len(da_collection[k].dims) == 2, f"Too many dimensions in dataarray {k}"

    # num_sims = len(da_collection)
    fig, axes = plt.subplots(1, len(times), figsize=(6 * len(times), 4), sharey=False)

    for time, ax in zip(times, axes):
        for field, da in da_collection.items():
            local_time = {tcoord: da[tcoord].isin(time)}
            data = da.sel(local_time).squeeze()
            if data.size > 0:
                data.plot(
                    ax=ax,
                    y=plot_map[field].zcoord,
                    linestyle=linestyle[field],
                    color=linecolor[field],
                    label=plot_map[field].label,
                )
        ax.legend()
        ax.set_xlabel("Cs", fontsize=14)
        ax.set_title(f"time: {time.item()/60} h", fontsize=14)
        ax.set_xlim(None, 0.5)
    return fig


def plot(
    ds_collection: Mapping[str, xr.Dataset],
    args: ArgumentParser,
    field_plot_map,
) -> None:
    """master plotting routine"""

    ureg = UnitRegistry()  # use to parse resolution
    ds = next(iter(ds_collection.values()))
    tunit = ds["t"].unit
    tmin = ds["t"].min().item() * ureg(tunit)
    tmax = ds["t"].max().item() * ureg(tunit)
    tlabel = f"times{tmin.to('h').magnitude:0g}-{tmax.to('h').magnitude:0g}h"
    if args["plot_path"] is not None:
        args["plot_path"].mkdir(parents=True, exist_ok=True)

    # plot1
    with timer("Plot 1 profiles", "s"):
        try:
            plot1_fig = plot1(
                ds_collection["ds_diag"],
                ds_collection["ds_iso"],
                ds_collection["ds_smag"],
                field_plot_map,
            )
            if args["plot_path"] is not None:
                plot1_fig.savefig(
                    args["plot_path"] / f"Profile_{tlabel}_mean_Cs_comparison.png",
                    dpi=180,
                )
        except KeyboardInterrupt:
            print("Detected Keyboard interrup, proceeding with vertical profiles")

    # interactive plotting out of time
    if args["plot_show"]:
        plt.show()
    plt.close()


if __name__ == "__main__":
    with timer("Total execution time", "min"):
        with timer("Arguments", "ms"):
            args = parse_args()
        verbose = args["verbose"]
        ds_collection, field_plot_map = io(args)
        for ds in ds_collection:
            print(ds, sorted(ds_collection[ds]))
        # make plots
        with timer("Make plots", "s"):
            plot(ds_collection, args, field_plot_map)

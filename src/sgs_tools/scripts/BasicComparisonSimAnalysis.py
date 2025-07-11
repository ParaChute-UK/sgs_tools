import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any, Callable, Collection, Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure
from numpy import arange, array, inf, ndarray
from pint import UnitRegistry  # type: ignore
from sgs_tools.io.um import data_ingest_UM
from sgs_tools.physics.fields import Reynolds_fluct_stress, vertical_heat_flux
from sgs_tools.plotting.collection_plots import (
    plot_horizontal_slice_tseries,
    plot_vertical_prof_time_slice_compare_sims_slice,
)
from sgs_tools.plotting.field_plot_map import (
    debug_field_plot_map,
    field_plot_kwargs,
    field_plot_map,
)
from sgs_tools.scripts.arg_parsers import add_dask_group, add_plotting_group
from sgs_tools.util.timer import timer

default_plotting_style = {
    "label": "sim1",
    "linestyle": "-",
    "color": "k",
    "linewidth": 1,
    "marker": "",
}

slice_fields = (
    "u",
    "v",
    "w",
    "theta",
    "s",
    "cs",
    "cs_theta",
    # debug
    "s2d",
    "s4d",
    "lm",
    "mm",
    "qn",
    "nn",
)
prof_fields = (
    "u",
    "v",
    "w",
    "w^2",
    "u'w'",
    "v'w'",
    "u'v'",
    "tke",
    "csDelta",
    "cs",
    "smag_visc_m",
    "theta",
    "vertical_heat_flux",
    "smag_visc_h",
    "cs_theta",
    "Richardson",
    "Richardson_diag",
    "q_t",
    # debug
    "s",
    "s2d",
    "s4d",
    "lm",
    "mm",
    "qn",
    "nn",
    # annisotropics
)

# prof_fields = ('cs1', 'cs2', 'cs3',
#                'cs_theta_1', 'cs_theta_2', 'cs_theta_3')
# slice_fields = prof_fields

cloud_fields = ("q_l", "q_i", "q_g")

verbose = False


def parse_args() -> dict[str, Any]:
    parser = ArgumentParser(
        description="""Create (and optionally save) standard diagnostic plots for
                    an atmospheric boundary layer UM simulation
                    Best-suited to one-parameter suite of simulations,
                    but can handle several varying parameters through plot_style_file
                """,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    fname = parser.add_argument_group("I/O datasets on disk")
    fname.add_argument(
        "input_files",
        type=Path,
        nargs="+",
        help=""" Location of a set simulation outputs -- UM NetCDF diagnostic files.
            Recognizes glob patterns and walks directory trees, e.g. './my_file_p[br]*nc'
            Can have multiple files per simulation, but only one glob pattern per simulation.
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
        "--plot_style_file",
        type=Path,
        default=None,
        help="""Configuration file describing a list of plot style and decorations to matched sequentially to each simulation.
                See plot_config_template.json for a template.
                If absent, will use ``default_plotting_style`` and cycle through different colors.
            """,
    )

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
    if args["plot_style_file"] is None:
        plot_styles = [default_plotting_style] * len(args["input_files"])
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, _ in enumerate(plot_styles):
            plot_styles[i]["color"] = colors[i % len(colors)]
            plot_styles[i]["label"] = f"sim{i}"
    else:
        with open(args["plot_style_file"]) as f:
            plot_styles = json.load(f)
        # ensure we have enough plotting styles
        assert len(plot_styles) >= len(args["input_files"])

    args["plot_map"] = plot_styles

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


def preprocess_dataset(
    ds: xr.Dataset, args: dict[str, Any]
) -> tuple[xr.Dataset, Dict[str, field_plot_kwargs]]:
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
    times: ndarray
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


def add_offline_fields(
    ds: xr.Dataset,
) -> tuple[xr.Dataset, Dict[str, field_plot_kwargs]]:
    # add offline fields
    offline_field_map = {}
    # velocity squared
    try:
        ds["w^2"] = ds["w"] ** 2
        ds["w^2"].attrs.update({"units": "m^2 s-2", "long_name": r"$w^2$"})
        offline_field_map["w^2"] = field_plot_kwargs(
            ds["w^2"].attrs["long_name"],
            "t",
            "z_theta",
            ("x_centre", "y_centre"),
            (None, None),
            "Oranges",
        )
    except KeyError:
        print("Can't add w^2 with missing ingredients")

    # vertical heat flux
    try:
        ds["vertical_heat_flux"] = vertical_heat_flux(
            ds["w"], ds["theta"], ["x_centre", "y_centre"]
        )
        ds["vertical_heat_flux"].attrs.update({"units": "K m s-1"})

        offline_field_map["vertical_heat_flux"] = field_plot_kwargs(
            ds["vertical_heat_flux"].attrs["long_name"],
            "t",
            "z_theta",
            ("x_centre", "y_centre"),
            (None, None),
            "Oranges",
        )
    except KeyError:
        print("Can't add vertical_heat_flux with missing ingredients")

    # momentum flux ~ Reynolds stress
    try:
        tau = Reynolds_fluct_stress(
            ds["u"],
            ds["v"],
            ds["w"],
            ["x_centre", "y_centre", "z_theta"],
            ["x_centre", "y_centre"],
        )
        for x in tau.c1.data:
            for y in tau.c2.data:
                ds[f"{x}{y}"] = tau.sel(c1=x, c2=y)
                ds[f"{x}{y}"].attrs.update(
                    {"units": "m^2 s-2", "long_name": f"${x}{y}$"}
                )
                offline_field_map[f"{x}{y}"] = field_plot_kwargs(
                    ds[f"{x}{y}"].attrs["long_name"],
                    "t",
                    "z_theta",
                    ("x_centre", "y_centre"),
                    (None, None),
                    "RdBu_r",
                )

        ds[f"tke"] = 0.5 * (
            tau.isel(c1=0, c2=0) + tau.isel(c1=1, c2=1) + tau.isel(c1=2, c2=2)
        )
        ds[f"tke"].attrs.update({"units": "m^2 s-2", "long_name": r"$0.5 (u'_i u'_i)$"})
        offline_field_map[f"tke"] = field_plot_kwargs(
            ds[f"tke"].attrs["long_name"],
            "t",
            "z_theta",
            ("x_centre", "y_centre"),
            (None, None),
            "Oranges",
        )
    except KeyError:
        print("Can't add components of Reynolds stress with missing ingredients")

    # extract cs from mixing length
    try:
        delta_x = ds["x_centre"].diff("x_centre")
        assert (
            delta_x[0] == delta_x[1:]
        ).all(), "non-homogeneous horizontal resolution"
        ds["cs"] = ds["csDelta"] / delta_x[0].item()
    except KeyError:
        print("Can't add cs with missing ingredients")

    # moisture species and cumulative humidity
    flist = [ds[q] for q in ["q_l", "q_i", "q_g"] if q in ds]
    if flist:
        ds["q_t"] = flist[0]
        for f in flist[1:]:
            ds["q_t"] += f
            ds["q_t"].attrs.update({"long_name": "$q_t$"})
        offline_field_map["q_t"] = field_plot_kwargs(
            ds[f"q_t"].attrs["long_name"],
            "t",
            "z_theta",
            ("x_centre", "y_centre"),
            (None, None),
            "RdYlBu_r",
        )
    else:
        print("Can't add q_t with missing ingredients")

    return ds, offline_field_map


def vert_profile_reduction(
    da: xr.DataArray,
    reduction: Callable | str,
    reduction_dims: Collection[str],
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
        assert callable(reduction)
        data = reduction(da, reduction_dims).squeeze()
    return data


def plot_horiz_slices(
    ds_collection: Mapping[str, xr.Dataset],
    fields: Iterable[str],
    zlevels: Iterable[float],
    field_plot_map,
) -> Dict[float, Dict[str, Figure]]:
    hor_slice: Dict[float, Dict[str, Figure]] = {}
    for z in zlevels:
        hor_slice[z] = {}
        for field in fields:
            if verbose:
                print(f"Plotting {field} at {field_plot_map[field].zcoord} ~ {z}")
            da_collection = {
                k: ds[field].sel({field_plot_map[field].zcoord: z}, method="nearest")
                for k, ds in ds_collection.items()
                if field in ds
            }
            if da_collection:
                f = plot_horizontal_slice_tseries(
                    da_collection,
                    "t",
                    field_plot_map[field].cmap,
                    field_plot_map[field].label,
                    field_plot_map[field].zcoord,
                )
                f.suptitle(field_plot_map[field].label)
                f.tight_layout()
                hor_slice[z][field] = f.get_figure()
    return hor_slice


def plot_vert_profiles(
    ds_collection: Mapping[str, xr.Dataset],
    fields: Iterable[str],
    reductions: Iterable[str],
    plot_map,
) -> Dict[str, Dict[str, Figure]]:
    vert_prof: Dict[str, Dict[str, Figure]] = {}
    for reduction in reductions:
        vert_prof[reduction] = {}
        for field in fields:
            da_collection = {}
            if verbose:
                print(f"Plotting {reduction} of {field}")
            for k, ds in ds_collection.items():
                try:
                    da_collection[k] = vert_profile_reduction(
                        ds[field], reduction, field_plot_map[field].hcoords
                    )
                except:
                    print(f"Skip missing field {field} from sim {k}")
            if da_collection:
                q = plot_vertical_prof_time_slice_compare_sims_slice(
                    da_collection,
                    plot_map,
                    f"{reduction} [ {field_plot_map[field].label} ]",
                    "t",
                    field_plot_map[field].zcoord,
                )
                # q.suptitle(field_plot_map[field].label)
                q.tight_layout()
                vert_prof[reduction][field] = q.get_figure()

    return vert_prof


def plot_clouds(
    ds_collection: Mapping[str, xr.Dataset],
    clevels: Iterable[float],
    field_plot_map,
    collection_plot_map,
) -> Figure | None:
    fig, _ = plt.subplots(len(ds_collection), 1, figsize=(6, len(ds_collection) * 6))
    axes = fig.axes
    empty = True
    for ax, k in zip(axes, ds_collection):
        if "q_t" in ds_collection[k]:
            data = ds_collection[k]["q_t"].mean(field_plot_map["q_t"].hcoords) * 1000
            data.plot.contourf(
                ax=ax,
                y=field_plot_map["q_t"].zcoord,
                x=field_plot_map["q_t"].tcoord,
                levels=clevels,
                robust=True,
                cmap=field_plot_map["q_t"].cmap,
                extend="max",
                add_colorbar=True,
            )
            ax.text(
                0.01,
                0.99,
                collection_plot_map["label_map"][k],
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=24,
            )
            # ax.tick_params(axis="x", labelsize=16)
            # ax.tick_params(axis="y", labelsize=16)
            empty = False
    if not empty:
        fig.tight_layout()
        return fig
    else:
        return None


def plot(
    ds_collection: Mapping[str, xr.Dataset],
    args: dict[str, Any],
    slice_fields: Iterable[str],
    prof_fields: Iterable[str],
    field_plot_map,
) -> None:
    """master plotting routine"""
    # plot horizontal slices
    with timer("Plot horizontal slices", "s"):
        try:
            hor_slice = plot_horiz_slices(
                ds_collection,
                slice_fields,
                args["hor_slice_levels"],
                field_plot_map,
            )
        except KeyboardInterrupt:
            print("Detected Keyboard interrup, proceeding with vertical profiles")
    # plot vertical profiles
    # transpose plot map and match to dataset labels
    plot_map: Dict[str, Dict[str, Any]] = {
        "color_map": {},
        "linestyle_map": {},
        "linewidth_map": {},
        "marker_map": {},
        "label_map": {},
    }
    for i, key in enumerate(ds_collection):
        plot_map["color_map"][key] = args["plot_map"][i]["color"]
        plot_map["linestyle_map"][key] = args["plot_map"][i]["linestyle"]
        plot_map["linewidth_map"][key] = args["plot_map"][i]["linewidth"]
        plot_map["marker_map"][key] = args["plot_map"][i]["marker"]
        plot_map["label_map"][key] = args["plot_map"][i]["label"]

    reductions = ("mean", "var")
    with timer("Plot vertical profiles", "s"):
        try:
            if args["skip_vert_profiles"]:
                prof_fields = []
            vert_prof = plot_vert_profiles(
                ds_collection, prof_fields, reductions, plot_map
            )
        except KeyboardInterrupt:
            print("Detected Keyboard interrup, proceeding with cloud plot.")

    # cloud plots
    cloud_fig = plot_clouds(
        ds_collection, arange(0.005, 0.15, 0.005), field_plot_map, plot_map
    )

    # save plots to disk
    if args["plot_path"] is not None:
        print(f"Saving plots to {args['plot_path']}")
        # parse time range; assume all datasets use the same time scale
        ureg = UnitRegistry()  # use to parse resolution
        ds = next(iter(ds_collection.values()))
        tunit = ds["t"].unit
        tmin = ds["t"].min().item() * ureg(tunit)
        tmax = ds["t"].max().item() * ureg(tunit)
        tlabel = f"times{tmin.to('h').magnitude:0g}-{tmax.to('h').magnitude:0g}h"
        args["plot_path"].mkdir(parents=True, exist_ok=True)
        for z in hor_slice:
            for f in hor_slice[z]:
                hor_slice[z][f].savefig(
                    args["plot_path"] / f"Slice_z{z:g}m_{tlabel}_{f}.png", dpi=180
                )
        for red in vert_prof:
            for f in vert_prof[red]:
                vert_prof[red][f].savefig(
                    args["plot_path"] / f"Profile_{tlabel}_{red}_{f}.png", dpi=180
                )
        if cloud_fig:
            cloud_fig.savefig(args["plot_path"] / f"Clouds_CL_{tlabel}.png", dpi=180)
    # interactive plotting out of time
    if args["plot_show"]:
        plt.show()
    plt.close()


def io(args) -> tuple[Dict[str, xr.Dataset], Dict[str, field_plot_kwargs]]:
    # read UM stasth files: data
    ds_collection: Dict[str, xr.Dataset] = {}
    with timer("Read Dataset", "s"):
        for f, res, plot_map in zip(
            args["input_files"], args["h_resolution"], args["plot_map"]
        ):
            print(f'{plot_map["label"]}: {f}')
            ds = data_ingest_UM(
                f,
                res,
                requested_fields=list(slice_fields + prof_fields + cloud_fields),
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

    return ds_collection, field_plot_map


def main():
    with timer("Total execution time", "min"):
        with timer("Arguments", "ms"):
            args = parse_args()
        verbose = args["verbose"]
        ds_collection, field_plot_map = io(args)

        # make plots
        with timer("Make plots", "s"):
            plot(ds_collection, args, slice_fields, prof_fields, field_plot_map)


if __name__ == "__main__":
    main()

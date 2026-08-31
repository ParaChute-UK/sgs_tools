from collections.abc import Collection, Iterable, Mapping
from typing import Any

import numpy as np
import xarray as xr
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def arrays_equal(a, b):
    """
    Robustly compare two arrays for equality, supporting numeric and non-numeric types.
    Uses allclose for numeric, array_equal otherwise.
    """
    # Convert to numpy arrays if needed
    a = np.asarray(a)
    b = np.asarray(b)
    # Check dtype kind
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return np.allclose(a, b, equal_nan=True)
    else:
        return np.array_equal(a, b)


def plot_vertical_prof_time_slice_compare_sims_slice(
    da_collection: Mapping[str, xr.DataArray],
    plot_kwargs: Mapping[str, Any],
    x_lbl: str,
    tcoord: str,
    zcoord: str,
    with_markers=False,
) -> Figure:
    """
    Plot a row of plots with a different time in each panel.
    Compare simulations from `da_collection` in each panel.

    :param da_collection: a dictionary of 2d xr.DataArrays to be plotted.
      Will use the keys to pick plotting style from plot_kwargs.
    :param plot_kwargs: a dictionary of plotting style parameters for each simulation.
    :param x_lbl: a display label for the plotted field, on the x-axis
    :param tcoord: name of time coordinate -- will generate one panel per time index
    :param zcoord: name of y-coordinate, leave as None for default
    :param with_markers: general flag to add markers to the plotted lines.
    """
    times = xr.DataArray([])
    for k in da_collection:
        if len(times) != 0:
            assert arrays_equal(times, da_collection[k][tcoord].data)
        else:
            times = da_collection[k][tcoord].data
        assert len(da_collection[k].dims) == 2, f"Too many dimensions in dataarray {k}"

    # attach backend-safe figure + canvas
    fig = Figure(figsize=(6 * len(times), 4))
    FigureCanvasAgg(fig)
    fig.subplots(1, len(times), sharey=False)
    axes = fig.axes
    for time, ax in zip(times, axes, strict=False):
        for k, da in da_collection.items():
            marker = plot_kwargs["marker_map"][k] if with_markers else None

            local_time = {tcoord: da[tcoord].isin(time)}
            data = da.sel(local_time).squeeze()
            if data.size > 0:
                data.plot(
                    ax=ax,
                    y=zcoord,
                    linestyle=plot_kwargs["linestyle_map"][k],
                    color=plot_kwargs["color_map"][k],
                    lw=plot_kwargs["linewidth_map"][k],
                    label=plot_kwargs["label_map"][k],
                    marker=marker,
                )  # type: ignore
            ax.legend()
            ax.set_xlabel(x_lbl, fontsize=14)
            ax.set_title(f"{tcoord}: {time.item()}", fontsize=14)
    return fig


def plot_horizontal_slice_tseries(
    da_collection: Mapping[str, xr.DataArray],
    tcoord: str,
    cmap: str,
    field_lbl: str,
    zcoord: str,
) -> Figure:
    """
    Plot a grid of horizontal slices in each panel
    each row corresponds to a different simulation
    each column corresponds to a different time.

    :param da_collection: a dictionary of 3d xr.DataArrays to be plotted.
      One of the dimensions must be `tcoord`
    :param tcoord: name of time coordinate -- will generate one column per time index
    :param cmap: colormap to use for plotting
    :param field_lbl: a display label for the plotted field
    """

    times = xr.DataArray([])
    for k in da_collection:
        if len(times) != 0:
            assert np.allclose(times, da_collection[k][tcoord])
        else:
            times = da_collection[k][tcoord].data
        assert len(da_collection[k].dims) == 3, f"Too many dimensions in dataarray {k}"

    num_times = len(times)
    num_sims = len(da_collection)
    # attach backend-safe figure + canvas
    fig = Figure(figsize=(6 * num_times, 4 * num_sims))
    FigureCanvasAgg(fig)
    axes = fig.subplots(
        num_sims,
        num_times,
        sharey=False,
        squeeze=False,
    )

    for i, (sim_lbl, da) in enumerate(da_collection.items()):
        for j, time in enumerate(times):
            local_time = {tcoord: da[tcoord].isin(time)}
            data = da.sel(local_time, method="nearest").squeeze()
            ax = axes[i][j]
            if data.size > 0:
                if ax is axes[i][-1]:
                    # with colorbar label
                    data.plot(
                        ax=ax,
                        y=data.dims[0],
                        cmap=cmap,
                        cbar_kwargs={"label": field_lbl},
                        robust=True,
                    )  # type: ignore
                else:
                    # no colorbar label
                    data.plot(
                        ax=ax, y=data.dims[0], cmap=cmap, cbar_kwargs={"label": None}
                    )  # type: ignore
            # ax.set_xlabel(, fontsize=14)
            ax.set_title(
                f"{sim_lbl}: z = {data[zcoord].item():g}m, time= {time / 60} h",
                fontsize=14,
            )
            if j > 0:
                ax.set_ylabel(None)
    fig.tight_layout()
    return fig


def plot_vertical_prof_time_slice_compare_fields(
    ds,
    fields: Iterable[str],
    reduction: str,
    zcoord: str,
    tslice: dict[str, Collection] | None = None,
    field_lbls: list[str] = [""] * 20,
    les_reference=None,
    zmax=1e6,
    ds_label="",
) -> Figure:
    """
    Plot a row of plots with a time slice in each panel.
    Compare fields in each panel.

    tslice : selection of times in minutes. if None will default
    to hourly schedule from 1 to 16 hours.
    reduction: 'mean' or 'median'
    """
    if tslice is None:
        tslice = {"t": np.arange(1, 16) * 60}
    times = list(tslice.values())[0]
    # attach backend-safe figure + canvas
    fig = Figure(figsize=(6 * len(times), 5))
    FigureCanvasAgg(fig)
    fig.subplots(1, len(times), sharey=False)
    axes = fig.axes

    tcoord = list(tslice.keys())[0]

    for time, ax in zip(times, axes, strict=False):
        # if les_reference is not None and reduction == "mean":
        #     k = "monc_les"
        #     les_reference.sel(time_series_60_60=time).plot(
        #         ax=ax,
        #         y="zn",
        #         ls=linestyle_map[k],
        #         color=color_map[k],
        #         lw=linewidth_map[k],
        #         label="mean " + label_map[k],
        #     )
        for i, field in enumerate(fields):
            local_time = {tcoord: ds[tcoord].isin(time)}

            reduction_dims = [x for x in ds[field].dims if x not in [zcoord, tcoord]]
            if reduction == "mean":
                data = (
                    ds[field]
                    .sel(local_time)
                    .mean(reduction_dims, skipna=True)
                    .squeeze()
                )
            elif reduction == "var":
                data = (
                    ds[field].sel(local_time).var(reduction_dims, skipna=True).squeeze()
                )
            elif reduction == "median":
                data = (
                    ds[field]
                    .sel(local_time)
                    .median(reduction_dims, skipna=True)
                    .squeeze()
                )
            else:
                raise ValueError(
                    f"Unrecognised reduction {reduction}, choose 'mean' or 'median'"
                )

            if data.size > 0:
                z = (data.dims)[0]
                data = data.where(data[z] < zmax, drop=True)
                data.plot(
                    ax=ax,
                    y=z,
                    linestyle="-",
                    color=f"C{i}",
                    label=field_lbls[i],
                )
        ax.legend()
        ax.set_xlabel("", fontsize=14)
        ax.set_title(f"time: {time / 60} h", fontsize=14)
    return fig


def plot_clouds(
    ds_collection: Mapping[str, xr.Dataset],
    clevels: Iterable[float],
    field_plot_map,
    collection_plot_map,
) -> Figure | None:
    # attach backend-safe figure + canvas
    fig = Figure(figsize=(6, len(ds_collection) * 6))
    FigureCanvasAgg(fig)
    fig.subplots(len(ds_collection), 1)
    axes = fig.axes
    empty = True
    for ax, k in zip(axes, ds_collection, strict=False):
        if "q_t" in ds_collection[k]:
            data = (
                ds_collection[k]["q_t"].mean(field_plot_map["q_t"].hcoords) * 1000
            ).compute()
            if len(field_plot_map["q_t"].tcoord) > 1:
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
            else:
                data.plot(
                    ax=ax,
                    y=field_plot_map["q_t"].zcoord,
                )  # type: ignore
            # ax.tick_params(axis="x", labelsize=16)
            # ax.tick_params(axis="y", labelsize=16)
            empty = False
    if not empty:
        fig.tight_layout()
        return fig

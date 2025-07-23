from typing import Any, Collection, Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_vertical_prof_time_slice_compare_sims_slice(
    da_collection: Mapping[str, xr.DataArray],
    plot_kwargs: Mapping[str, Any],
    x_lbl: str,
    tcoord: str,
    zcoord: str,
    with_markers=False,
):
    """plot a row of plots with a different time in each panel.
    Compare simulations from `da_collection` in each panel.
    :param da_collection: a dictionary of 2d xr.DataArrays to be plotted. will use the keys
                          to pick plotting style from plot_kwargs.
    :param plot_kwargs: a dictionary of plotting style parameters for each simulation.
    :param x_lbl: a display label for the plotted field, on the x-axis
    :param tcoord: name of time coordinate -- will generate one panel per time index
    :param zcoord: name of y-coordinate, leave as None for default
    :param with_markers: general flag to add markers to the plotted lines.
    """
    times = xr.DataArray([])
    for k in da_collection:
        if len(times) != 0:
            assert np.allclose(times, da_collection[k][tcoord])
        else:
            times = da_collection[k][tcoord].data
        assert len(da_collection[k].dims) == 2, f"Too many dimensions in dataarray {k}"

    # num_sims = len(da_collection)
    fig, _ = plt.subplots(1, len(times), figsize=(6 * len(times), 4), sharey=False)
    axes = fig.axes
    for time, ax in zip(times, axes):
        for k, da in da_collection.items():
            if with_markers:
                marker = plot_kwargs["marker_map"][k]
            else:
                marker = None

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
            ax.set_title(f"time: {time.item() / 60} h", fontsize=14)
    return fig


def plot_horizontal_slice_tseries(
    da_collection: Mapping[str, xr.DataArray],
    tcoord: str,
    cmap: str,
    field_lbl: str,
    zcoord: str,
):
    """plot a grid of horizontal slices in each panel
     each row corresponds to a different simulation
     each column corresponds to a different time.

    :param da_collection: a dictionary of 3d xr.DataArrays to be plotted. One of the dimensions must be `tcoord`
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
    fig, axes = plt.subplots(
        num_sims,
        num_times,
        figsize=(6 * num_times, 4 * num_sims),
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
    tslice: Dict[str, Collection] = {"t": np.arange(1, 16) * 60},
    field_lbls: list[str] = [""] * 20,
    les_reference=None,
    zmax=1e6,
    ds_label="",
):
    """plot a row of plots with a time slice in each panel.
    Compare fields in each panel.
    tslice : selection of times in minutes.
    reduction: 'mean' or 'median'
    """
    times = list(tslice.values())[0]
    fig, axes = plt.subplots(1, len(times), figsize=(6 * len(times), 5), sharey=False)

    tcoord = list(tslice.keys())[0]

    for time, ax in zip(times, axes):
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

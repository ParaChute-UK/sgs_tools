import matplotlib.pyplot as plt
import numpy as np


# Plotting
def tensor_plot(tens, grid_dims=["c1", "c2"], robust=True, **plot_kwargs):
    """plot 3x3 tensor 2-d field with coordinates c1&c2 on 3x3 panels"""
    fig, ax = plt.subplots(
        tens[grid_dims[1]].size,
        tens[grid_dims[0]].size,
        figsize=(tens[grid_dims[0]].size * 5.3, tens[grid_dims[1]].size * 5),
    )
    for i in range(tens[grid_dims[0]].size):
        for j in range(tens[grid_dims[1]].size):
            tens.isel({grid_dims[0]: i, grid_dims[1]: j}).plot(
                ax=ax[j][i], robust=robust, **plot_kwargs
            )
            ax[j][i].set_title(
                f"{grid_dims[0]} = {tens[grid_dims[0]][i].data}, "
                f"{grid_dims[1]} = {tens[grid_dims[1]][j].data}"
            )
    fig.tight_layout()
    return fig


def comparison_contour_plot(sca, sca_ref, name="test"):
    """single figure comparing two 2d scalars"""
    fig, ax = plt.subplots(2, 2, figsize=(13, 13))
    sca.plot(robust=True, ax=ax[0, 0])
    sca_ref.plot(robust=True, ax=ax[0, 1])
    (sca - sca_ref).plot(robust=True, ax=ax[1, 0])
    (sca / sca_ref - 1).plot(robust=True, ax=ax[1, 1])
    ax[0][0].set_title(name)
    ax[0][1].set_title(name + " reference")
    ax[1][0].set_title("difference")
    ax[1][1].set_title("rel difference")
    return fig


def comparison_line_plot(sca, sca_ref, name="test"):
    """single figure comparing two curves"""
    fig, ax = plt.subplots(2, 2, figsize=(13, 13))
    sca.plot(ax=ax[0, 0])
    sca_ref.plot(ax=ax[0, 1])
    (sca - sca_ref).plot(ax=ax[1, 0])
    (sca / sca_ref - 1).plot(ax=ax[1, 1])
    ax[0][0].set_title(name)
    ax[0][1].set_title(name + " reference")
    ax[1][0].set_title("difference")
    ax[1][1].set_title("rel difference")

    return fig


### plot decorators
def get_tdim_lbl(arr, tlvl):
    clbl = [t for t in arr.coords if t[:6] == "min15T"][0]
    if arr[clbl].shape == ():
        return f"{arr[clbl].item()}"
    else:
        return f"{arr[clbl][tlvl].item()}"


def get_zdim_lbl(arr, tlvl):
    clbl = [t for t in arr.coords if t[:2] == "z_"][0]
    if arr[clbl].shape == ():
        return f'z = {arr[clbl].item():g} {arr[clbl].attrs.get("units", "")}'
    else:
        return f'z = {arr[clbl][tlvl].item():g} {arr[clbl].attrs.get("units", "")}'


## Higher-order plots
def direct_comparison_field_plot(
    collection, labels=None, suptitle="", vmin=None, vmax=None, cmap=None
):
    """collection :: list of 2-d Xarray datasets
    suptitle: optional super-title of figure
    returns figure with 3 panels : reference, value and relative difference
    """
    fig, _ = plt.subplots(1, len(collection), figsize=(5.15 * len(collection), 5))
    ax = fig.axes
    if vmin is None:
        vmin = [None] * len(collection)
    if vmax is None:
        vmax = [None] * len(collection)
    if isinstance(cmap, str) or not hasattr(cmap, "__len__") or len(cmap) == 1:
        cmap = [cmap] * len(collection)

    for i, ds in enumerate(collection):
        # try:
        ydim = None
        for d in ds.dims:
            if "z_" in d:
                ydim = d
        ds.plot(ax=ax[i], robust=True, y=ydim, vmin=vmin[i], vmax=vmax[i], cmap=cmap[i])
        # except:
        #     pass
    if labels is not None:
        for i, label in enumerate(labels):
            ax[i].set_title(label)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def comparison_field_plot(
    collection,
    zslice,
    tslice,
    labels=None,
    suptitle="",
    vmin=None,
    vmax=None,
    cmap=None,
):
    """collection :: list of 4-d Xarray datasets
    field :: string, name of field to be plotted
    zslice, tslice :: dictionaries {'coord_name' : coord_index}
    suptitle: optional super-title of figure
    returns figure with 3 panels : reference, value and relative difference
    """
    reduced_fields = [ds.isel(tslice).isel(zslice) for ds in collection]
    fig = direct_comparison_field_plot(
        reduced_fields, labels, suptitle, vmin, vmax, cmap
    )
    return fig


def comparison_profile_plot(
    datasets,
    reduce_dims,
    take_slice,
    plot_coord,
    plot_field,
    height_range=slice(None),
    labels=None,
    ls=None,
    colors=None,
    axes=None,
):
    """datasets :: iterable of Xarray datasets, containing the plot_field as a 4-d DataArray
    reduce_coords :: iterable of dimesions to be collapsed (mean, std, ect.)
    take_slice :: dictionary of {'coord_name' : coord_index} to be sliced from each dataset
    plot_coord: string, name of remaining coordinated to be plotted against.
    plot_field: string, name of field to be plotted, must be present in all datasets
    height_range: optional, cutoff y-axis, any format accepted by the [height_range] syntax
    axes: optional, an iterable of matplotlib Axis objects to plot on, if absent, create own figure
    returns figure with 2 panels: comparing mean and std profiles of the sliced datasets
    """

    if labels is None:
        labels = [f"arr{i}" for i in range(len(datasets))]
    if colors is None:
        colors = ["k" for i in range(len(datasets))]

    if ls is None:
        ls = ["solid", "dashed", "dotted"]
    else:
        ls = len(colors) * ls

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    else:
        assert len(axes) == 2
    for i, ds in enumerate(datasets):
        # try:
        ds[plot_field].isel(take_slice)[height_range].mean(reduce_dims).plot(
            ax=axes[0],
            y=plot_coord,
            label=labels[i],
            color=colors[i % len(colors)],
            ls=ls[i % len(ls)],
        )
        ds[plot_field].isel(take_slice)[height_range].std(reduce_dims).plot(
            ax=axes[1],
            y=plot_coord,
            label=labels[i],
            color=colors[i % len(colors)],
            ls=ls[i % len(ls)],
        )
    # except:
    #     pass
    title = []
    print(take_slice.keys())
    for k in take_slice.keys():
        if "t" in k:
            print("t", take_slice[k])
            title.append(get_tdim_lbl(ds[plot_field], take_slice[k]))
        if "z" in k:
            print("z", take_slice[k])
            title.append(get_zdim_lbl(ds[plot_field], take_slice[k]))
    if title:
        axes[0].set_title(",".join(title))
        axes[1].set_title(",".join(title))

    axes[0].legend()
    axes[0].set_xlabel(f"mean {plot_field}")

    axes[1].legend()
    axes[1].set_ylabel(None)
    axes[1].set_xlabel(f"std {plot_field}")

    axes[0].get_figure().tight_layout()
    return axes[0].get_figure()


def error_profile_plot(
    f1,
    f2,
    tslice,
    height_range,
    red_coords=["x_centre", "y_centre"],
    ycoord="z_theta",
    axes=None,
):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    else:
        assert len(axes) == 2
    dic = tslice.copy()
    dic.update(height_range)
    (f1.isel(dic) / f2.isel(dic) - 1).mean(red_coords).plot(
        y=ycoord, label="mean relative difference", ax=axes[0]
    )
    np.abs(f1.isel(dic) - f2.isel(dic)).mean(red_coords).plot(
        y=ycoord, label="mean absolute difference", ax=axes[1]
    )
    axes[0].legend()
    axes[1].legend()
    axes[0].get_figure().tight_layout()
    return axes[0].get_figure()

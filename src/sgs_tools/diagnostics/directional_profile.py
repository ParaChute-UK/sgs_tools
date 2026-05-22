from collections.abc import Callable, Sequence

import xarray as xr
from xarray.core.types import T_Xarray

from sgs_tools.util.dask_adapt_chunking import chunk_ds


def profile_reduction(
    da: T_Xarray,
    reduction: Callable | str,
    reduction_dims: Sequence[str],
) -> T_Xarray:
    if reduction == "mean":
        data = da.mean(reduction_dims)
    elif reduction == "var":
        data = da.var(reduction_dims)
    elif reduction == "std":
        data = da.std(reduction_dims)
    elif reduction == "median":
        data = da.median(reduction_dims)
    elif reduction == "rms":
        data = ((da**2).sum(reduction_dims)) ** 0.5
    elif callable(reduction):
        data = reduction(da, reduction_dims)
    else:
        raise ValueError(f"Unrecognised reduction {reduction}.")
    return data


def directional_profile(
    simulation: xr.Dataset,
    red_dims: Sequence[str],
    stats: Sequence[str] = ["mean", "std"],
) -> xr.Dataset:
    s = chunk_ds(simulation, dict.fromkeys(red_dims, -1))
    prof = []
    for stat in stats:
        prof.append(profile_reduction(s, stat, red_dims))
    # prof = dask.persist(*prof)
    profile = xr.concat(prof, dim=xr.DataArray(stats, dims=["statistic"]))
    return profile

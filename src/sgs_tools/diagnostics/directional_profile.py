from typing import Sequence

import xarray as xr


def directional_profile(simulation: xr.Dataset, red_dims: Sequence[str]) -> xr.Dataset:
    s = simulation.chunk({x: -1 for x in red_dims})
    mean = s.mean(dim=red_dims)
    std = s.std(dim=red_dims)
    # mean, std = dask.persist(mean, std)
    profile = xr.concat(
        [mean, std], dim=xr.DataArray(["mean", "std"], dims=["statistic"])
    )
    return profile

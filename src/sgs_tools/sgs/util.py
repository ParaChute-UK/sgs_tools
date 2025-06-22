from typing import Hashable

import xarray as xr


# check that arr is uniform along `filter_dims` with spacing of `dx`
def _assert_coord_dx(filter_dims: list[Hashable], arr: xr.DataArray, dx: float) -> None:
    for c in filter_dims:
        assert (arr[c].diff(dim=c) == dx).all(), f"Not uniform dimension {c}: {arr[c]}"

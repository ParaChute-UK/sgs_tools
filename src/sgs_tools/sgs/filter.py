from dataclasses import dataclass
from typing import Hashable, Sequence

import dask.array as da
import numpy as np
import xarray as xr

# Filter kernels

#: 3x3 2d Gaussian filter -- binomial approximation
weight_gauss_3d = xr.DataArray(
    da.from_array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], chunks={}) / 16.0,
    dims=["w1", "w2"],
)

#: 5x5 2d Gaussian filter -- binomial approximation
weight_gauss_5d = (
    xr.DataArray(
        da.from_array(
            [
                [1, 2, 2, 2, 1],
                [2, 4, 4, 4, 2],
                [2, 4, 4, 4, 2],
                [2, 4, 4, 4, 2],
                [1, 2, 2, 2, 1],
            ],
            chunks={},
        ),
        dims=["w1", "w2"],
    )
    / 64
)


def box_kernel(shape: list[int]) -> xr.DataArray:
    """returns a normalized box kernel with given shape

    :param shape:
    :return: the kernel array with dimesions ``w1``, ``w2``
    """
    np_arr = np.ones(shape) / np.prod(shape)
    da_arr = da.from_array(np_arr, chunks={})
    return xr.DataArray(da_arr, dims=["w1", "w2"])


# Filter objects
@dataclass(frozen=True)
class Filter:
    """Basic filter class with kernel along dimensions
    the dimensions of kernel and filter_dims are matched one-to-one as given

    :ivar kernel: filter kernel
    :ivar filter_dims: dimensions along which to perform filtering;
        will be paired with dimensions of the kernel.
    """

    kernel: xr.DataArray
    filter_dims: Sequence[Hashable]

    def scales(self) -> tuple[int, ...]:
        return tuple(self.kernel.shape)

    def scale(self) -> float:
        return float(np.prod(self.scales()) ** (1 / len(self.filter_dims)))

    def scale(self) -> float:
        shape = self.scales()
        return np.prod(shape) ** (1 / len(shape))

    def scales(self) -> tuple[int, ...]:
        return self.kernel.shape

    def _filter_kernel_map(self) -> dict[Hashable, str]:
        """matches the dimesions of the `kernel` against `self.filter_dims`"""
        assert len(self.filter_dims) == len(self.kernel.dims)
        return {d: str(self.kernel.dims[i]) for i, d in enumerate(self.filter_dims)}

    def with_dims(self, dims: list[Hashable]):
        """return a new filter with same kernel and updated filter_dims

        :param dims: new dimensions; must be the same length as the original ones.
        """
        assert len(dims) == len(self.filter_dims)
        return Filter(self.kernel, dims)

    def filter(self, field: xr.DataArray) -> xr.DataArray:
        """filterer field

        :param field: array to be filtered; must contain all of `filter_dims`
        """
        assert (d in field.dims for d in self.filter_dims)
        dic_dims = self._filter_kernel_map()
        dic_roll: dict[Hashable, int] = {}
        for d in self.filter_dims:
            axnum = self.kernel.get_axis_num(dic_dims[d])
            assert isinstance(axnum, int)  # appease xarray typing
            dic_roll[d] = self.kernel.shape[axnum]

        filtered = field.rolling(dic_roll).construct(dic_dims).dot(self.kernel)
        # arr.rolling().construct.() blows up the underlying chunksizes -- restore
        filtered = filtered.chunk(
            {k: field.chunks[i] for i, k in enumerate(field.dims)}  # type: ignore
        )
        return filtered


@dataclass(frozen=True)
class IdentityFilter(Filter):
    """identity filter

    :ivar kernel: filter kernel will be ignored
    """

    def filter(self, field: xr.DataArray) -> xr.DataArray:
        """returns original field

        :param field: array to be filtered
        """
        return field

@dataclass(frozen=True)
class CoarseGrain(Filter):
    """Coarse-graining filter class with kernel along dimensions
    the dimensions of kernel and filter_dims are matched one-to-one as given

        :ivar kernel: filter kernel
    :ivar filter_dims: dimensions along which to perform filtering;
        will be paired with dimensions of the kernel.
    """
    def filter(self, field: xr.DataArray) -> xr.DataArray:
        """coarse grain field;
        Note: Warning - unlike Filter.filter, here the output size is different from the input size

        :param field: array to be filtered; must contain all of `filter_dims`
        """
        assert (d in field.dims for d in self.filter_dims)
        dic_dims = self._filter_kernel_map()
        window: dict[Hashable, int] = {}
        for d in self.filter_dims:
            axnum = self.kernel.get_axis_num(dic_dims[d])
            assert isinstance(axnum, int)  # appease xarray typing
            window[d] = self.kernel.shape[axnum]
        return field.coarsen(window, boundary='pad').reduce(np.average, keep_attrs=True, weights= self.kernel)

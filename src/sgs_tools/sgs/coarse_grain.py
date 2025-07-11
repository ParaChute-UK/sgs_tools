from dataclasses import dataclass
from typing import Dict

import numpy as np
import xarray as xr

# TODO: collate with Filter via a Protocol??


@dataclass(frozen=True)
class CoarseGrain:
    """Coarse-graining filter class with kernel along dimensions
    the dimensions of kernel and filter_dims are matched one-to-one as given

        :ivar kernel: filter kernel
    :ivar filter_dims: dimensions along which to perform filtering;
        will be paired with dimensions of the kernel.
    """

    window: Dict[str, int]

    @property
    def filter_dims(self):
        return list(self.window.keys())

    def scales(self) -> tuple[int, ...]:
        return tuple(self.window.values())

    def scale(self) -> float:
        return float(np.prod(self.scales()) ** (1 / len(self.window)))

    def __rechunked__(self, field: xr.DataArray) -> xr.DataArray:
        if not field.chunks:
            return field
        else:
            chunksizes = dict(zip(field.dims, field.chunks))
            new_chunksizes = {}
            for d in self.window:
                orig = chunksizes[d][0]
                window = self.window[d]
                if orig < window:
                    new_chunksizes[d] = window
                elif orig % window != 0:
                    new_chunksizes[d] = int(orig // window)
                else:
                    pass
                    # new_chunksizes[d] = orig

                return field.chunk(new_chunksizes)

    def filter(self, field: xr.DataArray) -> xr.DataArray:
        """coarse grain `field`;
        Note: unlike Filter.filter, here the output size is different from the input size

        :param field: array to be filtered; must contain all of `filter_dims`
        """
        # Note this needs further optimisation
        rechunked = self.__rechunked__(field)
        return rechunked.coarsen(self.window, boundary="trim").mean(keep_attrs=True)  # type: ignore


def coarse_grain_fluct(field: xr.DataArray, coarse: CoarseGrain) -> xr.DataArray:
    # coarse grain and upsample
    field_mean = coarse.filter(field).reindex_like(field, method="nearest")
    return field - field_mean

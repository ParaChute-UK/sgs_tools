from dataclasses  import dataclass
from typing import Dict
import xarray as xr
import numpy as np

#should be collated with Filter somehow??

@dataclass(frozen=True)
class CoarseGrain():
    """Coarse-graining filter class with kernel along dimensions
    the dimensions of kernel and filter_dims are matched one-to-one as given

        :ivar kernel: filter kernel
    :ivar filter_dims: dimensions along which to perform filtering;
        will be paired with dimensions of the kernel.
    """
    window: Dict[str, int]

    def scales(self) -> tuple[int]:
        return tuple(self.window.values())

    def scale(self) -> float:
        return np.prod(self.scales()) ** (1 / len(self.window))

    def filter(self, field: xr.DataArray) -> xr.DataArray:
        """coarse grain field;
        Note: Warning - unlike Filter.filter, here the output size is different from the input size

        :param field: array to be filtered; must contain all of `filter_dims`
        """
        return field.coarsen(self.window, boundary='pad').mean(keep_attrs=True)



def coarse_grain_fluct(field, coarse):
    #coarse grain and upsample
    field_mean = coarse.filter(field).reindex_like(field, method='nearest')
    return field - field_mean


from dataclasses import dataclass
from typing import Protocol, Sequence

import xarray as xr

from .filter import Filter


class SGSModel(Protocol):
    """Base subgrid-scale (SGS) model class

    * :meth:`sgs_tensor`: returns the SGS tensor for a given filter
    """

    def sgs_tensor(self, filter: Filter) -> xr.DataArray:
        """compute model for SGS tensor :math:`$\\tau$` for a given filter

        :param filter: compute the SGS tensor at this scale
        """


@dataclass(frozen=True)
class LinCombSGSModel:
    """Linear combination of subgrid-scale (SGS) models

    :param models: List of SGS model components
    """

    model_components: Sequence[SGSModel]

    def sgs_tensor_list(self, filter: Filter) -> Sequence[xr.DataArray]:
        return [mod.sgs_tensor(filter) for mod in self.model_components]

    def sgs_tensor(self, filter: Filter) -> xr.DataArray:
        return xr.concat(self.sgs_tensor_list(filter), dim="new").sum(dim="new")

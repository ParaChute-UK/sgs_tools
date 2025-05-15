from abc import ABC, abstractmethod
from dataclasses import dataclass

import xarray as xr

from ..geometry.tensor_algebra import tensor_self_outer_product
from .filter import Filter, IdentityFilter


@dataclass(frozen=True)
class SGSModel(ABC):
    """Base subgrid-scale (SGS) model class

    * :meth:`sgs_tensor`: returns the SGS tensor for a given filter
    """

    tensor_dims: tuple[str, str]

    @abstractmethod
    def sgs_tensor(self, filter: Filter) -> xr.DataArray:
        """compute model for SGS tensor :math:`$\\tau$` for a given filter

        :param filter: compute the SGS tensor at this scale
        """


@dataclass(frozen=True)
class DynamicSGSModel(ABC):
    """Dynamic SGS model based on Germano's identity

    :ivar StaticModel: Static (scale-unaware) SGS model

    * :meth:`M_Germano_tensor`: returns the Germano model tensor "M" for a given filter
    * :meth:`Leonard_tensor`: returns the Leonard tensor "L" for a given filter
    """

    StaticModel: SGSModel

    def M_Germano_tensor(self, filter: Filter) -> xr.DataArray:
        """compute the Mij Germano model tensor as
        (<tau(at grid)> - alpha^2 tau(at filter))
        where (delta * alpha) is the area/volume spanned by the filter kernel

        :param filter: Filter used to separate "large" and "small" scales
        """
        id = IdentityFilter(xr.DataArray(), filter.filter_dims)
        filtered = filter.filter(self.StaticModel.sgs_tensor(id))
        resolved = self.StaticModel.sgs_tensor(filter)
        alpha_sq = filter.kernel.size
        M = filtered - alpha_sq * resolved
        return M

    # this is abstract because the mulitplication operation
    # changes for vector and scalar models
    @abstractmethod
    def Leonard_tensor(self, filter: Filter) -> xr.DataArray:
        """compute the Leonard tensor for a given filter

        :param filter: Filter used to separate "large" and "small" scales
        """
        ...


@dataclass(frozen=True)
class DynamicVelocityModel(DynamicSGSModel):
    """Dynamic SGS model for the velocity equation based on Germano's identity

    :ivar StaticModel: Static (scale-unaware) SGS model
    :ivar vel: grid-scale/base velocity field
    :ivar tensor_dims: labels of dimensions indexing tensor components

    * :meth:`M_Germano_tensor`: returns the Germano model tensor "M" for a given filter
    * :meth:`Leonard_tensor`: returns the Leonard tensor "L" for a given filter
    """

    vel: xr.DataArray
    tensor_dims: tuple[str, str]

    def Leonard_tensor(self, filter: Filter) -> xr.DataArray:
        """compute the Leonard tensor as
            :math:`\overline{v_i v_j} - \overline{v_i} \overline{v_j}`,
            where :math:`\overline{\\ast}` means filtering

        :param filter: Filter used to separate "large" and "small" scales
        """
        resolved = tensor_self_outer_product(filter.filter(self.vel), *self.tensor_dims)
        filtered = filter.filter(tensor_self_outer_product(self.vel, *self.tensor_dims))
        L = filtered - resolved
        return L


@dataclass(frozen=True)
class DynamicHeatModel(DynamicSGSModel):
    """Dynamic SGS model for the (potential) temperature based on Germano's identity

    :ivar StaticModel: Static (scale-unaware) SGS model
    :ivar vel: grid-scale/base velocity field
    :ivar theta: grid-scale/base temperature field

    * :meth:`M_Germano_tensor`: returns the Germano model tensor "M" for a given filter
    * :meth:`Leonard_tensor`: returns the Leonard tensor "L" for a given filter
    """

    vel: xr.DataArray
    theta: xr.DataArray

    def Leonard_tensor(self, filter: Filter) -> xr.DataArray:
        """compute the Leonard tensor as
            :math:`\overline{v_i \\theta} - \overline{v_i} \overline{\\theta}`,
            where :math:`\overline{\\ast}` means filtering

        :param filter: Filter used to separate "large" and "small" scales
        """
        resolved = filter.filter(self.vel) * filter.filter(self.theta)
        filtered = filter.filter(self.vel * self.theta)
        L = filtered - resolved
        return L

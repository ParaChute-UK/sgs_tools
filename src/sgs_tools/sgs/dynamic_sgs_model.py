from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import xarray as xr

from ..geometry.tensor_algebra import tensor_self_outer_product
from .dynamic_coefficient import Minimisation
from .filter import Filter, IdentityFilter
from .sgs_model import LinCombSGSModel, SGSModel


class LeonardTensor(Protocol):
    """compute the Leonard tensor for a large-scale equation

    * :meth:`compute`: returns the SGS tensor for a given filter
    """

    def compute(self, filter: Filter) -> xr.DataArray:
        """compute the Leonard tensor given a test filter"""


@dataclass(frozen=True)
class LeonardVelocityTensor:
    """Leonard tensor for the velocity

    :ivar vel: grid-scale/base velocity field
    :param contraction_dims: labels of dimensions
                             to form the :math:$v_i v_j$` tensor
    """

    vel: xr.DataArray
    tensor_dims: tuple[str, str]

    def compute(self, filter: Filter) -> xr.DataArray:
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
class LeonardThetaTensor:
    """Leonard tensor for the (potential) temperature :math:`$\theta$`

    :ivar vel: grid-scale/base velocity field
    :ivar theta: grid-scale/base temperature field
    """

    vel: xr.DataArray
    theta: xr.DataArray
    tensor_dims: tuple[str, str]

    def compute(self, filter: Filter) -> xr.DataArray:
        """compute the Leonard tensor as
            :math:`\overline{v_i \\theta} - \overline{v_i} \overline{\\theta}`,
            where :math:`\overline{\\ast}` means filtering

        :param filter: Filter used to separate "large" and "small" scales
        """
        resolved = filter.filter(self.vel) * filter.filter(self.theta)
        filtered = filter.filter(self.vel * self.theta)
        L = filtered - resolved
        return L


def M_Germano_tensor(sgs_model: SGSModel, filter: Filter) -> xr.DataArray:
    """compute the Mij Germano model tensor as
    (<tau(at grid)> - alpha^2 tau(at filter))
    where (delta * alpha) is the area/volume spanned by the filter kernel

    :param filter: Filter used to separate "large" and "small" scales
    """
    id = IdentityFilter(xr.DataArray(), filter.filter_dims)
    filtered = filter.filter(sgs_model.sgs_tensor(id))
    resolved = sgs_model.sgs_tensor(filter)
    alpha_sq = filter.kernel.size
    M = filtered - alpha_sq * resolved
    return M


@dataclass(frozen=True)
class DynamicModel:
    static_model: SGSModel
    leonard: LeonardTensor

    def compute_coeff(
        self, test_filter: Filter, minimisation: Minimisation
    ) -> xr.DataArray:
        L = self.leonard.compute(test_filter)
        M = M_Germano_tensor(self.static_model, test_filter)
        return minimisation.compute(L, [M])

    def sgs_tensor(self):
        return self.compute_coeff() * self.static_model.sgs_tensor(
            self.minimisation.test_filter
        )


@dataclass(frozen=True)
class LinCombDynamicModel:
    static_model: LinCombSGSModel
    leonard: LeonardTensor

    def compute_coeff(
        self, test_filter: Filter, minimisation: Minimisation
    ) -> xr.DataArray:
        L = self.leonard.compute(test_filter)
        M = [
            M_Germano_tensor(m, test_filter) for m in self.static_model.model_components
        ]
        return minimisation.compute(L, M)

    def sgs_tensor(
        self, test_filter: Filter, minimisation: Minimisation
    ) -> xr.DataArray:
        tau_list = self.static_model.sgs_tensor_list(test_filter)
        coeff = self.compute_coeff(test_filter, minimisation)
        # ensure label alignment between static_model and minimisation
        cdim = minimisation.coeff_dim
        if cdim in tau_list[0].dims:
            coeff = coeff.rename({cdim: cdim + "_dummy"})
            cdim = cdim + "_dummy"
        tau_arr = xr.concat(tau_list, dim=cdim)
        return (coeff * tau_arr).sum(dim=cdim)

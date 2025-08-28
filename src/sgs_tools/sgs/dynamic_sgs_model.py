from dataclasses import dataclass
from typing import Protocol

import xarray as xr

from ..geometry.tensor_algebra import tensor_self_outer_product
from .dynamic_minimisation import Minimisation
from .filter import Filter, IdentityFilter
from .sgs_model import LinCombSGSModel, SGSModel


class LeonardTensor(Protocol):
    """compute the Leonard tensor for a large-scale equation

    :meth:`compute`: returns the Leonard tensor for a given filter
    """

    def compute(self, filter: Filter) -> xr.DataArray:
        """"""


@dataclass(frozen=True)
class LeonardVelocityTensor:
    """Leonard tensor for the velocity

    :ivar vel: grid-scale/base velocity field
    :ivar tensor_dims: tensor dimensions
    """

    vel: xr.DataArray
    tensor_dims: tuple[str, str]

    def compute(self, filter: Filter) -> xr.DataArray:
        r"""compute the Leonard tensor as
            :math:`\overline{v_i v_j} - \overline{v_i} \overline{v_j}`,
            where :math:`\overline{\ast}` means filtering

        :param filter: Filter used to separate "large" and "small" scales
        """
        resolved = tensor_self_outer_product(filter.filter(self.vel), *self.tensor_dims)
        filtered = filter.filter(tensor_self_outer_product(self.vel, *self.tensor_dims))
        L = filtered - resolved
        return L


@dataclass(frozen=True)
class LeonardThetaTensor:
    r"""Leonard tensor for the (potential) temperature :math:`\theta`

    :ivar vel: grid-scale/base velocity field
    :ivar theta: grid-scale/base temperature field
    """

    vel: xr.DataArray
    theta: xr.DataArray
    tensor_dims: tuple[str, str]

    def compute(self, filter: Filter) -> xr.DataArray:
        r"""compute the Leonard tensor as
            :math:`\overline{v_i \theta} - \overline{v_i} \overline{\theta}`,
            where :math:`\overline{\ast}` means filtering

        :param filter: Filter used to separate "large" and "small" scales
        """
        resolved = filter.filter(self.vel) * filter.filter(self.theta)
        filtered = filter.filter(self.vel * self.theta)
        L = filtered - resolved
        return L


def M_Germano_tensor(sgs_model: SGSModel, filter: Filter) -> xr.DataArray:
    r"""compute the Mij Germano model tensor as
    :math:`(<\tau(\matrhm{at grid})> - \alpha^2 \tau(\mathrm{at filter}))`,
    where :math:`\Delta * \alpha` is the area/volume spanned by the filter kernel
    and :math:`\Delta` is the filter scale.

    :param sgs_model: SGS model used to compute the SGS tensor.
    :param filter: Filter used to separate "large" and "small" scales
    """
    id = IdentityFilter(xr.DataArray(), filter.filter_dims)
    filtered = filter.filter(sgs_model.sgs_tensor(id))
    resolved = sgs_model.sgs_tensor(filter)
    alpha_sq = filter.kernel.size
    M = filtered - alpha_sq * resolved
    return M


class DynamicModelProtcol(Protocol):
    """compute the Leonard tensor for a large-scale equation

    :meth compute_coeff : returns the SGS tensor for a given filter
    :meth sgs_tensor: returns the SGS tensor for a given test and regularisation filter
    """

    def compute_coeff(self, test_filter: Filter, reg_filter: Filter) -> xr.DataArray:
        """ """

    def sgs_tensor(self, test_filter: Filter, reg_filter: Filter) -> xr.DataArray:
        """ """


@dataclass(frozen=True)
class DynamicModel:
    r"""Dynamic model for SGS model

    :ivar static_model: Static SGS model
    :ivar leonard: Leonard tensor for the large-scale equation
    :ivar minimisation: Minimisation method to compute coefficients
    :meth compute_coeff: computes the amplitude coefficients for the SGS model
    :meth sgs_tensor: computes the SGS tensor for a given test and regularisation filters
    """

    static_model: SGSModel
    leonard: LeonardTensor
    minimisation: Minimisation

    def compute_coeff(self, test_filter: Filter, reg_filter: Filter) -> xr.DataArray:
        """compute the amplitude coefficient for the SGS model

        :param test_filter: Filter used to separate "large" and "small" scales
        :param reg_filter: Filter used for regularisation
        """

        L = self.leonard.compute(test_filter)
        M = M_Germano_tensor(self.static_model, test_filter)
        return self.minimisation.compute(L, [M], reg_filter)

    def sgs_tensor(self, test_filter: Filter, reg_filter: Filter) -> xr.DataArray:
        """compute the SGS tensor for a given test and regularisation filter

        :param test_filter: Filter used to separate "large" and "small" scales
        :param reg_filter: Filter used for regularisation
        """

        return self.compute_coeff(
            test_filter, reg_filter
        ) * self.static_model.sgs_tensor(test_filter)


@dataclass(frozen=True)
class LinCombDynamicModel:
    """Dynamic model for linear combination SGS model

    :ivar static_model: Static linear combination SGS model
    :ivar leonard: Leonard tensor for the large-scale equation
    :ivar minimisation: Minimisation method to compute coefficients
    :meth compute_coeff: computes the coefficients for the SGS model
    :meth sgs_tensor: computes the SGS tensor for a given test and regularisation filter
    """

    static_model: LinCombSGSModel
    leonard: LeonardTensor
    minimisation: Minimisation

    def compute_coeff(self, test_filter: Filter, reg_filter: Filter) -> xr.DataArray:
        """compute the coefficients for the SGS model

        :param test_filter: Filter used to separate "large" and "small" scales
        :param reg_filter: Filter used for regularisation
        """

        L = self.leonard.compute(test_filter)
        M = [
            M_Germano_tensor(m, test_filter) for m in self.static_model.model_components
        ]
        return self.minimisation.compute(L, M, reg_filter)

    def sgs_tensor(self, test_filter: Filter, reg_filter: Filter) -> xr.DataArray:
        """compute the SGS tensor for a given test and regularisation filter

        :param test_filter: Filter used to separate "large" and "small" scales
        :param reg_filter: Filter used for regularisation
        """

        tau_list = self.static_model.sgs_tensor_list(test_filter)
        coeff = self.compute_coeff(test_filter, reg_filter)
        # ensure alignment of coefficient index label between static_model and minimisation
        cdim = self.minimisation.coeff_dim
        if cdim in tau_list[0].dims:
            coeff = coeff.rename({cdim: cdim + "_dummy"})
            cdim = cdim + "_dummy"
        tau_arr = xr.concat(tau_list, dim=cdim)
        return (coeff * tau_arr).sum(dim=cdim)

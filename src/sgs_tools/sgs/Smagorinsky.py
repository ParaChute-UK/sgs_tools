from dataclasses import dataclass

import xarray as xr  # only used for type hints

from ..geometry.tensor_algebra import Frobenius_norm
from .dynamic_coefficient import Minimisation
from .dynamic_sgs_model import DynamicModel, LeonardThetaTensor, LeonardVelocityTensor
from .filter import Filter
from .util import _assert_coord_dx


@dataclass(frozen=True)
class SmagorinskyVelocityModel:
    """Smagorinsky model for the velocity equation

    :ivar vel: grid-scale velocity
    :ivar strain: grid-scale rate-of-strain
    :ivar cs: Smagorinsky coefficient
    :ivar dx: constant resolution with respect to dimension to-be-filtered
    """

    vel: xr.DataArray
    strain: xr.DataArray
    cs: float
    dx: float
    tensor_dims: tuple[str, str]

    def sgs_tensor(self, filter: Filter) -> xr.DataArray:
        r"""compute model for SGS tensor
            :math:`\tau = (c_s \Delta) ^2 |\overline{S_{ij}}| \overline{S_{ij}}`
            for a given `filter` (which can be trivial, i.e. ``IdentityFilter``)

        :param filter: Filter used to separate "large" and "small" scales
        """

        # only makes sense for uniform coordinates in the filtering directions
        # with spacing of self.dx
        for arr in [self.vel, self.strain]:
            _assert_coord_dx(list(filter.filter_dims), arr, self.dx)

        sij = filter.filter(self.strain)
        snorm = Frobenius_norm(sij, self.tensor_dims)
        tau = (self.cs * self.dx) ** 2 * snorm * sij
        return tau


@dataclass(frozen=True)
class SmagorinskyHeatModel:
    """Smagorinsky model for the Heat equation

    :ivar vel: grid-scale velocity
    :ivar grad_theta: grid-scale (potential) temperature gradient
    :ivar strain: grid-scale rate-of-strain
    :ivar ctheta: Smagorinsky coefficient for the heat equation
    :ivar dx: constant resolution with respect to dimension to-be-filtered
    """

    vel: xr.DataArray
    grad_theta: xr.DataArray
    strain: xr.DataArray
    ctheta: float
    dx: float
    tensor_dims: tuple[str, str]

    def sgs_tensor(self, filter):
        r"""compute model for SGS tensor
            :math:`\tau =  c_\theta \Delta^2 |\overline{S_{ij}}| \overline{\nabla \theta}`
            for a given filter (which can be trivial, i.e. IdentityFilter)

        :param filter: Filter used to separate "large" and "small" scales
        """

        # only makes sense for uniform coordinates in the filtering directions
        # with spacing of self.dx
        for arr in [self.vel, self.grad_theta, self.strain]:
            _assert_coord_dx(filter.filter_dims, arr, self.dx)

        snorm = Frobenius_norm(filter.filter(self.strain), self.tensor_dims)
        grad_theta = filter.filter(self.grad_theta)
        tau = self.ctheta * self.dx**2 * snorm * grad_theta
        return tau


def DynamicSmagorinskyVelocityModel(
    smag_vel: SmagorinskyVelocityModel, minimisation: Minimisation
) -> DynamicModel:
    leonard = LeonardVelocityTensor(smag_vel.vel, smag_vel.tensor_dims)
    return DynamicModel(smag_vel, leonard, minimisation)


def DynamicSmagorinskyHeatModel(
    smag_theta: SmagorinskyHeatModel, theta: xr.DataArray, minimisation: Minimisation
) -> DynamicModel:
    leonard = LeonardThetaTensor(smag_theta.vel, theta, smag_theta.tensor_dims)
    return DynamicModel(smag_theta, leonard, minimisation)

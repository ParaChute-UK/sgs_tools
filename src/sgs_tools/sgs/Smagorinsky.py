from dataclasses import dataclass

import xarray as xr  # only used for type hints

from ..geometry.tensor_algebra import Frobenius_norm
from .dynamic_sgs_model import DynamicModel, LeonardThetaTensor, LeonardVelocityTensor
from .filter import Filter
from .util import _assert_coord_dx


@dataclass(frozen=True)
class SmagorinskyVelocityModel:
    """Smagorinsky model for the velocity equation

    :ivar strain: grid-scale rate-of-strain
    :ivar cs: Smagorinsky coefficient
    :ivar dx: constant resolution with respect to dimension to-be-filtered
    """

    strain: xr.DataArray
    cs: float
    dx: float
    tensor_dims: tuple[str, str]

    def sgs_tensor(self, filter: Filter) -> xr.DataArray:
        """compute model for SGS tensor
            :math:`$\\tau = (c_s \Delta) ^2 |\overline{Sij}| \overline{Sij}$`
            for a given `filter` (which can be trivial, i.e. ``IdentityFilter``)

        :param filter: Filter used to separate "large" and "small" scales
        """

        # only makes sense for uniform coordinates in the filtering directions
        # with spacing of self.dx
        _assert_coord_dx(filter.filter_dims, self.strain, self.dx)

        sij = filter.filter(self.strain)
        snorm = Frobenius_norm(sij, self.tensor_dims)
        tau = (self.cs * self.dx) ** 2 * snorm * sij
        return tau


@dataclass(frozen=True)
class SmagorinskyHeatModel:
    """Smagorinsky model for the Heat equation

    :ivar grad_theta: grid-scale (potential) temperature gradient
    :ivar strain: grid-scale rate-of-strain
    :ivar ctheta: Smagorinsky coefficient for the heat equation
    :ivar dx: constant resolution with respect to dimension to-be-filtered
    """

    grad_theta: xr.DataArray
    strain: xr.DataArray
    ctheta: float
    dx: float
    tensor_dims: tuple[str, str]

    def sgs_tensor(self, filter):
        """compute model for SGS tensor
            :math:`$\\tau =  c_\\theta \\Delta^2 |\overline{Sij}| \overline{\\nabla \\theta} $`
            for a given filter (which can be trivial, i.e. IdentityFilter)

        :param filter: Filter used to separate "large" and "small" scales
        """

        # only makes sense for uniform coordinates in the filtering directions
        # with spacing of self.dx
        for arr in [self.grad_theta, self.strain]:
            _assert_coord_dx(filter.filter_dims, arr, self.dx)

        snorm = Frobenius_norm(filter.filter(self.strain), self.tensor_dims)
        grad_theta = filter.filter(self.grad_theta)
        tau = self.ctheta * self.dx**2 * snorm * grad_theta
        return tau


def DynamicSmagorinskyVelocityModel(
    smag_vel: SmagorinskyVelocityModel, vel: xr.DataArray,
) -> DynamicModel:
    leonard = LeonardVelocityTensor(vel, smag_vel.tensor_dims)
    return DynamicModel(smag_vel, leonard)


def DynamicSmagorinskyHeatModel(
    smag_theta: SmagorinskyHeatModel, vel: xr.DataArray, theta: xr.DataArray,
) -> DynamicModel:
    leonard = LeonardThetaTensor(vel, theta, smag_theta.tensor_dims)
    return DynamicModel(smag_theta, leonard)

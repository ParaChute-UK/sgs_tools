from dataclasses import dataclass
from typing import Hashable

import xarray as xr  # only used for type hints

from ..geometry.tensor_algebra import Frobenius_norm
from .dynamic_sgs_model import DynamicModel, LeonardThetaTensor, LeonardVelocityTensor
from .filter import Filter


# check that arr is uniform along `filter_dims` with spacing of `dx`
def _assert_coord_dx(filter_dims: list[Hashable], arr: xr.DataArray, dx: float) -> None:
    for c in filter_dims:
        assert (arr[c].diff(dim=c) == dx).all(), f"Not uniform dimension {c}: {arr[c]}"


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
            :math:`$\\tau = (c_s \Delta) ^2 |\overline{Sij}| \overline{Sij}$`
            for a given `filter` (which can be trivial, i.e. ``IdentityFilter``)

        :param filter: Filter used to separate "large" and "small" scales
        """

        # only makes sense for uniform coordinates in the filtering directions
        # with spacing of self.dx
        for arr in [self.vel, self.strain]:
            _assert_coord_dx(filter.filter_dims, arr, self.dx)

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
            :math:`$\\tau =  c_\\theta \\Delta^2 |\overline{Sij}| \overline{\\nabla \\theta} $`
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
    smag_vel: SmagorinskyVelocityModel,
) -> DynamicModel:
    leonard = LeonardVelocityTensor(smag_vel.vel, smag_vel.tensor_dims)
    return DynamicModel(smag_vel, leonard)


def DynamicSmagorinskyHeatModel(
    smag_theta: SmagorinskyHeatModel, theta: xr.DataArray
) -> DynamicModel:
    leonard = LeonardThetaTensor(smag_theta.vel, theta, smag_theta.tensor_dims)
    return DynamicModel(smag_theta, leonard)

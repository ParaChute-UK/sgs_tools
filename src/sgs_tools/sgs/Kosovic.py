import warnings
from dataclasses import dataclass
from typing import Sequence

import xarray as xr

from ..geometry.tensor_algebra import (
    symmetrise,
    traceless,
)
from .dynamic_coefficient import LillyMinimisation2Model, Minimisation
from .dynamic_sgs_model import LeonardVelocityTensor, LinCombDynamicModel
from .filter import Filter
from .sgs_model import LinCombSGSModel
from .util import _assert_coord_dx


@dataclass(frozen=True)
class SSquaredVelocityModel:
    r"""Kosovic JFM 1997, 336 and Speziale 1991 Ann Rev Fluid Mech 23
    compoment :math:`\tau = (c_s \Delta)^2 \mathrm{Traceless}(\overline{S_{ik}} \overline{S_{kj}})`

    :ivar strain: grid-scale rate-of-strain
    :ivar cs: Smagorinsky coefficient
    :ivar dx: constant resolution with respect to dimension to-be-filtered
    :ivar tensor_dims: labels of dimensions indexing tensor components
    """

    strain: xr.DataArray
    cs: float
    dx: float
    tensor_dims: tuple[str, str]

    def sgs_tensor(self, filter: Filter) -> xr.DataArray:
        r"""compute model for SGS tensor
           :math:`\tau = (c_s \Delta) ^2 \overline{S_{ik}}\overline{S_{kj}}`
           for a given `filter` (which can be trivial, i.e. ``IdentityFilter``)

        :param filter: Filter used to separate "large" and "small" scales
        """

        # only makes sense for uniform coordinates in the filtering directions
        # with spacing of self.dx
        _assert_coord_dx(filter.filter_dims, self.strain, self.dx)

        sij = filter.filter(self.strain)
        sleft = sij.rename({self.tensor_dims[1]: "dummy1"})
        sright = sij.rename({self.tensor_dims[0]: "dummy1"})
        s_prod = traceless(xr.dot(sleft, sright, dim="dummy1", optimize=True))

        tau = (self.cs * self.dx) ** 2 * s_prod
        return tau


@dataclass(frozen=True)
class SOmegaVelocityModel:
    r"""Kosovic JFM 1997, 336 / Speziale 1991 Ann Rev Fluid Mech 23
    component :math:`\tau = (c_s \Delta) ^2 \mathrm{Sym}[\overline{S_{ik}}\overline{\Omega_{kj}}]`

    :ivar strain: grid-scale rate-of-strain
    :ivar rot: grid-scale rate-of-rotation
    :ivar cs: Smagorinsky coefficient
    :ivar dx: constant resolution with respect to dimension to-be-filtered
    :ivar tensor_dims: labels of dimensions indexing tensor components
    """

    strain: xr.DataArray
    rot: xr.DataArray
    cs: float
    dx: float
    tensor_dims: tuple[str, str]

    def sgs_tensor(self, filter: Filter) -> xr.DataArray:
        r"""compute model for SGS tensor
            :math:`\tau = (c_s \Delta) ^2 \mathrm{Sym}[\overline{S_{ik}}\overline{\Omega_{kj}}]`
            for a given `filter` (which can be trivial, i.e. ``IdentityFilter``)

        :param filter: Filter used to separate "large" and "small" scales
        """

        # only makes sense for uniform coordinates in the filtering directions
        # with spacing of self.dx
        _assert_coord_dx(filter.filter_dims, self.strain, self.dx)
        _assert_coord_dx(filter.filter_dims, self.rot, self.dx)
        warnings.warn(
            """Warning: No check that self.strain is symmetric and self.rot is antisymmetric.
          If false will produce trace-full result"""
        )

        sij = filter.filter(self.strain).rename({self.tensor_dims[1]: "dummy1"})
        rotij = filter.filter(self.rot).rename({self.tensor_dims[0]: "dummy1"})
        s_prod = symmetrise(xr.dot(sij, rotij, dim="dummy1", optimize=True))
        # assumes that self.rot is anti-symmetric, and self.strain is symmetric
        # so we don't need to make the product traceless explicitly
        tau = (self.cs * self.dx) ** 2 * s_prod
        return tau


def DynamicKosovicModel(
    sij: xr.DataArray,
    omegaij: xr.DataArray,
    vel: xr.DataArray,
    res: float,
    compoment_coeff: Sequence[float],
    tensor_dims: tuple[str, str] = ("c1", "c2"),
    minimisation: Minimisation = LillyMinimisation2Model(
        contraction_dims=["c1", "c2"], coeff_dim="cdim"
    ),
) -> LinCombDynamicModel:
    r"""Dynamic version of the model by
    Carati & Cabot Proceedings of the 1996 Summer Program -- Center for Turbulence Research

    :param sij: grid-scale rate-of-strain tensor
    :param omegaij: grid-scale rate-of-rotation tensor
    :param vel: velocity field used for dynamic coefficient computation
    :param res: constant resolution with respect to dimension to-be-filtered
    :param compoment_coeff: tuple of three Smagorinsky coefficients for parallel, perpendicular, and normal components
    :param tensor_dims: labels of dimensions indexing tensor components, defaults to ("c1", "c2")
    :return: Combined SGS model with dynamically computed coefficients
    """
    static_model = LinCombSGSModel(
        [
            SSquaredVelocityModel(
                strain=sij,
                cs=compoment_coeff[0],
                dx=res,
                tensor_dims=tensor_dims,
            ),
            SOmegaVelocityModel(
                strain=sij,
                rot=omegaij,
                cs=compoment_coeff[1],
                dx=res,
                tensor_dims=tensor_dims,
            ),
        ]
    )
    leonard = LeonardVelocityTensor(vel, tensor_dims)
    return LinCombDynamicModel(static_model, leonard, minimisation)

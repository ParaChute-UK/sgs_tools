from dataclasses import dataclass
from typing import Sequence

import dask.array as da
import xarray as xr

from ..geometry.tensor_algebra import (
    Frobenius_norm,
    symmetrise,
    tensor_self_outer_product,
    traceless,
)
from .dynamic_sgs_model import LeonardVelocityTensor, LinCombDynamicModel
from .filter import Filter
from .sgs_model import LinCombSGSModel
from .util import _assert_coord_dx


def s_parallel(
    s: xr.DataArray, n: xr.DataArray, tensor_dims: tuple[str, str]
) -> xr.DataArray:
    """(n_i (s.n)_j + (s.n)_i n_j - 2/3 \delta_ij (n.s.n) )"""
    assert len(n.dims) == 1
    assert len(tensor_dims) == 2
    assert n.dims[0] in tensor_dims
    assert all([d in s.dims for d in tensor_dims])

    s_vec = xr.dot(s, n, dim=n.dims)  # s = S.n
    S_parallel = symmetrise(n * s_vec, dims=tensor_dims)  # the first two terms
    S_parallel_traceless = traceless(S_parallel, dims=tensor_dims)  # add the last term
    return S_parallel_traceless


def s_perpendicular(
    s: xr.DataArray, n: xr.DataArray, tensor_dims: tuple[str, str]
) -> xr.DataArray:
    return s - s_parallel(s, n, tensor_dims)


@dataclass(frozen=True)
class SparallelVelocityModel:
    """Carati & Cabot Proceedings of the 1996 Summer Program -- Center for Turbulence Research
    S_parallel component = |S| Traceless[Symmetric[(S.n)n]]

    :ivar strain: grid-scale rate-of-strain
    :ivar cs: Smagorinsky coefficient
    :ivar dx: constant resolution with respect to dimension to-be-filtered
    :ivar n: triple of floats to be coerce as a 3d constant vector along one of the tensor dimensions
    :ivar tensor_dims: labels of dimensions indexing tensor components
    """

    strain: xr.DataArray
    cs: float
    dx: float
    tensor_dims: tuple[str, str]
    n: Sequence[float]

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
        chunks = sij.chunks[sij.get_axis_num(self.tensor_dims[0])]
        n_dask = da.from_array(self.n, chunks=chunks)
        n = xr.DataArray(
            n_dask,
            dims=[self.tensor_dims[0]],
            coords={self.tensor_dims[0]: self.strain.coords[self.tensor_dims[0]]},
        )
        s_par = s_parallel(sij, n, self.tensor_dims)
        snorm = Frobenius_norm(sij, self.tensor_dims)
        tau = (self.cs * self.dx) ** 2 * snorm * s_par
        return tau.assign_coords(
            {tdim: [1, 2, 3] for tdim in self.tensor_dims if tdim in tau.dims}
        )


@dataclass(frozen=True)
class SperpVelocityModel:
    """Carati & Cabot Proceedings of the 1996 Summer Program -- Center for Turbulence Research
    S_perp component = |S| Traceless(S - (S.n + n.S))

    :ivar strain: grid-scale rate-of-strain
    :ivar cs: Smagorinsky coefficient
    :ivar dx: constant resolution with respect to dimension to-be-filtered
    :ivar n: triple of floats to be coerce as a 3d constant vector along one of the tensor dimensions
    :ivar tensor_dims: labels of dimensions indexing tensor components
    """

    strain: xr.DataArray
    cs: float
    dx: float
    n: Sequence[float]
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
        chunks = sij.chunks[sij.get_axis_num(self.tensor_dims[0])]
        n_dask = da.from_array(self.n, chunks=chunks)
        n = xr.DataArray(
            n_dask,
            dims=[self.tensor_dims[0]],
            coords={self.tensor_dims[0]: self.strain.coords[self.tensor_dims[0]]},
        )
        s_per = s_perpendicular(sij, n, self.tensor_dims)
        snorm = Frobenius_norm(sij, self.tensor_dims)
        tau = (self.cs * self.dx) ** 2 * snorm * s_per
        return tau.assign_coords(
            {tdim: [1, 2, 3] for tdim in self.tensor_dims if tdim in tau.dims}
        )


@dataclass(frozen=True)
class NVelocityModel:
    """Carati & Cabot Proceedings of the 1996 Summer Program -- Center for Turbulence Research
       N component = |S|(n.s.n) Traceless(n * n)

    :ivar strain: grid-scale rate-of-strain
    :ivar cs: Smagorinsky coefficient
    :ivar dx: constant resolution with respect to dimension to-be-filtered
    :ivar n: triple of floats to be coerce as a 3d constant vector along one of the tensor dimensions
    :ivar tensor_dims: labels of dimensions indexing tensor components
    """

    strain: xr.DataArray
    cs: float
    dx: float
    n: Sequence[float]
    tensor_dims: tuple[str, str]

    def sgs_tensor(self, filter: Filter) -> xr.DataArray:
        """compute model for SGS tensor
            :math:`$\\tau = (c_s \Delta) ^2 |\overline{Sij}| \overline{Sik Nkj}$`
            for a given `filter` (which can be trivial, i.e. ``IdentityFilter``)

        :param filter: Filter used to separate "large" and "small" scales
        """

        # only makes sense for uniform coordinates in the filtering directions
        # with spacing of self.dx
        _assert_coord_dx(filter.filter_dims, self.strain, self.dx)
        n_dask = da.from_array(self.n, chunks=-1)
        n = xr.DataArray(
            n_dask,
            dims=[self.tensor_dims[0]],
            coords={self.tensor_dims[0]: self.strain.coords[self.tensor_dims[0]]},
        )
        nij = traceless(
            tensor_self_outer_product(n, self.tensor_dims[0], self.tensor_dims[1])
        )
        # rechunk like sij for consistency with other model components
        nij = nij.chunk()
        sij = filter.filter(self.strain)
        sn = xr.dot(sij, nij, dim=self.tensor_dims)
        snorm = Frobenius_norm(sij, self.tensor_dims)
        tau = (self.cs * self.dx) ** 2 * snorm * (sn * nij)
        return tau.assign_coords(
            {tdim: [1, 2, 3] for tdim in self.tensor_dims if tdim in tau.dims}
        )


def DynamicCaratiCabotModel2(
    sij: xr.DataArray,
    vel: xr.DataArray,
    res: float,
    compoment_coeff: Sequence[float],
    n=Sequence[float],
    tensor_dims: tuple[str, str] = ("c1", "c2"),
) -> LinCombDynamicModel:
    """Dynamic version of the model by
    Carati & Cabot Proceedings of the 1996 Summer Program -- Center for Turbulence Research
    the model version without the third term NiNj

    :param sij: grid-scale rate-of-strain tensor
    :param vel: velocity field used for dynamic coefficient computation
    :param res: constant resolution with respect to dimension to-be-filtered
    :param compoment_coeff: tuple of three Smagorinsky coefficients for parallel, perpendicular, and normal components
    :param n: triple of floats to be coerced as a 3d constant vector along one of the tensor dimensions
    :param tensor_dims: labels of dimensions indexing tensor components, defaults to ("c1", "c2")
    :return: Combined SGS model with dynamically computed coefficients
    """
    static_model = LinCombSGSModel(
        [
            SparallelVelocityModel(
                strain=sij,
                cs=compoment_coeff[0],
                dx=res,
                n=n,
                tensor_dims=tensor_dims,
            ),
            SperpVelocityModel(
                strain=sij,
                cs=compoment_coeff[1],
                dx=res,
                n=n,
                tensor_dims=tensor_dims,
            )
        ]
    )
    leonard = LeonardVelocityTensor(vel, tensor_dims)
    return LinCombDynamicModel(static_model, leonard)


def DynamicCaratiCabotModel3(
    sij: xr.DataArray,
    vel: xr.DataArray,
    res: float,
    compoment_coeff: Sequence[float],
    n=Sequence[float],
    tensor_dims: tuple[str, str] = ("c1", "c2"),
) -> LinCombDynamicModel:
    """Dynamic version of the model by
    Carati & Cabot Proceedings of the 1996 Summer Program -- Center for Turbulence Research
    Adding the ignored term NiNj

    :param sij: grid-scale rate-of-strain tensor
    :param vel: velocity field used for dynamic coefficient computation
    :param res: constant resolution with respect to dimension to-be-filtered
    :param compoment_coeff: tuple of three Smagorinsky coefficients for parallel, perpendicular, and normal components
    :param n: triple of floats to be coerced as a 3d constant vector along one of the tensor dimensions
    :param tensor_dims: labels of dimensions indexing tensor components, defaults to ("c1", "c2")
    :return: Combined SGS model with dynamically computed coefficients
    """
    static_model = LinCombSGSModel(
        [
            SparallelVelocityModel(
                strain=sij,
                cs=compoment_coeff[0],
                dx=res,
                n=n,
                tensor_dims=tensor_dims,
            ),
            SperpVelocityModel(
                strain=sij,
                cs=compoment_coeff[1],
                dx=res,
                n=n,
                tensor_dims=tensor_dims,
            ),
            NVelocityModel(
                strain=sij,
                cs=compoment_coeff[2],
                dx=res,
                n=n,
                tensor_dims=tensor_dims,
            ),
        ]
    )
    leonard = LeonardVelocityTensor(vel, tensor_dims)
    return LinCombDynamicModel(static_model, leonard)

from typing import Callable, Collection, Iterable

import xarray as xr

from ..geometry.staggered_grid import compose_vector_components_on_grid
from ..geometry.tensor_algebra import symmetrise, tensor_self_outer_product, traceless
from ..geometry.vector_calculus import grad_vector


def strain_from_vel(
    vel: xr.Dataset | Iterable[xr.DataArray],
    space_dims: Iterable[str],
    vec_dim: str,
    new_dim: str = "c2",
    make_traceless: bool = True,
    grad_operator: Callable = grad_vector,
) -> xr.DataArray:
    """compute rate of strain from velocity

    :param vel: input velocity array (on collocated grid)
    :param space_dims: labels of spacial dimensions
    :param vec_dim: label of vector dimension
    :param new_dim: label of new dimension indexing derivatives
    :param make_traceless: should we make the strain traceless
    :param grad_operator: operator that computes vector gradient (To be replaced by a grid)
    """
    gradvel = grad_operator(vel, space_dims, new_dim)
    # perform manual alignment of c1 and c2 indices (assumed sorted)
    c2 = gradvel[new_dim].data
    gradvel[new_dim] = gradvel[vec_dim].data
    sij = symmetrise(gradvel, [vec_dim, new_dim])
    if make_traceless:
        sij = traceless(sij, (vec_dim, new_dim))
    sij[new_dim] = c2
    sij.name = "rate-of-strain"
    sij.attrs["long_name"] = r"$S$"
    return sij


def vertical_heat_flux(
    vert_vel: xr.DataArray,
    pot_temperature: xr.DataArray,
    hor_axes: Collection[str],
) -> xr.DataArray:
    r"""compute vertical heat flux :math:`$w' \\theta'$` from :math:`w` and :math:`$\\theta$`

    :param vert_vel: vertical velocity field :math:`w`
    :param pot_temperature: potential temperature :math:`$\\theta$`

    :param hor_axes: labels of horizontal dimensions
        (w.r.t which to compute the fluctuations)
    """
    assert set(vert_vel.dims) == set(
        pot_temperature.dims
    ), "Mismatched dimensions of vert_vel and pot_temperature"
    w, theta = xr.align(
        vert_vel, pot_temperature, join="exact"
    )  # assert matching coordinates
    w_prime = w - w.mean(dim=hor_axes)
    theta_prime = theta - theta.mean(hor_axes)
    ans = w_prime * theta_prime
    ans.name = "vertical_heat_flux"
    ans.attrs["long_name"] = r"$w' \theta'$ "
    return ans


def Reynolds_fluct_stress(
    u: xr.DataArray,
    v: xr.DataArray,
    w: xr.DataArray,
    target_dims: list[str],
    fluctuation_axes: Collection[str],
) -> xr.DataArray:
    r"""compute Reynolds stress :math:`$\mathbf{u}'_i \mathbf{u}'_j$`

    :param u: velocity field component 1
    :param v: velocity field component 2
    :param w: velocity field component 3

    :param target_dims: axes on which the interpolate the stress --
        must be contained among the coordinates of ``u, v, w``
    :param fluctuation_axes: labels of dimensions
        w.r.t which to compute the fluctuations. Subset of ``target_dims``.

    Note: First performs an interpolation to ``target_dims`` and then computes the fluctuations
    w.r.t. ``fluctuation_axes``. There can be a commutation error when the
    interpolation happens along dimensions other than ``fluctuation_axes``.
    """
    # first interpolate
    vel = compose_vector_components_on_grid(
        [u, v, w], target_dims=target_dims, vector_dim="c1", drop_coords=True
    )
    # then take the fluctuations
    vel_prime = vel - vel.mean(dim=fluctuation_axes)
    vel_prime["c1"] = ["u'", "v'", "w'"]
    # take the outer product
    ans = tensor_self_outer_product(vel_prime)
    # add attributes
    ans.name = "Reynolds_fluct_stress"
    ans.attrs["long_name"] = r"$u'_i u'_j$"
    return ans


def Fluct_TKE(
    u: xr.DataArray,
    v: xr.DataArray,
    w: xr.DataArray,
    target_dims: list[str],
    fluctuation_axes: Collection[str],
) -> xr.DataArray:
    r"""compute fluctuating TKE :math:`$\mathbf{u}'_i \mathbf{u}'_i / 2$`

    :param u: velocity field component 1
    :param v: velocity field component 2
    :param w: velocity field component 3

    :param target_dims: axes on which the interpolate the stress --
        must be contained among the coordinates of ``u, v, w``
    :param fluctuation_axes: labels of dimensions
        w.r.t which to compute the fluctuations. Subset of ``target_dims``.

    Note: First performs an interpolation to ``target_dims`` and then computes the fluctuations
    w.r.t. ``fluctuation_axes`` and then square. There can be a commutation error when the
    interpolation happens along dimensions other than ``fluctuation_axes``.
    There is uncertainty from whether the interpolation happens before/after the squaring.
    """
    # first interpolate
    vel = compose_vector_components_on_grid(
        [u, v, w], target_dims=target_dims, vector_dim="c1", drop_coords=True
    )
    # then take the fluctuations
    vel_prime = vel - vel.mean(dim=fluctuation_axes)
    vel_prime["c1"] = ["u'", "v'", "w'"]
    # take the outer product
    ans = xr.dot(vel_prime, vel_prime, dim="c1") / 2
    # add attributes
    ans.name = "fluct_tke"
    ans.attrs["long_name"] = r"$u'_i u'_i / 2$"
    return ans

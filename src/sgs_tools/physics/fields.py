from typing import Callable, Collection, Iterable

import xarray as xr

from ..geometry.tensor_algebra import symmetrise, traceless
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
    return sij


def vertical_heat_flux(
    vert_vel: xr.DataArray, pot_temperature: xr.DataArray, hor_axes: Collection[str]
) -> xr.DataArray:
    """compute vertical heat flux :math:`$w' \\theta'$` from :math:`w` and :math:`$\\theta$`

    :param vert_vel: vertical velocity field :math:`w`
    :param pot_temperature: potential temperature :math:`$\\theta$`

    :param hor_axes: labels of horizontal dimensions
        (w.r.t which to compute the fluctuations)
    """
    w_prime = vert_vel - vert_vel.mean(dim=hor_axes)
    theta_prime = pot_temperature - pot_temperature.mean(hor_axes)
    ans = w_prime * theta_prime
    ans.name = "vertical_heat_flux"
    ans["long_name"] = r"$w' \theta'$ "
    return ans

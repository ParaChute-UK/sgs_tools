from typing import Sequence

import xarray as xr

from sgs_tools.geometry.tensor_algebra import tensor_self_outer_product
from sgs_tools.physics.fields import strain_from_vel
from sgs_tools.sgs.coarse_grain import CoarseGrain
from sgs_tools.sgs.filter import Filter


def momentum_stresses(
    vel: xr.DataArray,
    filter: Filter | CoarseGrain,
    vec_dim: str = "c1",
    new_dim: str = "c2",
    space_dims: Sequence[str] = ["x", "y", "z"],
) -> xr.Dataset:
    r"""compute a range of momemtum stresses at a scale defined by `filter`, namely
    the full filtered stress: `:math: \widetile { u v}`;
    the sgs stress: `:math: \tau = \widetile { u v} - \widetile { u} \widetilde{v}`;
    the Reynolds stress: `:math: \widetile { u' v'}`, where the "'" indicates
    fluctuation with respect to the filter, i.e. `:math: u' = u -  \widetile {u}`;
    the filtered rate-of-strain: `:math: S = (\widetilde{u}_{i,j} + \widetilde{u}_{j,i}) / 2`;

    :param vel: input velocity array (on collocated grid)
    :param filter: filtering operator that defines the `:math: \widetilde{}`
    :param vec_dim: dimension holding the vector components of vel -- will use as the first tensor dimension of the output
    :param new_dim: new dimension to use fo the second tensor dimension of the output. must not be present in vel
    """

    assert new_dim not in vel.dims
    assert vec_dim in vel.dims

    output = []
    # filt(v)
    vel_mean = filter.filter(vel).persist()

    # filt(v v)
    covariance = tensor_self_outer_product(vel, vec_dim=vec_dim, new_dim=new_dim)
    filtered = filter.filter(covariance)
    filtered.name = "filt_v_stress"
    filtered.attrs["long_name"] = r"$ \langle  u_i u_j \rangle $"
    output.append(filtered)

    # tau
    # filt(v) filt(v)
    resolved = tensor_self_outer_product(vel_mean, vec_dim=vec_dim, new_dim=new_dim)
    sgs = filtered - resolved
    sgs.name = "sgs_v_stress"
    sgs.attrs["long_name"] = r"$\tau$"
    output.append(sgs)

    # reynolds
    vel_prime = vel - vel_mean.reindex_like(
        vel, method="nearest"
    )  # up-sample vel_mean to the vel grid (in case Filter is a Coarse-graining)
    # relegate this to fluctuation function -- filter method?
    fluct_cov = tensor_self_outer_product(vel_prime, vec_dim=vec_dim, new_dim=new_dim)
    reynolds = filter.filter(fluct_cov)
    reynolds.name = "Reynolds_stress"
    reynolds.attrs["long_name"] = r"$ \langle u_i' u_j' \rangle $"
    output.append(reynolds)

    # strain at scale -- leave making it traceless to the client

    if all(vel_mean[x].size >= 2 for x in space_dims):
        rechunked = vel_mean.chunk({x: "auto" for x in space_dims})
        strain = strain_from_vel(
            rechunked,
            space_dims=space_dims,
            vec_dim=vec_dim,
            new_dim=new_dim,
            make_traceless=False,
        )
        output.append(strain)

    return xr.merge(output, compat="no_conflicts").chunk({vec_dim: -1, new_dim: -1})

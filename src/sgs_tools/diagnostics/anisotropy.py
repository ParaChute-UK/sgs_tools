from typing import Dict, Hashable

import xarray as xr
import xarray_einstats

from sgs_tools.geometry.tensor_algebra import anisotropy_renorm
from sgs_tools.sgs.coarse_grain import CoarseGrain
from sgs_tools.sgs.filter import Filter
from sgs_tools.sgs.sgs_stresses import momentum_stresses

name_dic: Dict[Hashable, Hashable] = {
    "filt_v_stress": "Filtered stress",
    "sgs_v_stress": "SGS stress",
    "Reynolds_stress": "Reynolds stress",
    "rate-of-strain": "Rate-of-strain",
}


def anisotropy_analysis(
    velocity: xr.DataArray,
    filt: Filter | CoarseGrain,
    vec_dim: str = "c1",
) -> xr.Dataset:
    tensor_dims = [vec_dim, vec_dim + "_1"]

    # rechunk along vector and filering dimensions
    vel = velocity.chunk({x: -1 for x in [vec_dim] + list(filt.filter_dims)})
    # collect tensors to be analysed
    tensors = momentum_stresses(vel, filt, *tensor_dims)
    # renormalise by trace
    ani_tensors = anisotropy_renorm(tensors, tensor_dims)
    # compute eigen values of anisotropy tensor
    # fill nans with 0: nans come from stagnation points/laminar flow
    # (v.v==0 or gradv.gradv == 0), so fill with zeroes
    # Note -- this expect symmetric matrices and take the Lower triangular part
    eigen_values = xarray_einstats.linalg.eigvalsh(
        ani_tensors.fillna(0),  # type: ignore
        tensor_dims,
        dask="parallelized",
    )
    # restore lost attributes frp, egvalsh computation
    for v in eigen_values:
        eigen_values[v].attrs = tensors[v].attrs
        eigen_values[v].name = tensors[v].name
        eigen_values[v].attrs["name"] = name_dic[tensors[v].name]
    return eigen_values  # type: ignore

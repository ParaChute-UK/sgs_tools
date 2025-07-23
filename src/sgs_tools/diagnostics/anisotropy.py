from typing import Dict, Hashable

import xarray as xr
import xarray_einstats

from sgs_tools.geometry.tensor_algebra import anisotropy_renorm
from sgs_tools.sgs.coarse_grain import CoarseGrain
from sgs_tools.sgs.filter import Filter
from sgs_tools.sgs.sgs_stresses import momentum_stresses
from sgs_tools.util.timer import timer

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

    # rechunk along vector dimensions
    vel = velocity.chunk({x: -1 for x in [vec_dim]}).persist()

    # with performance_report(filename = profile_path):
    # with client.get_task_stream() as task[f'{sim}_{flt_lbl}']:
    with timer("Filtering:"):
        tensors_view = momentum_stresses(vel, filt, *tensor_dims)
        # triggers async computation now, avoids DAG duplication
        tensors = tensors_view.persist()

    with timer("Anisotropy:"):
        ani_tensors = anisotropy_renorm(tensors, tensor_dims)
    with timer("e-values:"):
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
            eigen_values[v].attrs = tensors_view[v].attrs
            eigen_values[v].name = tensors_view[v].name
            eigen_values[v].attrs["name"] = name_dic[tensors_view[v].name]

        # triggers async computation now
        # evals = xr.merge(eigen_values).persist()
        evals = eigen_values.persist()
    return evals  # type: ignore

import numpy as np
import xarray as xr

# Vector calculus


def grad_scalar(
    sca: xr.DataArray, space_dims: list[str], new_dim_name: str = "c1", name=None
) -> xr.DataArray:
    """gradient vector of a scalar using centred 2nd order difference, reduced to 1st order at  boundaries

    :param sca: input scalar, with spatial coordinates `space_dims`
    :param space_dims: labels for the spatial dimensions (to be differentiated against)
    :param new_dim_name: the name of the new dimension (of the differential direction)
    :param name: name for output dataarray. If None, the name of the input scalar is used.
    :return: gradient (assuming cartesian geometry)
    """

    grad = []
    for dim in space_dims:
        grad.append(sca.differentiate(dim, edge_order=1))
    grad_xarr = xr.concat(
        grad, dim=xr.DataArray(range(1, len(space_dims) + 1), dims=[new_dim_name])
    )
    if name is not None:
        grad_xarr.name = name
    return grad_xarr


def grad_vector(
    vec: xr.DataArray, space_dims: list[str], new_dim_name: str = "c2", name=None
) -> xr.DataArray:
    """gradient tensor of a vector using centred 2nd order difference, reduced to 1st order at  boundaries

    :param vec: input vector, with spatial coordinates `space_dims`
    :param space_dims: labels for the spatial dimensions (to be differentiated against)
    :param new_dim_name: the name of the new dimension (of the differential direction)
    :param name: name for output dataarray. If None, the name of the input vector is used.
    :return: gradient (assuming cartesian geometry)
    """

    gradvec = []
    for dim in space_dims:
        gradvec.append(vec.differentiate(dim, edge_order=1))
    gradvec_xarr = xr.concat(
        gradvec, dim=xr.DataArray(range(1, len(space_dims) + 1), dims=[new_dim_name])
    )
    if name is not None:
        gradvec_xarr.name = name
    return gradvec_xarr


def grad_vector_lin(
    vec: xr.DataArray, space_dims: list[str], new_dim_name: str = "c2", name=None
) -> xr.DataArray:
    """gradient tensor of a vector -- 1st order backward finite-difference

    :param vec: input vector, with spatial coordinates `space_dims`
    :param space_dims: labels for the spatial dimensions (to be differentiated against)
    :param new_dim_name: the name of the new dimension (of the differential direction)
    :param name: name for output dataarray. If None, the name of the input vector is used.
    :return: gradient (assuming cartesian geometry)
    """
    gradvec = []
    for dim in space_dims:
        coord = vec[dim].astype(float)  # just in case
        val = vec.astype(float)

        gradvec.append(
            (val - val.shift({dim: -1}, fill_value=np.nan))
            / (coord - coord.shift({dim: -1}, fill_value=np.nan))
        )
    gradvec_xarr = xr.concat(
        gradvec, dim=xr.DataArray(range(1, len(space_dims) + 1), dims=[new_dim_name])
    )
    if name is not None:
        gradvec_xarr.name = name
    return gradvec_xarr

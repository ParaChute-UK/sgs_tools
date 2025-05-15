import xarray as xr

from .filter import Filter
from .sgs_model import DynamicSGSModel


def LillyMinimisation(
    model: DynamicSGSModel,
    filter: Filter,
    filter_regularize: Filter,
    contraction_dims: list[str],
) -> xr.DataArray:
    """Compute dynamic coefficient using Germano identity as :math:`$\overline{L M} / \overline{M M}$`.
        where :math:`$\overline{*}$` means regularisation

    "param model: Dynamic SGS model used for computing the Model :math:`M` and Leonard :math:`L` tensors
    :param filter: Filter used by the SGS model
    :param filter_regularize: Filter used to regularize the coefficient calculation
    :param contraction_dims: labels of dimensions to be contracted to form LM and MM tensors/scalars. Must have at least one.
    """
    L = model.Leonard_tensor(filter)
    M = model.M_Germano_tensor(filter)

    MM = xr.dot(M, M, dim=contraction_dims)
    LM = xr.dot(L, M, dim=contraction_dims)
    filt_LM = filter_regularize.filter(LM)
    filt_MM = filter_regularize.filter(MM)

    coeff = filt_LM / filt_MM
    return coeff

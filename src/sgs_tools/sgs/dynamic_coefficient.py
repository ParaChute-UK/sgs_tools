import warnings

import dask.array as da
import numpy as np
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


def LinComb2ModelLillyMinimisation(
    model1: DynamicSGSModel,
    model2: DynamicSGSModel,
    filter: Filter,
    filter_regularize: Filter,
    contraction_dims: list[str],
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute dynamic coefficients of a 2-component models using Germano identity as :math:`$L = C1 M1 + C2 M2$`.
        using regularized least-square minimisation (inverting the {M_i M_j} matrix analytically)

    :param model1, model2: Dynamic SGS models used for computing the Model :math:`M` and Leonard :math:`L` tensors.
                   Both models must have the same Leonard tensors (unchecked). Will use the one from model 1.
    :param filter: Filter used for the SGS model
    :param filter_regularize: Filter used to regularize the coefficient calculation
    :param contraction_dims: labels of dimensions to be contracted to form LM and MM tensors/scalars. Must have at least one.
    """
    # TODO: turn this into an assert check
    warnings.warn(
        "Warning: No check that all input models have the same leonard tensor!"
    )
    L = model1.Leonard_tensor(filter)
    M1 = model1.M_Germano_tensor(filter)
    M2 = model2.M_Germano_tensor(filter)
    # Filtered Leonard contractions
    LM1 = filter_regularize.filter(xr.dot(L, M1, dim=contraction_dims, optimize=True))
    LM2 = filter_regularize.filter(xr.dot(L, M2, dim=contraction_dims, optimize=True))
    # Model matrix
    M11 = filter_regularize.filter(xr.dot(M1, M1, dim=contraction_dims, optimize=True))
    M12 = filter_regularize.filter(xr.dot(M1, M2, dim=contraction_dims, optimize=True))
    M22 = filter_regularize.filter(xr.dot(M2, M2, dim=contraction_dims, optimize=True))
    # Model determinant
    detM = M11 * M22 - M12**2
    # the adjoint matrix  = inverse * detM
    # | M22 -M12 |
    # |-M12  M11 |

    # coeff = M_inv @ LM
    coeff1 = (M22 * LM1 - M12 * LM2) / detM
    coeff2 = (-M12 * LM1 + M11 * LM2) / detM
    return coeff1, coeff2


def LinComb3ModelLillyMinimisation(
    model1: DynamicSGSModel,
    model2: DynamicSGSModel,
    model3: DynamicSGSModel,
    filter: Filter,
    filter_regularize: Filter,
    contraction_dims: list[str],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Compute dynamic coefficients of a 3-component models using Germano identity as :math:`$L = C1 M1 + C2 M2 + C3 M3$`.
       using regularized least-square minimisation (inverting the {M_i M_j} matrix explicitly)

    :param model1, model2, model3: Dynamic SGS models used for computing the Model :math:`M` and Leonard :math:`L` tensors.
                   All models must have the same Leonard tensors (unchecked). Will use the one from model 1.
    "param models: List of dynamic SGS models used for computing the Model :math:`M` and Leonard :math:`L` tensors
    :param filter: Filter used by the SGS model
    :param filter_regularize: Filter used to regularize the coefficient calculation
    :param contraction_dims: labels of dimensions to be contracted to form LM and MM tensors/scalars. Must have at least one.
    """
    # TODO: turn this into an assert check
    warnings.warn(
        "Warning: No check that all input models have the same leonard tensor!"
    )

    L = model1.Leonard_tensor(filter)
    M1 = model1.M_Germano_tensor(filter)
    M2 = model2.M_Germano_tensor(filter)
    M3 = model3.M_Germano_tensor(filter)
    # Filtered Leonard contractions
    LM1 = filter_regularize.filter(xr.dot(L, M1, dim=contraction_dims, optimize=True))
    LM2 = filter_regularize.filter(xr.dot(L, M2, dim=contraction_dims, optimize=True))
    LM3 = filter_regularize.filter(xr.dot(L, M3, dim=contraction_dims, optimize=True))

    # Model matrix
    M11 = filter_regularize.filter(xr.dot(M1, M1, dim=contraction_dims, optimize=True))
    M12 = filter_regularize.filter(xr.dot(M1, M2, dim=contraction_dims, optimize=True))
    M13 = filter_regularize.filter(xr.dot(M1, M3, dim=contraction_dims, optimize=True))
    M22 = filter_regularize.filter(xr.dot(M2, M2, dim=contraction_dims, optimize=True))
    M23 = filter_regularize.filter(xr.dot(M2, M3, dim=contraction_dims, optimize=True))
    M33 = filter_regularize.filter(xr.dot(M3, M3, dim=contraction_dims, optimize=True))

    # Model determinant
    detM = (
        M11 * M22 * M33
        + 2 * M12 * M23 * M13
        - M12**2 * M33
        - M13**2 * M22
        - M23**2 * M11
    )
    # adjoint matrix = inverse * detM
    AdjM11 = M22 * M33 - M23**2
    AdjM12 = M13 * M23 - M12 * M33
    AdjM13 = M12 * M23 - M13 * M22
    AdjM22 = M11 * M33 - M13**2
    AdjM23 = M12 * M13 - M11 * M23
    AdjM33 = M11 * M22 - M12**2
    # contracting with the leonard vector and filtering numerator
    coeff1 = (AdjM11 * LM1 + AdjM12 * LM2 + AdjM13 * LM3) / detM
    coeff2 = (AdjM12 * LM1 + AdjM22 * LM2 + AdjM23 * LM3) / detM
    coeff3 = (AdjM13 * LM1 + AdjM23 * LM2 + AdjM33 * LM3) / detM

    return coeff1, coeff2, coeff3


def LinCombModelLillyMinimisation(
    models: list[DynamicSGSModel],
    filter: Filter,
    filter_regularize: Filter,
    contraction_dims: list[str],
) -> xr.DataArray:
    """Compute dynamic coefficients of a general milti-component model using Germano identity as :math:`$L = \Sum _i C_ M_i$`.
        using regularized least-square minimisation (solving the linear system numerically with `np.linalg.solve`)

    :param models: List of dynamic SGS models used for computing the Model :math:`M` and Leonard :math:`L` tensors.
                   Both models must have the same Leonard tensors (unchecked). Will use the one from model 1.
    :param filter: Filter used by the SGS model
    :param filter_regularize: Filter used to regularize the coefficient calculation
    :param contraction_dims: labels of dimensions to be contracted to form LM and MM tensors/scalars. Must have at least one.
    """
    # TODO: turn this into an assert check
    warnings.warn(
        "Warning: No check that all input models have the same leonard tensor!"
    )

    # consider memory consumption for M
    L = models[0].Leonard_tensor(filter)
    M_list = [m.M_Germano_tensor(filter) for m in models]
    M = xr.concat(M_list, dim="coeff_dim1")
    # Filtered Leonard contractions
    LM = filter_regularize.filter(xr.dot(L, M, dim=contraction_dims, optimize=True))
    # Filtered Model-Leonard contractions
    MM = filter_regularize.filter(
        xr.dot(
            M,
            M.rename({"coeff_dim1": "coeff_dim2"}),
            dim=contraction_dims,
            optimize=True,
        )
    )

    # FIXME: can't get the distributed xarray API to handle this trivially
    # so go down to dask mapping of numpy.linalg.solve -> need to reorder axes
    LM = LM.transpose(..., "coeff_dim1")
    LM_expanded = LM.expand_dims(dim="rhs", axis=-1)
    MM = MM.transpose(..., "coeff_dim1", "coeff_dim2")
    # mm_condition = (
    #     da.map_blocks(
    #         np.linalg.cond,
    #         MM,
    #         p=None, # order of the norm
    #         dtype=MM.data.dtype,
    #     )
    #     .compute()
    #     .max()
    # )

    mm_condition = (
        xr.apply_ufunc(
            np.linalg.cond,
            MM,
            kwargs={"p": None},  # order of the norm
            input_core_dims=[["coeff_dim1", "coeff_dim2"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[MM.dtype],
        )
        .compute()
        .max()
    )

    if mm_condition > 1e15:
        s = f"Warning: Large condtion number max={mm_condition:g} for the MM tensor. May degrade accuracy of coefficients"
        warnings.warn(s)

    coefficients = xr.apply_ufunc(
        np.linalg.solve,
        MM,
        LM_expanded,
        input_core_dims=[["coeff_dim1", "coeff_dim2"], ["coeff_dim1", "rhs"]],
        output_core_dims=[["coeff_dim1", "rhs"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[LM.dtype],
    )

    return coefficients.squeeze("rhs")

    # rewrap with Xarray metadata
    # Note: need to make this eager, otherwise get a 0-size object from da.map_blocks
    # coefficients = da.map_blocks(np.linalg.solve, MM, LM, dtype=MM.data.dtype)
    # xr.DataArray(coefficients.compute().squeeze('rhs'), dims=LM.dims, coords=LM.coords)

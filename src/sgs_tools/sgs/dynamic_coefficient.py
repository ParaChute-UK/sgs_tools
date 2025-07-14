import warnings
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import xarray as xr

from ..util.dask_opt_util import dask_layered
from .filter import Filter


class Minimisation(Protocol):
    r"""
    Protocol for solving the over-determined tensor equation
    :math:`L = \sum_i c_i M_i`, where `L` and `M_i` are tensors and
    :math:`c_i` are scalar coefficients to be computed.

    :ivar contraction_dims: Names of the dimensions to contract when forming the
        tensor products :math:`L M_i` and :math:`M_i M_j`.
    :ivar coeff_dim: Dimension label along which the resulting coefficients
        :math:`c_i` are concatenated.
    """

    @property
    def reg_filter(self) -> Filter: ...

    @property
    def contraction_dims(self) -> Sequence[str]: ...

    @property
    def coeff_dim(self) -> str: ...

    def compute(
        self, L: xr.DataArray, Mi: Sequence[xr.DataArray], reg_filter: Filter
    ) -> xr.DataArray:
        r"""solve for :math:{c_i}` the over-determined system :math:`L = \sum_i(c_i M_i)`.

        :param L: LHS tensor
        :param M: a sequence of RHS tensors
        :param reg_filter: Filter used to regularize the contracted tensor products.
        """


@dataclass(frozen=True)
class LillyMinimisation1Model:
    r"""Lilly Minimisation (least square error) for a 1-global-coefficient model using
       the Lilly identity as :math:`$\overline{L \cdot M} / \overline{M \cdot M}$`.
       where :math:`$\cdot$`means  tensor contraction, :math:`$\overline{*}$` means regularisation filtering

    :param contraction_dims: labels of dimensions to be contracted to form :math:`L M_i `and :math:`M_i M_j` products.
    :param coeff_dim: label of dimension along which to concatenate the arrays :math:`c_i`
    """

    contraction_dims: Sequence[str]
    coeff_dim: str

    @dask_layered("LillyMinimisation1Model")
    def compute(
        self, L: xr.DataArray, Mi: Sequence[xr.DataArray], reg_filter: Filter
    ) -> xr.DataArray:
        """Compute :math:`$\overline{L\cdot M} / \overline{M \cdot M}$`.
        where :math:`$\overline{*}$` means regularisation filtering

        :param L: LHS tensor
        :param M: a sequence of *1* RHS tensors
        :param reg_filter: Filter used to regularize the contracted tensor products.
        """
        assert len(Mi) == 1
        assert all(t in L.dims for t in self.contraction_dims)
        assert all(t in Mi[0].dims for t in self.contraction_dims)

        MM = xr.dot(Mi[0], Mi[0], dim=self.contraction_dims)
        LM = xr.dot(L, Mi[0], dim=self.contraction_dims)
        filt_LM = reg_filter.filter(LM)
        filt_MM = reg_filter.filter(MM)

        coeff = filt_LM / filt_MM
        return coeff


@dataclass(frozen=True)
class LillyMinimisation2Model:
    r"""Lilly Minimisation (least square error) for a 2-coefficient model using
       the Lilly identity as :math:`$L = \sum_i^2 c_i M_i$`.

    :param contraction_dims: labels of dimensions to be contracted to form :math:`L M_i `and :math:`M_i M_j` products.
    :param coeff_dim: label of dimension along which to concatenate the arrays :math:`c_i`
    """

    contraction_dims: Sequence[str]
    coeff_dim: str

    @dask_layered("LillyMinimisation2Model")
    def compute(
        self, L: xr.DataArray, Mi: Sequence[xr.DataArray], reg_filter: Filter
    ) -> xr.DataArray:
        """Compute dynamic coefficients of a 2-component models using Germano identity as :math:`$L = C1 M1 + C2 M2$`.
           using regularized least-square minimisation (inverting the :math:`$\overline{M_i M_j}$` matrix analytically)

        :param L: LHS tensor
        :param M: a sequence of *2* RHS tensors
        :param reg_filter: Filter used to regularize the contracted tensor products.
        """
        assert len(Mi) == 2
        assert all(t in L.dims for t in self.contraction_dims)
        for Mi_tensor in Mi:
            assert all(t in Mi_tensor.dims for t in self.contraction_dims)

        # Filtered Leonard contractions
        LM1 = reg_filter.filter(
            xr.dot(L, Mi[0], dim=self.contraction_dims, optimize=True)
        )
        LM2 = reg_filter.filter(
            xr.dot(L, Mi[1], dim=self.contraction_dims, optimize=True)
        )
        # Model matrix
        M11 = reg_filter.filter(
            xr.dot(Mi[0], Mi[0], dim=self.contraction_dims, optimize=True)
        )
        M12 = reg_filter.filter(
            xr.dot(Mi[0], Mi[1], dim=self.contraction_dims, optimize=True)
        )
        M22 = reg_filter.filter(
            xr.dot(Mi[1], Mi[1], dim=self.contraction_dims, optimize=True)
        )
        # Model determinant
        detM = M11 * M22 - M12**2
        # the adjoint matrix  = inverse * detM
        # | M22 -M12 |
        # |-M12  M11 |

        # coeff = M_inv @ LM
        coeff1 = (M22 * LM1 - M12 * LM2) / detM
        coeff2 = (-M12 * LM1 + M11 * LM2) / detM
        assert self.coeff_dim not in coeff1.dims, "Avoid collision in concat dim"
        return xr.concat([coeff1, coeff2], dim=self.coeff_dim)


@dataclass(frozen=True)
class LillyMinimisation3Model:
    r"""Lilly Minimisation (least square error) for a 3-coefficient model using
       the Lilly identity as :math:`$L = \sum_i^3 c_i M_i$`.

    :param contraction_dims: labels of dimensions to be contracted to form :math:`L M_i `and :math:`M_i M_j` products.
    :param coeff_dim: label of dimension along which to concatenate the arrays :math:`c_i`
    """

    contraction_dims: Sequence[str]
    coeff_dim: str

    @dask_layered("LillyMinimisation3Model")
    def compute(
        self, L: xr.DataArray, Mi: Sequence[xr.DataArray], reg_filter: Filter
    ) -> xr.DataArray:
        """Compute dynamic coefficients of a 3-component models using Germano identity as :math:`$L = C1 M1 + C2 M2 + C3 M3$`.
        using regularized least-square minimisation (inverting the {M_i M_j} matrix explicitly)

        :param L: LHS tensor
        :param M: a sequence of *3* RHS tensors
        :param reg_filter: Filter used to regularize the contracted tensor products.
        """
        assert len(Mi) == 3
        assert all(t in L.dims for t in self.contraction_dims)
        for Mi_tensor in Mi:
            assert all(t in Mi_tensor.dims for t in self.contraction_dims)

        # Filtered Leonard contractions
        LM1 = reg_filter.filter(
            xr.dot(L, Mi[0], dim=self.contraction_dims, optimize=True)
        )
        LM2 = reg_filter.filter(
            xr.dot(L, Mi[1], dim=self.contraction_dims, optimize=True)
        )
        LM3 = reg_filter.filter(
            xr.dot(L, Mi[2], dim=self.contraction_dims, optimize=True)
        )

        # Model matrix
        M11 = reg_filter.filter(
            xr.dot(Mi[0], Mi[0], dim=self.contraction_dims, optimize=True)
        )
        M12 = reg_filter.filter(
            xr.dot(Mi[0], Mi[1], dim=self.contraction_dims, optimize=True)
        )
        M13 = reg_filter.filter(
            xr.dot(Mi[0], Mi[2], dim=self.contraction_dims, optimize=True)
        )
        M22 = reg_filter.filter(
            xr.dot(Mi[1], Mi[1], dim=self.contraction_dims, optimize=True)
        )
        M23 = reg_filter.filter(
            xr.dot(Mi[1], Mi[2], dim=self.contraction_dims, optimize=True)
        )
        M33 = reg_filter.filter(
            xr.dot(Mi[2], Mi[2], dim=self.contraction_dims, optimize=True)
        )

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

        assert self.coeff_dim not in coeff1.dims, "Avoid collision in concat dim"
        return xr.concat([coeff1, coeff2, coeff3], dim=self.coeff_dim)


@dataclass(frozen=True)
class LillyMinimisationNModel:
    r"""Lilly Minimisation (least square error) for an N-coefficient model using
       the Lilly identity as :math:`$L = \sum_i^N c_i M_i$`.

    :param contraction_dims: labels of dimensions to be contracted to form :math:`L M_i `and :math:`M_i M_j` products.
    :param coeff_dim: label of dimension along which to concatenate the arrays :math:`c_i`
    """

    contraction_dims: Sequence[str]
    coeff_dim: str

    @dask_layered("LillyMinimisationNModel")
    def compute(
        self, L: xr.DataArray, Mi: Sequence[xr.DataArray], reg_filter: Filter
    ) -> xr.DataArray:
        """Solve the system  :math:`$\overline{L \cdot M_i} = \sum_i c_j \overline{M_i \cdot \M_j}$`
        using np.linalg.SVD, where :math:`L \cdot M_i` and :math:`M_i \cdot M_j` are scalar fields

        :param L: LHS tensor
        :param M: a sequence of RHS tensors
        :param reg_filter: Filter used to regularize the contracted tensor products.
        """
        assert all(t in L.dims for t in self.contraction_dims)
        for M in Mi:
            assert all(t in M.dims for t in self.contraction_dims)

        # consider memory consumption for M
        M = xr.concat(Mi, dim=self.coeff_dim)
        # Filtered Leonard contractions
        LM = reg_filter.filter(xr.dot(L, M, dim=self.contraction_dims, optimize=True))
        # Filtered Model-Leonard contractions
        MM = reg_filter.filter(
            xr.dot(
                M,
                M.rename({self.coeff_dim: self.coeff_dim + "_dummy"}),
                dim=self.contraction_dims,
                optimize=True,
            )
        )

        LM = LM.transpose(..., self.coeff_dim)
        LM_expanded = LM.expand_dims(dim="rhs", axis=-1)
        MM = MM.transpose(..., self.coeff_dim, self.coeff_dim + "_dummy")

        mm_condition = (
            xr.apply_ufunc(
                np.linalg.cond,
                MM.chunk({self.coeff_dim: -1, self.coeff_dim + "_dummy": -1}),
                kwargs={"p": None},  # order of the norm
                input_core_dims=[[self.coeff_dim, self.coeff_dim + "_dummy"]],
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
            MM.chunk({self.coeff_dim: -1, self.coeff_dim + "_dummy": -1}),
            LM_expanded.chunk({self.coeff_dim: -1, "rhs": -1}),
            input_core_dims=[
                [self.coeff_dim, self.coeff_dim + "_dummy"],
                [self.coeff_dim, "rhs"],
            ],
            output_core_dims=[[self.coeff_dim, "rhs"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[LM.dtype],
        )

        return coefficients.squeeze("rhs")

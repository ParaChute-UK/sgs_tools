import numpy as np
import xarray as xr  # only used for type hints
from sgs_tools.sgs.dynamic_coefficient import (
    LillyMinimisation1Model,
    LillyMinimisation2Model,
    LillyMinimisation3Model,
    LillyMinimisationNModel,
)
from sgs_tools.sgs.filter import IdentityFilter


def random():
    arr = np.random.rand(30, 30, 30, 3, 3)
    arr = (arr + arr.transpose(0, 1, 2, 4, 3)) / 2
    return xr.DataArray(
        arr, dims=["x", "y", "z", "c1", "c2"], coords={"c1": [1, 2, 3], "c2": [1, 2, 3]}
    )


def linear():
    arr = np.einsum(
        "i,j,k,l,m -> ijklm",
        np.arange(1, 31),
        np.arange(1, 31),
        np.arange(1, 31),
        np.arange(1, 4),
        np.arange(1, 4),
        dtype=float,
    )
    return xr.DataArray(
        arr, dims=["x", "y", "z", "c1", "c2"], coords={"c1": [1, 2, 3], "c2": [1, 2, 3]}
    )


def test_LillyMinimisation():
    scale = np.random.rand()
    id = IdentityFilter(None, ["x", "y"])
    min = LillyMinimisation1Model(["c1", "c2"], "cdim")
    for arr in random(), linear():
        L = scale * arr
        M = arr
        coeff = min.compute(L, [M], id)
        assert np.allclose(scale, coeff.squeeze().data)


def test_LinComb2ModelLillyMinimisation():
    id = IdentityFilter(None, ["x", "y"])
    min = LillyMinimisation2Model(["c1", "c2"], "cdim")
    for scale1, scale2 in [(1, 0), (3, 3)]:
        for arr in random(), linear():
            L = scale1 * arr + scale2 * arr**2
            M = [arr, arr**2]
            coeff = min.compute(L, M, id)
            assert np.allclose(
                scale1, coeff.isel(cdim=0)
            ), f"scales: {scale1}, {scale2}, coeff mean: {np.mean(coeff.isel(cdim=0))}, std: {np.std(coeff.isel(cdim=0))}"
            assert np.allclose(
                scale2, coeff.isel(cdim=1)
            ), f"scales: {scale2}, {scale2}, coeff mean: {np.mean(coeff.isel(cdim=1))}, std: {np.std(coeff.isel(cdim=1))}"


def LinComb3ModelLillyMinimisation():
    id = IdentityFilter(None, ["x", "y"])
    min = LillyMinimisation3Model(["c1", "c2"], "cdim")
    for scale1, scale2, scale3 in [
        np.random.rand(3),
        (1, 1, 0),
        (1, 2, 0),
        (1, 0, 0),
    ]:
        for arr in random(), linear():
            L = scale1 * arr + scale2 * arr**2 + scale3 * arr**3
            M = [arr, arr**2, arr**3]
            coeff = min.compute(L, M, id)
            assert np.allclose(
                scale1, coeff.isel(cdim=0)
            ), f"scale1: {scale1}, coeff mean: {np.mean(coeff.isel(cdim=0))}, std: {np.std(coeff.isel(cdim=0))}"
            assert np.allclose(
                scale2, coeff.isel(cdim=1)
            ), f"scale1: {scale2}, coeff mean: {np.mean(coeff.isel(cdim=1))}, std: {np.std(coeff.isel(cdim=1))}"
            assert np.allclose(
                scale3, coeff.isel(cdim=2)
            ), f"scale1: {scale3}, coeff mean: {np.mean(coeff.isel(cdim=2))}, std: {np.std(coeff.isel(cdim=2))}"


def test_LinCombModelLillyMinimisation():
    id = IdentityFilter(None, ["x", "y"])
    min = LillyMinimisationNModel(["c1", "c2"], "cdim")
    for n_mod in range(1, 4):
        scale = np.random.rand(n_mod)
        for arr in random(), linear():
            M = [arr ** (i + 1) for i in range(n_mod)]
            M = [
                M[i] / M[i].mean() for i in range(n_mod)
            ]  # improve matrix conditioning
            L = scale[0] * M[0]
            for i in range(1, n_mod):
                L += scale[i] * M[i]
            coeff = min.compute(L, M, id)
            assert np.allclose(scale, coeff)

from dataclasses import dataclass

import numpy as np
import xarray as xr  # only used for type hints
from sgs_tools.sgs.dynamic_coefficient import (
    LillyMinimisation,
    LinComb2ModelLillyMinimisation,
    LinComb3ModelLillyMinimisation,
    LinCombModelLillyMinimisation,
)
from sgs_tools.sgs.filter import Filter, IdentityFilter
from sgs_tools.sgs.sgs_model import DynamicSGSModel
from xarray.core.types import T_Xarray


@dataclass(frozen=True)
class DynamicTestModel(DynamicSGSModel):
    Marray: T_Xarray
    Larray: T_Xarray

    def M_Germano_tensor(self, filter: Filter) -> xr.DataArray:
        assert isinstance(filter, IdentityFilter)
        return filter.filter(self.Marray)

    def Leonard_tensor(self, filter: Filter) -> xr.DataArray:
        assert isinstance(filter, IdentityFilter)
        return filter.filter(self.Larray)


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
    id = IdentityFilter(None, ["x", "y"])
    scale = np.random.rand()
    for arr in random(), linear():
        vm = DynamicTestModel(None, arr, scale * arr)
        coeff = LillyMinimisation(vm, id, id, ["c1", "c2"])
        assert np.allclose(scale, coeff.squeeze().data)


def test_LinComb2ModelLillyMinimisation():
    id = IdentityFilter(None, ["x", "y"])
    for scale1, scale2 in [np.random.rand(2), (1, 0), (3, 3)]:
        for arr in random(), linear():
            leonard = scale1 * arr + scale2 * arr**2
            vm1 = DynamicTestModel(None, arr, leonard)
            vm2 = DynamicTestModel(None, arr**2, leonard)

            c1, c2 = LinComb2ModelLillyMinimisation(vm1, vm2, id, id, ["c1", "c2"])
            assert np.allclose(scale1, c1)
            assert np.allclose(scale2, c2)


def LinComb3ModelLillyMinimisation():
    id = IdentityFilter(None, ["x", "y"])
    for scale1, scale2, scale3 in [
        np.random.rand(3),
        (1, 1, 0),
        (1, 2, 0),
        (1, 0, 0),
    ]:
        for arr in random(), linear():
            leonard = scale1 * arr + scale2 * arr**2 + scale3 * arr**3
            vm1 = DynamicTestModel(None, arr, leonard)
            vm2 = DynamicTestModel(None, arr**2, leonard)
            vm3 = DynamicTestModel(None, arr**3, leonard)

            c1, c2, c3 = LinComb3ModelLillyMinimisation(
                vm1, vm2, vm3, id, id, ["c1", "c2"]
            )
            assert np.allclose(
                scale1, c1
            ), f"scale1: {scale1}, {scale2}, {scale3}, {np.mean(c1)}, {np.std(c1)}"
            assert np.allclose(scale2, c2), f"scale2: {scale2}"
            assert np.allclose(scale3, c3), f"scale3: {scale3}"


def test_LinCombModelLillyMinimisation():
    id = IdentityFilter(None, ["x", "y"])
    for n_mod in range(1, 4):
        scale = np.random.rand(n_mod)
        for arr in random(), linear():
            m = [arr ** (i + 1) for i in range(n_mod)]
            m = [
                m[i] / m[i].mean() for i in range(n_mod)
            ]  # improve matrix conditioning
            leonard_n = scale[0] * m[0]
            for i in range(1, n_mod):
                leonard_n += scale[i] * m[i]
            vm = [DynamicTestModel(None, m[i], leonard_n) for i in range(n_mod)]
            coeff = LinCombModelLillyMinimisation(vm, id, id, ["c1", "c2"])
            assert np.allclose(scale, coeff)

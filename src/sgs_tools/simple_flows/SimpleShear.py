from collections.abc import Iterable
from dataclasses import dataclass

import xarray as xr

from ..geometry.grid import CoordScalar, Grid


@dataclass(frozen=True)
class SimpleShear:
    """prescribe velocity for a simple shear flow on a general grid
        roughly: product(``amplitudes [i] * x[i]`` for i in ``dimensions``)
        where ``x[i]`` is the coordinate in the i'th direction

    :ivar grid: grid which provices coordinates
    :ivar dimensions: labels of shearing directions
    :ivar velcomp: 0-indexed component of velocity vector that is sheared
      (all others are set to 0)
    :ivar amplitudes: amplification factor of gradient along `dimensions`
    """

    grid: Grid
    dimensions: Iterable[str]
    velcomp: int
    amplitudes: list[float]

    def velocity(self) -> xr.DataArray:
        """produce a velocity field"""
        v_shear = xr.DataArray(1.0)
        for i, d in enumerate(self.dimensions):
            scalar_gdt = ScalarGradient(self.grid, d, self.amplitudes[i], 0.0)
            v_shear = v_shear * scalar_gdt.field()
        v_zero = xr.zeros_like(v_shear)
        vel = xr.concat([v_shear, v_zero, v_zero], dim="c1")
        return vel.roll(shifts={"c1": self.velcomp})


@dataclass(frozen=True)
class ScalarGradient:
    """prescribe a scalar field with a constant gradient on a cartesian mesh

    :ivar grid: grid which provices coordinates
    :ivar dimension: label of direcion of coordinate
    :ivar gdt: value of gradient
    :ivar offset: reference value at coordinate origin
    """

    grid: Grid
    dimension: str
    gdt: float
    offset: float = 0.0

    def field(self) -> xr.DataArray:
        """produce a scalar field with a constant gradient"""
        coord = CoordScalar(self.grid, self.dimension, self.gdt)
        coord_array = coord.scalar()
        coord_array += self.offset - coord_array.isel({self.dimension: 0})
        return coord_array

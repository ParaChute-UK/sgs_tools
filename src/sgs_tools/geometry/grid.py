from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from xarray import DataArray


@dataclass(frozen=True)
class Grid(ABC):
    """Abstract base grid object.

    * :meth:`mesh`: return a dictionary of type {axis_name : `np.meshgrid` of coordinate}
    * :meth:`coords`: return a dictionary of type {axis_name : 1d coordinate values}
    """

    @abstractmethod
    def mesh(self, shape: list[int]) -> dict[str, NDArray]:
        """:returns:  a dictionary of type {axis_name : `np.meshgrid` of coordinate}"""
        ...

    @abstractmethod
    def coords(self, shape: list[int]) -> dict[str, NDArray]:
        """:returns:  a dictionary of type {axis_name : 1d coordinate values}"""
        ...


@dataclass(frozen=True)
class UniformCartesianGrid:
    """Uniform Cartesian Grid. Dimension is infered by length of delta.

    :ivar origin: origin of grid
    :ivar delta: step size
    """

    origin: list[int]
    delta: list[int]

    def coords(self, shape: list[int]) -> dict[str, NDArray]:
        coords = {}
        for i in range(len(self.delta)):
            coords[f"x{i + 1}"] = (
                np.linspace(0, self.delta[i] * (shape[i] - 1), shape[i])
                + self.origin[i]
            )
        return coords

    def mesh(self, shape: list[int]) -> dict[str, NDArray]:
        coords = self.coords(shape)
        lbls = coords.keys()
        axes = coords.values()

        coord_mesh = np.meshgrid(*axes, indexing="ij")
        return dict(zip(lbls, coord_mesh))


@dataclass(frozen=True)
class CoordScalar:
    """prescribe a scalar field that is a grid coordinate
    scaled with a constant amplitude

    :ivar grid: grid which provides coordinate variables
    :ivar dimension: label of coordinate direction
    :ivar amplitude: factor by which to scale the coordinate
    """

    grid: Grid
    dimension: str  # coordinate direction
    amplitude: float  # scaling of scalar (multiplying by the coordinate)

    def scalar(self, shape: list[int]) -> DataArray:
        coords = self.grid.coords(shape)
        scalar = self.amplitude * self.grid.mesh(shape)[self.dimension]
        return DataArray(scalar, dims=list(coords.keys()), coords=coords)

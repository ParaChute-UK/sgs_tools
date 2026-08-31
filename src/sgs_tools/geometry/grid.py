from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from xarray import DataArray


@dataclass(frozen=True)
class Grid(ABC):
    """Abstract base grid object.

    * :meth:`mesh`: return a dictionary of type
       {axis_name : `np.meshgrid` of coordinate}
    * :meth:`coords`: return a dictionary of type {axis_name : 1d coordinate values}
    """

    @abstractmethod
    def mesh(self) -> dict[str, NDArray]:
        """:returns:  a dictionary of type {axis_name : `np.meshgrid` of coordinate}"""
        ...

    @abstractmethod
    def coords(self) -> dict[str, NDArray]:
        """:returns:  a dictionary of type {axis_name : 1d coordinate values}"""
        ...


@dataclass(frozen=True)
class UniformCartesianGrid(Grid):
    """Uniform Cartesian Grid. Dimension is inferred by length of delta.

    :ivar origin: origin of grid
    :ivar delta: step size
    :ivar shape: shape of grid, i.e. number of points in each direction
    """

    origin: list[int]
    delta: list[int]
    shape: list[int]

    def coords(self) -> dict[str, NDArray]:
        coords = {}
        for i in range(len(self.delta)):
            coords[f"x{i + 1}"] = (
                np.linspace(0, self.delta[i] * (self.shape[i] - 1), self.shape[i])
                + self.origin[i]
            )
        return coords

    def mesh(self) -> dict[str, NDArray]:
        coords = self.coords()
        lbls = coords.keys()
        axes = coords.values()

        coord_mesh = np.meshgrid(*axes, indexing="ij")
        return dict(zip(lbls, coord_mesh, strict=False))


@dataclass(frozen=True)
class VaryingCartesianGrid(Grid):
    """Cartesian grid with varying spacing in each direction.

    Coordinates are specified directly as 1-D arrays; ``delta`` (the
    per-axis spacing arrays) is computed from ``axes`` at construction
    and cached as a frozen field.

    :ivar coords: sequence of 1-D coordinate arrays, one per spatial axis.
    :ivar delta: tuple of spacing arrays ``np.diff(ax)`` for each axis,
      set automatically — do not pass on construction.
    """

    coordinates: tuple[NDArray, ...]
    delta: tuple[NDArray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta", tuple(np.diff(ax) for ax in self.coordinates))

    @property
    def shape(self):
        return (len(c) for c in self.coordinates)

    def coords(self) -> dict[str, NDArray]:
        """Return a dict ``{axis_name: 1-D coordinate array}``."""

        return {f"x{i + 1}": c for i, c in enumerate(self.coordinates)}

    def mesh(self) -> dict[str, NDArray]:
        """Return a dict ``{axis_name: N-D meshgrid array}``."""
        coords = self.coords()
        coord_mesh = np.meshgrid(*coords.values(), indexing="ij")
        return dict(zip(coords.keys(), coord_mesh, strict=False))


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

    def scalar(self) -> DataArray:
        coords = self.grid.coords()
        scalar = self.amplitude * self.grid.mesh()[self.dimension]
        return DataArray(scalar, dims=list(coords.keys()), coords=coords)

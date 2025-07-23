import numpy as np
import pytest
import xarray as xr

from sgs_tools.geometry.grid import CoordScalar, UniformCartesianGrid
from sgs_tools.geometry.staggered_grid import (
    compose_vector_components_on_grid,
    diff_lin_on_grid,
    grad_on_cart_grid,
    grad_vec_on_grid,
    interpolate_to_grid,
)
from sgs_tools.simple_flows.SimpleShear import ScalarGradient


@pytest.fixture
def u_grid():
    return UniformCartesianGrid([0.5, 0, 0.5], [1, 1, 1])


@pytest.fixture
def v_grid():
    return UniformCartesianGrid([0, 0.5, 0.5], [1, 1, 1])


@pytest.fixture
def w_grid():
    return UniformCartesianGrid([0.5, 0.5, 0], [1, 1, 1])


@pytest.fixture
def vel_shear(u_grid, v_grid, w_grid):
    u = (
        CoordScalar(grid=u_grid, direction=0, amplitude=1)
        .scalar([64, 64, 64])
        .rename({"x1": "x_centre", "x2": "y_face", "x3": "z_centre"})
    )

    v = (
        CoordScalar(grid=v_grid, direction=1, amplitude=2)
        .scalar([64, 64, 64])
        .rename({"x1": "x_face", "x2": "y_centre", "x3": "z_centre"})
    )

    w = (
        CoordScalar(grid=w_grid, direction=2, amplitude=3)
        .scalar([64, 64, 64])
        .rename({"x1": "x_centre", "x2": "y_centre", "x3": "z_face"})
    )
    return xr.Dataset({"u": u, "v": v, "w": w})


@pytest.fixture
def scalar_x_gdt(u_grid):
    return (
        ScalarGradient(u_grid, "x1", 1.0, 0.0)
        .field([64, 64, 64])
        .rename({"x1": "x_centre", "x2": "y_face", "x3": "z_centre"})
    )


@pytest.fixture
def scalar_y_gdt(v_grid):
    return (
        ScalarGradient(v_grid, "x2", 2.0, 0.0)
        .field([64, 64, 64])
        .rename({"x1": "x_face", "x2": "y_centre", "x3": "z_centre"})
    )


@pytest.fixture
def scalar_z_gdt(w_grid):
    return (
        ScalarGradient(w_grid, "x3", 3.0, 0.0)
        .field([64, 64, 64])
        .rename({"x1": "x_centre", "x2": "y_centre", "x3": "z_face"})
    )


# Fixture for common test data
@pytest.fixture
def sample_dataset():
    # Create a sample dataset with staggered grid coordinates
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)
    x_c = np.linspace(0.5, 10.5, 11)

    data = np.random.rand(11, 11)
    ds = xr.Dataset(
        {
            "field": (["x_face", "y_centre"], data),
            "x_face": x,
            "y_centre": y,
            "x_centre": x_c,
        }
    )
    return ds


@pytest.fixture
def vector_components():
    # Create sample vector components
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)

    u = xr.DataArray(
        np.random.rand(11, 11),
        dims=["x_face", "y_centre"],
        coords={"x_face": x, "y_centre": y},
        name="u",
    )
    v = xr.DataArray(
        np.random.rand(11, 11),
        dims=["x_centre", "y_face"],
        coords={"x_centre": x, "y_face": y},
        name="v",
    )
    return [u, v]


def test_diff_lin_on_grid(scalar_x_gdt, scalar_y_gdt, scalar_z_gdt):
    grad = diff_lin_on_grid(scalar_x_gdt, "x_centre")
    const_nan_right = xr.full_like(grad, 1.0)
    const_nan_right[-1, :, :] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_x_gdt, "y_face")
    const_nan_right = xr.full_like(grad, 0.0)
    const_nan_right[:, 0, :] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_x_gdt, "z_centre")
    const_nan_right = xr.full_like(grad, 0.0)
    const_nan_right[:, :, -1] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_y_gdt, "x_face")
    const_nan_right = xr.full_like(grad, 0.0)
    const_nan_right[0, :, :] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_y_gdt, "y_centre")
    const_nan_right = xr.full_like(grad, 2.0)
    const_nan_right[:, -1, :] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_y_gdt, "z_centre")
    const_nan_right = xr.full_like(grad, 0.0)
    const_nan_right[:, :, -1] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_z_gdt, "x_centre")
    const_nan_right = xr.full_like(grad, 0.0)
    const_nan_right[-1, :, :] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_z_gdt, "y_centre")
    const_nan_right = xr.full_like(grad, 0.0)
    const_nan_right[:, -1, :] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_z_gdt, "z_face")
    const_nan_right = xr.full_like(grad, 3.0)
    const_nan_right[:, :, 0] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)


def test_interpolate_to_grid(sample_dataset):
    target_dims = ["x_centre", "y_centre"]
    result = interpolate_to_grid(sample_dataset, target_dims)

    assert isinstance(result, xr.Dataset)
    assert all(dim in result.dims for dim in target_dims)

    # For linear interpolation, center point should be average of adjacent face points
    x_idx = len(sample_dataset.x_face) // 2
    y_idx = len(sample_dataset.y_centre) // 2

    expected_val = 0.5 * (
        sample_dataset.field.isel(x_face=x_idx)
        + sample_dataset.field.isel(x_face=x_idx - 1)
    )

    np.testing.assert_allclose(
        result.field.isel(x_centre=x_idx - 1, y_centre=y_idx),
        expected_val.isel(y_centre=y_idx),
        rtol=1e-10,
    )


def test_compose_vector_components_basic(vector_components):
    result = compose_vector_components_on_grid(
        vector_components,
        target_dims=["x_centre", "y_centre"],
        vector_dim="c1",
        name="velocity",
    )

    assert isinstance(result, xr.DataArray)
    assert "c1" in result.dims
    assert result.name == "velocity"
    assert result.shape[0] == 2  # Two components


def test_grad_on_cart_grid(sample_dataset):
    result = grad_on_cart_grid(
        sample_dataset.field,
        space_dims=["x_face", "y_centre"],
        periodic_field=[False, False],
    )

    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 2  # One derivative per dimension


def test_grad_vec_on_grid(vector_components):
    ds = xr.Dataset({comp.name: comp for comp in vector_components})

    result = grad_vec_on_grid(
        ds,
        target_dims=["x_centre", "y_centre"],
        new_dim_name=["c1", "c2"],
        name="gradient",
    )

    assert isinstance(result, xr.DataArray)
    assert "c1" in result.dims
    assert "c2" in result.dims
    assert result.name == "gradient"


# Error cases
def test_interpolate_to_grid_missing_dims():
    ds = xr.Dataset({"field": (["x"], np.random.rand(5))})
    with pytest.raises(AssertionError):
        interpolate_to_grid(ds, target_dims=["y"])


def test_compose_vector_components_mismatched_dims():
    # Create components with mismatched dimensions
    u = xr.DataArray(np.random.rand(5), dims=["x"], name="u")
    v = xr.DataArray(np.random.rand(6), dims=["y"], name="v")

    with pytest.raises(AssertionError):
        compose_vector_components_on_grid([u, v])

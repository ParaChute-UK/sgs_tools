import numpy as np
import pytest
import xarray as xr
from sgs_tools.geometry.vector_calculus import grad_scalar, grad_vector, grad_vector_lin


@pytest.fixture
def sample_scalar_field():
    """Create a simple 2D scalar field"""
    x = np.linspace(0, 2 * np.pi, 10)
    y = np.linspace(0, 2 * np.pi, 10)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Create scalar field x^2 + y^2
    data = X**2 + Y**2

    return xr.DataArray(data, dims=["x", "y"], coords={"x": x, "y": y})


@pytest.fixture
def sample_vector_field():
    """Create a simple 2D vector field"""
    x = np.linspace(0, 2 * np.pi, 10)
    y = np.linspace(0, 2 * np.pi, 10)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Create vector field [x^2 + y^2, x^2 - y^2]
    data1 = X**2 + Y**2
    data2 = X**2 - Y**2
    vector = np.stack([data1, data2])

    return xr.DataArray(
        vector, dims=["c1", "x", "y"], coords={"c1": [1, 2], "x": x, "y": y}
    )


class TestGradientOperations:
    def test_grad_scalar(self, sample_scalar_field):
        """Test gradient of scalar field"""
        result = grad_scalar(sample_scalar_field, space_dims=["x", "y"])

        assert isinstance(result, xr.DataArray)
        assert set(result.dims) == {"c1", "x", "y"}
        assert result.shape[0] == 2  # 2D gradient

        # Analytical derivatives of x^2 + y^2
        x, y = sample_scalar_field.x, sample_scalar_field.y
        X, Y = np.meshgrid(x, y, indexing="ij")
        expected_dx = 2 * X
        expected_dy = 2 * Y

        # Test at interior point
        idx = 4  # middle of the grid
        np.testing.assert_allclose(
            result.isel(x=idx, y=idx).values,
            [expected_dx[idx, idx], expected_dy[idx, idx]],
            rtol=1e-2,
        )

    def test_grad_scalar_custom_dim(self, sample_scalar_field):
        """Test gradient with custom dimension name"""
        result = grad_scalar(
            sample_scalar_field, space_dims=["x", "y"], new_dim_name="custom_dim"
        )
        assert "custom_dim" in result.dims
        assert result.sizes["custom_dim"] == 2

    def test_grad_vector(self, sample_vector_field):
        """Test gradient of vector field"""
        result = grad_vector(sample_vector_field, space_dims=["x", "y"])

        assert isinstance(result, xr.DataArray)
        assert set(result.dims) == {"c1", "c2", "x", "y"}
        assert result.shape[:2] == (2, 2)  # 2x2 gradient tensor

        # Analytical derivatives
        x, y = sample_vector_field.x, sample_vector_field.y
        X, Y = np.meshgrid(x, y, indexing="ij")

        # d(x^2 + y^2)/dx, d(x^2 + y^2)/dy
        # d(x^2 - y^2)/dx, d(x^2 - y^2)/dy
        expected = np.array(
            [
                [2 * X, 2 * Y],
                [2 * X, -2 * Y],
            ]
        )

        # Test at interior point
        idx = 4
        np.testing.assert_allclose(
            result.isel(x=idx, y=idx).values, expected[:, :, idx, idx], rtol=1e-2
        )

    def test_grad_vector_lin(self, sample_vector_field):
        """Test linear gradient of vector field"""
        # Create simple linear vector field
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # v = [2x + y, x - 3y]
        data1 = 2 * X + Y
        data2 = X - 3 * Y
        vector = np.stack([data1, data2])

        field = xr.DataArray(
            vector, dims=["c1", "x", "y"], coords={"c1": [1, 2], "x": x, "y": y}
        )
        result = grad_vector_lin(field, space_dims=["x", "y"])

        # Expected constant gradient tensor:
        # [dv1/dx  dv1/dy] = [2  1]
        # [dv2/dx  dv2/dy] = [1 -3]
        expected = np.array([[2.0, 1.0], [1.0, -3.0]])
        # Test interior point
        idx = 2
        np.testing.assert_allclose(
            result.isel(x=idx, y=idx).values, expected, rtol=1e-10
        )
        assert isinstance(result, xr.DataArray)
        assert set(result.dims) == {"c1", "c2", "x", "y"}
        assert result.shape[:2] == (2, 2)


class TestEdgeCases:
    def test_grad_scalar_single_dim(self):
        """Test gradient of 1D scalar field"""
        x = np.linspace(0, 2 * np.pi, 10)
        data = np.sin(x)
        field = xr.DataArray(data, dims=["x"], coords={"x": x})

        result = grad_scalar(field, space_dims=["x"])
        assert result.shape[0] == 1  # 1D gradient

    def test_grad_vector_missing_dims(self):
        """Test gradient with missing dimensions"""
        invalid_field = xr.DataArray(np.random.rand(2, 5), dims=["c1", "x"])
        with pytest.raises((ValueError, KeyError)):
            grad_vector(invalid_field, space_dims=["x", "y"])


class TestNameHandling:
    def test_grad_scalar_naming(self, sample_scalar_field):
        """Test name preservation in gradient"""
        sample_scalar_field.name = "temperature"
        result = grad_scalar(
            sample_scalar_field, space_dims=["x", "y"], name="temperature_gradient"
        )
        assert result.name == "temperature_gradient"

    def test_grad_vector_naming(self, sample_vector_field):
        """Test name preservation in vector gradient"""
        sample_vector_field.name = "velocity"
        result = grad_vector(
            sample_vector_field, space_dims=["x", "y"], name="velocity_gradient"
        )
        assert result.name == "velocity_gradient"


if __name__ == "__main__":
    pytest.main([__file__])

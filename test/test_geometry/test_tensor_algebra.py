import numpy as np
import pytest
import xarray as xr
from sgs_tools.geometry.tensor_algebra import (
    Frobenius_norm,
    antisymmetrise,
    matrix_prod,
    symmetrise,
    tensor_self_outer_product,
    trace,
    traceless,
)


@pytest.fixture
def sample_vector():
    # Create a sample 3D vector field
    return xr.DataArray(
        np.array([1.0, 2.0, 3.0]), dims=["c1"], coords={"c1": np.arange(3)}
    )


@pytest.fixture
def sample_tensor():
    # Create a sample 3x3 tensor field
    coords = np.arange(3)
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    return xr.DataArray(data, dims=["c1", "c2"], coords={"c1": coords, "c2": coords})


def test_tensor_self_outer_product(sample_vector):
    result = tensor_self_outer_product(sample_vector)

    assert isinstance(result, xr.DataArray)
    assert set(result.dims) == {"c1", "c2"}
    assert result.shape == (3, 3)

    # Manual verification: a_i * a_j
    expected = sample_vector.values * sample_vector.values[:, None]
    np.testing.assert_array_equal(result.values, expected)


def test_tensor_self_outer_product_custom_dims(sample_vector):
    result = tensor_self_outer_product(
        sample_vector.rename({"c1": "dim1"}), vec_dim="dim1", new_dim="dim2"
    )
    assert set(result.dims) == {"dim1", "dim2"}


def test_trace(sample_tensor):
    result = trace(sample_tensor)

    assert isinstance(result, xr.DataArray)
    assert "c1" not in result.dims
    assert "c2" not in result.dims

    # Manual verification: sum of diagonal elements
    expected = 0
    for i in range(3):
        expected += sample_tensor[i, i]
    np.testing.assert_almost_equal(result.values, expected)


def test_trace_with_name(sample_tensor):
    sample_tensor.name = "test_tensor"
    result = trace(sample_tensor, name="traced_tensor")
    assert result.name == "Tr test_tensor"


def test_traceless(sample_tensor):
    result = traceless(sample_tensor)

    assert isinstance(result, xr.DataArray)
    assert set(result.dims) == {"c1", "c2"}

    # Verify the result is actually traceless
    tr = trace(result)
    np.testing.assert_almost_equal(tr.values, 0.0, decimal=10)


def test_Frobenius_norm(sample_tensor):
    result = Frobenius_norm(sample_tensor)

    assert isinstance(result, xr.DataArray)
    assert "c1" not in result.dims
    assert "c2" not in result.dims

    # Manual verification: sqrt(sum(a_ij * a_ij))
    expected = np.sqrt(np.sum(sample_tensor.values**2))
    np.testing.assert_almost_equal(result.values, expected)


def test_symmetrise(sample_tensor):
    result = symmetrise(sample_tensor)

    assert isinstance(result, xr.DataArray)
    assert set(result.dims) == {"c1", "c2"}

    # Verify symmetry: a_ij = a_ji
    np.testing.assert_array_equal(result.values, result.values.T)

    # Manual verification for one element
    expected_12 = 0.5 * (sample_tensor.values[1, 0] + sample_tensor.values[0, 1])
    np.testing.assert_almost_equal(result.values[0, 1], expected_12)


def test_symmetrise_with_name(sample_tensor):
    result = symmetrise(sample_tensor, name="symmetric_tensor")
    assert result.name == "symmetric_tensor"


def test_antisymmetrise(sample_tensor):
    result = antisymmetrise(sample_tensor)

    assert isinstance(result, xr.DataArray)
    assert set(result.dims) == {"c1", "c2"}

    # Verify symmetry: a_ij = a_ji
    np.testing.assert_array_equal(result.values, -result.values.T)

    # Manual verification for one element
    expected_12 = 0.5 * (sample_tensor.values[0, 1] - sample_tensor.values[1, 0])
    np.testing.assert_almost_equal(result.values[0, 1], expected_12)


def test_antisymmetrise_with_name(sample_tensor):
    result = antisymmetrise(sample_tensor, name="antisymmetric_tensor")
    assert result.name == "antisymmetric_tensor"


def test_matrix_prod():
    # Create sample data
    # First test case: 2D matrix multiplication
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[5, 6], [7, 8]])

    tensor1 = xr.DataArray(
        data1, dims=["row", "col"], coords={"row": [0, 1], "col": [0, 1]}
    )

    tensor2 = xr.DataArray(
        data2, dims=["height", "width"], coords={"height": [0, 1], "width": [0, 1]}
    )

    # Expected result for left matrix multiplication
    expected = np.matmul(data1, data2)
    result = matrix_prod(tensor1, tensor2, "col", "height")
    np.testing.assert_array_equal(result.values, expected)
    assert result.dims == ("row", "width")

    # Expected result for left matrix multiplication
    expected = np.matmul(data2, data1)
    result = matrix_prod(tensor1, tensor2, "row", "width")
    np.testing.assert_array_equal(result.values.T, expected)
    assert result.dims == ("col", "height")


class TestEdgeCases:
    def test_tensor_self_outer_product_invalid_dims(self):
        # Test when new_dim already exists
        vec = xr.DataArray(
            np.array([1.0, 2.0, 3.0]), dims=["c1"], coords={"c1": np.arange(3)}
        )
        with pytest.raises(AssertionError):
            tensor_self_outer_product(vec, new_dim="c1")

    def test_trace_non_matching_dimensions(self):
        # Create tensor with different sizes along dimensions
        tensor = xr.DataArray(
            np.random.rand(3, 4),
            dims=["c1", "c2"],
            coords={"c1": np.arange(3), "c2": np.arange(4)},
        )
        with pytest.raises((ValueError, AssertionError)):
            trace(tensor)

    def test_traceless_with_nan(self):
        # Create tensor with NaN
        tensor = xr.DataArray(
            np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            dims=["c1", "c2"],
            coords={"c1": np.arange(3), "c2": np.arange(3)},
        )
        result = traceless(tensor)
        assert np.any(np.isnan(result))

    def test_symmetrise_different_coords(self):
        # Create tensor with different coordinates
        tensor = xr.DataArray(
            np.random.rand(3, 4),
            dims=["c1", "c2"],
            coords={"c1": [1, 2, 3], "c2": [4, 5, 6, 7]},
        )
        with pytest.raises((ValueError, AssertionError)):
            symmetrise(tensor)

    def test_matrix_prod_errors(self):
        # Test error cases
        data1 = np.array([[1, 2], [3, 4]])
        tensor1 = xr.DataArray(data1, dims=["row", "col"])

        # Test non-existent dimension
        with pytest.raises(AssertionError):
            matrix_prod(tensor1, tensor1, "nonexistent", "col")

        # Test with 'dummy' in dimensions
        tensor_dummy = xr.DataArray(data1, dims=["dummy", "col"])
        with pytest.raises(AssertionError):
            matrix_prod(tensor_dummy, tensor1, "dummy", "col")

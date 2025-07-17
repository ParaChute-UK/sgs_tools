import numpy as np
import pytest
import xarray as xr
from sgs_tools.sgs.coarse_grain import CoarseGrain


def test_coarse_grain_filter():
    # Create a sample 4x4 field
    field_data = np.arange(1.0, 17.0).reshape(4, 4)

    field = xr.DataArray(field_data, dims=["x", "y"])

    # Create a sample  kernels
    cg_2d = CoarseGrain(window={"x": 2, "y": 2})
    cg_1d = CoarseGrain(window={"x": 2})

    expected_2d = np.array([[3.5, 5.5], [11.5, 13.5]])
    expected_1d = np.array([[3, 11], [4, 12], [5, 13], [6, 14]]).T

    for cg, expected_data in zip([cg_2d, cg_1d], [expected_2d, expected_1d]):
        # Apply the filter
        result = cg.filter(field)
        #
        expected = xr.DataArray(
            expected_data,
            dims=["x", "y"],
        )

        # Check if the result matches the expected output
        xr.testing.assert_allclose(result, expected)


def test_coarse_grain_invalid_dims():
    # Test with missing dimensions
    field = xr.DataArray(
        np.random.rand(4),
        dims=["x"],  # Only one dimension
    )

    coarse_grain = CoarseGrain(window={"x": 2, "y": 2})

    # Should raise a Value error trying to locate the missing 'y' dimension
    with pytest.raises(ValueError):
        coarse_grain.filter(field)

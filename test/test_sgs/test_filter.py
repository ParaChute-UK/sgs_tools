import pytest
import numpy as np
import xarray as xr
from sgs_tools.sgs.filter import CoarseGrain

def test_coarse_grain_filter():
    # Create a sample 4x4 field
    field_data = np.arange(1.0,17.0).reshape(4,4)

    field = xr.DataArray(
        field_data,
        dims=['x', 'y'],
        coords={
            'x': np.arange(4),
            'y': np.arange(4)
        }
    )

    # Create a sample  kernels
    kernel_2d = xr.DataArray(
        np.array([[0.25, 0.25], [0.25, 0.25]]),
        dims=['x', 'y']
    )

    kernel_deg = xr.DataArray(
            np.array([[0.5, 0.5]]),
            dims=['x', 'y']
        )

    kernel_1d = xr.DataArray(
            np.array([0.5, 0.5]),
            dims=['x']
        )
    expected_2d= np.array([
            [3.5, 5.5],
            [11.5, 13.5]
        ])
    expected_1d= np.array([[3, 11], [4, 12], [5, 13], [6, 14]])

    for kernel, expected_data in zip([kernel_2d, kernel_deg, kernel_1d],
                                     [expected_2d, expected_1d, expected_1d]):
        # Initialize the CoarseGrain filter
        coarse_grain = CoarseGrain(kernel=kernel, filter_dims=kernel.dims)

        # Apply the filter
        result = coarse_grain.filter(field)

        #
        expected = xr.DataArray(
            expected_data,
            dims=['x', 'y'],
        )

        # Check if the result matches the expected output
        xr.testing.assert_allclose(result, expected)

def test_coarse_grain_invalid_dims():
    # Test with missing dimensions
    kernel = xr.DataArray(
        np.array([[0.25, 0.25], [0.25, 0.25]]),
        dims=['x', 'y']
    )

    field = xr.DataArray(
        np.random.rand(4),
        dims=['x']  # Only one dimension
    )

    coarse_grain = CoarseGrain(kernel=kernel, filter_dims=['x', 'y'])

    # Should raise an assertion error due to missing dimension
    with pytest.raises(AssertionError):
        coarse_grain.filter(field)

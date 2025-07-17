import numpy as np
import pytest
import xarray as xr
import xrft
from sgs_tools.diagnostics.spectra import radial_spectrum


@pytest.fixture
def rand():
    Nx = 40
    Ny = 20
    dx = np.random.rand()
    dy = np.random.rand()
    xcoord = dx * (
        np.arange(-Nx // 2, -Nx // 2 + Nx) + np.random.randint(-Nx // 2, Nx // 2)
    )
    ycoord = dy * (
        np.arange(-Ny // 2, -Ny // 2 + Ny) + np.random.randint(-Ny // 2, Ny // 2)
    )

    da = xr.DataArray(
        np.random.rand(Nx * Ny).reshape([Nx, Ny]),
        dims=["x", "y"],
        coords={"x": xcoord, "y": ycoord},
    )
    return {"val": da, "dx": dx, "dy": dy}


@pytest.fixture
def plane_wave():
    # discretized domain
    P = 2  # domain size (square)
    Nx = 40  # resolution (equal)
    dx = P / Nx  # domain discretization
    dy = dx
    xcoord = np.linspace(0, P, Nx, endpoint=False)
    ycoord = xcoord

    # simple wave
    k_freq = np.array([4, 4])  # num. of periods in domain along x & y (units: 1/P)
    wavelength = P / k_freq  # wavelength in (units: length)
    k = 2 * np.pi / wavelength  # angular wave number (units: rad/length)

    da = xr.DataArray(
        np.sin(k[0] * xcoord)[:, None] * np.sin(k[1] * ycoord)[None, :],
        dims=["x", "y"],
        coords={"x": xcoord, "y": ycoord},
    )
    return {"val": da, "dx": dx, "dy": dy, "lambda": wavelength}


# Assert Parseval's using xrft.dft
# this is for illustration of xrft normalisations
def test_parseval_1d_dft(rand, plane_wave):
    for data in [rand, plane_wave]:
        da = data["val"].isel(y=0)
        dx = data["dx"]
        FT = xrft.fft(da, dim="x", true_phase=True, detrend=None, true_amplitude=True)

        np.testing.assert_allclose(
            (np.abs(da) ** 2).sum().load() * dx,
            (np.abs(FT) ** 2).sum().load() * FT["freq_x"].spacing,
        )


# Assert Parseval's using xrft.power_spectrum with scaling='density'
# this is for illustration of xrft normalisations
def test_parseval_1d_powerspec(rand, plane_wave):
    for data in [rand, plane_wave]:
        da = data["val"].isel(y=0)
        dx = data["dx"]

        ps = xrft.power_spectrum(
            da,
            dim=("x"),
            scaling="density",
            detrend=None,
        )
        np.testing.assert_allclose(
            (np.abs(da) ** 2).sum().load() * dx,
            ps.sum().load(),
        )


# Assert 2dim Parseval's using xrft.power_spectrum with scaling='density'
# this is for illustration of xrft normalisations
def test_parseval_2d_powerspec(rand, plane_wave):
    for data in [rand, plane_wave]:
        da = data["val"]
        dx = data["dx"]
        dy = data["dy"]

        ps = xrft.power_spectrum(
            da,
            dim=("x", "y"),
            scaling="density",
            detrend=None,
        )
        np.testing.assert_allclose(
            (np.abs(da) ** 2).sum().load() * dx * dy,
            ps.sum(["freq_x", "freq_y"]).load(),
        )


# radial spectrum Parseval test
def test_parseval_2d_radial_powerspec(rand, plane_wave):
    for data in [rand, plane_wave]:
        da = data["val"]

        ps = xrft.power_spectrum(
            da,
            dim=("x", "y"),
            scaling="density",
            detrend=None,
            true_phase=False,
        )
        for scaling in ("spectrum", "density"):
            psr = radial_spectrum(
                ps,
                ("freq_x", "freq_y"),
                radial_bin_width=(ps["freq_x"][1] - ps["freq_x"][0]).item(),
                truncate=False,
                scaling=scaling,
                prefix="freq_",
            )
            np.testing.assert_allclose(
                (psr * psr["freq_dA"]).sum().load(),
                ps.sum(["freq_x", "freq_y"]).load(),
            )


# radial spectrum wavenumber test
def test_wavelength_2d_radial_powerspec(plane_wave):
    da = plane_wave["val"]
    wavelengths = plane_wave["lambda"]
    ps = xrft.power_spectrum(
        da,
        dim=("x", "y"),
        scaling="density",
        detrend=None,
        true_phase=True,
    )
    psr = radial_spectrum(
        ps,
        ("freq_x", "freq_y"),
        radial_bin_width=(ps["freq_x"][1] - ps["freq_x"][0]).item(),
        truncate=False,
        scaling="spectrum",
        bin_anchor="left",
        prefix="freq_",
    )

    peak_freq_expected = np.sqrt((1 / wavelengths**2).sum())
    max_freq = psr.argmax(...)
    assert (
        psr["freq_r"].isel(max_freq) <= peak_freq_expected
        and psr["freq_r"].isel(max_freq) + psr["freq_dr"].isel(max_freq)
        >= peak_freq_expected
    )

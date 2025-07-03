import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr
import xrft  # type: ignore
from sgs_tools.io.netcdf_writer import NetCDFWriter


def radial_spectrum(
    ps: xr.DataArray,
    fftdim: Sequence[str],
    nbins: int,
    bin_anchor: str = "center",
    truncate: bool = True,
    scaling: str = "spectrum",
    prefix="freq_",
) -> xr.DataArray:
    r"""
    Isotropize a 2D power spectrum or cross spectrum
    by taking an "azimuthal" average over the specified dimensions.

    .. math::
        \text{iso}_{ps}[kr^*] = \sum_{k: |k| = kr} |\mathbb{F}(da')|^2 * w
        kr^* = \langle k \rangle_{k \in [kr_0, kr_1]}

    where :math:`kr` is the radial wavenumber. and :math:`w` are defined implicitly through the scaling. Always
        :math:`\sum ps = \sum iso_ps * iso_ps[prefix+'dA']`
        This satisfies Parseval if that :math:`\sum ps = \sum real\_data**2 * dx * dy`.

    Parameters
    ----------
    ps : `xarray.DataArray`
        The power spectrum or cross spectrum to be isotropized.
    fftdim : list
        The fft dimensions overwhich the isotropization must be performed.
    nbins : int
        Number of linearly spaced bins in which to distribute the power spectrum
    truncate : bool, optional
        If True, the spectrum will be truncated for wavenumbers larger than min(max(ps[fftdim])).
    bin_anchor: str, optional
        Where to place the radial wavenumber within the bin. Choices {'left', 'right', 'centre', 'com'}. Default: 'mean'
        if 'com' : compute as the centre-of-mass radius: :math: `\sum (ps * k_r) / \sum (ps)` before rescaling.
    scaling: str, optional, default: density
        Rescale the power spectrum to satisfy :math:`\sum ps = \sum iso_ps * iso_ps[prefix+'dA']`
        * `density`: set :math: `iso_ps[prefix+'dA'](k_r) = \pi * (k_r^{top}^2 - k_r^{bottom}^2)`, where :math:`k_r^{top}` and :math:`k_r^{bottom}` are the bin edges.
        * `spectrum`: set :math: `iso_ps[prefix+'dA'](k_r) = 1`
    """
    # name of new spectral dimension
    dim_name = prefix + "r"

    # compute radial wavenumber bins
    fftcoords = xr.Dataset(
        {f"d{i}": (d, ps.coords[d].values) for i, d in enumerate(fftdim)}
    )
    freq_r = ((fftcoords**2).to_dataarray().sum("variable") ** 0.5).rename(dim_name)

    if nbins > max([fftcoords[d].shape[0] for d in fftcoords]):
        warnings.warn(
            f"nbins {nbins} > max number of linear frequencies"
            ", likely to have empty bins with nan values"
        )

    # select radial bins
    if truncate:
        last_bin_edge = min([abs(fftcoords[x]).max().item() for x in fftcoords])
    else:
        last_bin_edge = freq_r.max().item()

    # last_bin_edge *= 1.001 # add tolerance for floating point comparison
    kr_bins = np.linspace(0, last_bin_edge, nbins + 1, endpoint=True)
    kr_delta = kr_bins[1:] - kr_bins[:-1]

    # total spectral power in annulus
    iso_ps = (
        ps.groupby_bins(freq_r, bins=kr_bins, right=True, include_lowest=True)
        .sum()
        .rename({f"{dim_name}_bins": dim_name})
        .drop_vars(dim_name)
    )

    # select reference wave number
    if bin_anchor == "center":
        kr_ref = (kr_bins[1:] + kr_bins[:-1]) / 2
    elif bin_anchor == "left":
        kr_ref = kr_bins[:-1]
    elif bin_anchor == "right":
        kr_ref = kr_bins[1:]
    elif bin_anchor == "com":
        kr_ref = (
            (freq_r * ps)
            .groupby_bins(freq_r, bins=kr_bins, right=True, include_lowest=True)
            .sum()
            / iso_ps
        ).data
    else:
        raise ValueError(
            f"Unrecognised bin_anchor {bin_anchor}. "
            "Choose from 'center', 'left', 'right', 'com'."
        )
    # add a bin coordinates
    iso_ps.coords[dim_name] = kr_ref
    iso_ps[dim_name].attrs["anchor"] = bin_anchor
    iso_ps = iso_ps.assign_coords({prefix + "dr": (dim_name, kr_delta)})

    # rescale amplitude
    if scaling == "density":
        annulus_area = np.pi * (kr_bins[1:] ** 2 - kr_bins[:-1] ** 2)
        iso_ps = iso_ps / annulus_area
        iso_ps = iso_ps.assign_coords({prefix + "dA": (dim_name, annulus_area)})
        iso_ps[prefix + "dA"].attrs["description"] = "pi * (rmax^2 - rmin^2)"
        if not truncate:
            msg = "Energy density scaling is inconsistent beyond the min(max(linear frequency)). Interpete with caution!"
            warnings.warn(msg, FutureWarning)
    elif scaling == "spectrum":
        iso_ps = iso_ps.assign_coords({prefix + "dA": (dim_name, np.ones_like(iso_ps))})
    else:
        raise ValueError(
            f"Unrecognised scaling {scaling}. " "Choose from 'spectrum', 'density'."
        )
    return iso_ps


def spectra_1d_nd_radial(
    simulation: xr.Dataset,
    hdims: Sequence[str],
    power_spectra_fields: Sequence[str],
    cross_spectra_fields: Sequence[tuple[str, str]],
    writer: NetCDFWriter,
    output_path: Path,
) -> None:
    """Note: resulting cooordinates are in units of"""
    spec = {}
    extra_coords = []
    for x in hdims:
        extra_coords += [
            d for d in simulation.coords if d != x and x in simulation[d].dims
        ]
    sim = simulation.drop_vars(extra_coords, errors="ignore")

    smooth = 2  # smoothing factor for radial spectral bins
    # mypy magic
    fftdim = [f"k_{x}" for x in hdims]

    for field in power_spectra_fields:
        data = sim[field]
        for x in hdims:
            spec[f"{field}_F{x}"] = xrft.power_spectrum(
                data,
                dim=x,
                real_dim=x,
                scaling="density",
                detrend=None,
                prefix="k_",
                true_phase=True,
            )
        spec[f"{field}_F{''.join(hdims)}"] = xrft.power_spectrum(
            data,
            dim=hdims,
            scaling="density",
            prefix="k_",
            detrend=None,
        )
        spec[f"{field}_Fr"] = radial_spectrum(
            spec[f"{field}_F{''.join(hdims)}"],
            fftdim=fftdim,
            nbins=max([data[d].size for d in hdims]) // smooth,
            bin_anchor="left",
            truncate=False,
            scaling="density",
        )

    for field1, field2 in cross_spectra_fields:
        data1 = sim[field1]
        data2 = sim[field2]
        for x in hdims:
            spec[f"{field1}_{field2}_F{x}"] = xrft.cross_spectrum(
                data1,
                data2,
                dim=x,
                real_dim=x,
                scaling="density",
                prefix="k_",
                true_phase=True,
                detrend=None,
            )
        spec[f"{field1}_{field2}_F{''.join(hdims)}"] = xrft.cross_spectrum(
            data1, data2, dim=hdims, scaling="density", prefix="k_", true_phase=True
        )
        spec[f"{field1}_{field2}_Fr"] = radial_spectrum(
            spec[f"{field1}_{field2}_F{''.join(hdims)}"],
            fftdim=fftdim,
            nbins=max([data[d].size for d in hdims]) // smooth,
            bin_anchor="left",
            truncate=False,
            scaling="density",
        )

    spec_ds = xr.Dataset(spec).compute()

    # rechunk for IO optimisation ??
    # have to do explicit rechunking because UM date-time coordinate is an object
    # spec_ds = spec_ds.chunk({dim: "auto" for dim in ["x", "y", "z"] if dim in spec_ds.dims})
    writer.write(spec_ds, output_path)

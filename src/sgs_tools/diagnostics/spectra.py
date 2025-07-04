import warnings
from typing import Sequence

import numpy as np
import xarray as xr
import xrft  # type: ignore


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
        msg = (
            f"nbins {nbins} > max number of linear frequencies"
            ", likely to have empty bins with nan values"
        )
        warnings.warn(UserWarning(msg))

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
            warnings.warn(msg)
    elif scaling == "spectrum":
        iso_ps = iso_ps.assign_coords({prefix + "dA": (dim_name, np.ones_like(iso_ps))})
    else:
        raise ValueError(
            f"Unrecognised scaling {scaling}. " "Choose from 'spectrum', 'density'."
        )
    return iso_ps


def spectra_1d_radial(
    simulation: xr.Dataset,
    hdims: Sequence[str],
    power_spectra_fields: Sequence[str],
    cross_spectra_fields: Sequence[tuple[str, str]],
    radial_smooth_factor: int = 2,
    fillnan: float = 0.0,
) -> xr.Dataset:
    """
    :param: radial_smooth_factor: smoothing factor for radial spectral bins. If 2 will have radial bin widht is 2*linear wavenumber.
    Note: resulting spectral cooordinates are in units of inverse length, not radians/length.
    """
    spec = {}
    extra_coords = []
    for x in hdims:
        extra_coords += [
            d for d in simulation.coords if d != x and x in simulation[d].dims
        ]
    sim = simulation.drop_vars(extra_coords, errors="ignore")
    assert (
        "r" not in hdims
    ), "'r' dimension found in hdims, but it is reserved for radial spectra"

    # power spectra
    for field in power_spectra_fields:
        data = sim[field].fillna(fillnan)  # xrft doesn't play well with nan
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
        nd_spectrum = xrft.power_spectrum(
            data,
            dim=hdims,
            scaling="density",
            prefix="k_",
            detrend=None,
        )
        spec[f"{field}_Fr"] = radial_spectrum(
            nd_spectrum,
            fftdim=[f"k_{x}" for x in hdims],
            nbins=max([data[d].size for d in hdims]) // radial_smooth_factor,
            bin_anchor="left",
            truncate=False,
            scaling="density",
            prefix="k_",
        )

    # cross spectra
    for field1, field2 in cross_spectra_fields:
        data1 = sim[field1].fillna(fillnan)  # xrft doesn't play well with nan
        data2 = sim[field2].fillna(fillnan)  # xrft doesn't play well with nan
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
        nd_spectrum = xrft.cross_spectrum(
            data1, data2, dim=hdims, scaling="density", prefix="k_", true_phase=True
        )
        spec[f"{field1}_{field2}_Fr"] = radial_spectrum(
            nd_spectrum,
            fftdim=[f"k_{x}" for x in hdims],
            nbins=max([data[d].size for d in hdims]) // radial_smooth_factor,
            bin_anchor="left",
            truncate=False,
            scaling="density",
            prefix="k_",
        )
    # reduce along non-spectral hdims
    for f in spec:
        if not f.endswith("_Fr"):
            non_spec_hdims = [x for x in hdims if x in spec[f].dims]
            spec[f] = spec[f].sum(non_spec_hdims)
            spec[f].attrs["statistic"] = f"sum_along_{non_spec_hdims}"
    spec_ds = xr.Dataset(spec)

    # add clarifying attributes for linear spectra
    for x in hdims:
        spec_ds.attrs[f"N{x}"] = sim[x].size
        # the fourier spectrum call ensures that the hdims coordinates
        # are regularly spaced
        spec_ds.attrs[f"d{x}"] = (sim[x][1] - sim[x][0]).item()

    # add header clarifying spectral conventions
    spec_ds.attrs["convention"] = """Total signal power =
        = (g**2).sum([<x>,...]) * d<x> * ...
        = g_F<x>.sum(k_<x>) * d<x>
        = (g_Fr*F<k_dA>).sum(k_dr)* """
    return spec_ds

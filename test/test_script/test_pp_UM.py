import shutil
from pathlib import Path

import pytest

import sgs_tools.scripts.post_process as pp_um


@pytest.fixture
def test_args():
    return [
        "test/test_script/df667_800m_L63_Slicea_pr.nc",
        "um",
        "__test_pp_UM",
        "--h_resolution",
        "800",
        "--overwrite_existing",
        "--z_chunk_size",
        "10",
        "--t_chunk_size",
        "1",
        "--z_range",
        "0",
        "1000",
        "--cross_spectra_fields",
        "u,v",
        "v,w",
        "--power_spectra_fields",
        "u",
        "v",
        "w",
        "theta",
        "--hdims",
        "x",
        "y",
        "--vertical_profiles",
        "--vprofile_fname_out",
        "profiles.nc",
        "--horizontal_spectra",
        "--hspectra_fname_out",
        "spectra.nc",
        "--radial_smooth_factor",
        "1",
        "--radial_truncation",
        "--anisotropy",
        "--aniso_fname_out",
        "aniso.nc",
        "--box_delta_scales",
        "2",
        "4",
        "--box_meter_scales",
        "100",
        "--box_domain_scales",
        "0.5",
        "--gauss_scales",
        "2",
    ]


def test_main_full_pipeline(test_args):
    tmp_path = Path(test_args[2])
    tmp_path.mkdir(exist_ok=False, parents=False)
    try:
        # Execute
        args = pp_um.parse_args(test_args)
        pp_um.run(args)

        # Assert outputs exist
        assert (tmp_path / args["vprofile_fname_out"]).exists()
        assert (tmp_path / args["hspectra_fname_out"]).exists()
        assert len(list(tmp_path.glob(f"{args['aniso_fname_out'].stem}*.nc"))) > 0

    finally:
        # Cleanup
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

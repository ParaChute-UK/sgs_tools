from pathlib import Path

import pytest

import sgs_tools.scripts.post_process as pp_um
from sgs_tools.scripts.fname_out import build_output_fname


@pytest.fixture
def test_args():
    return [
        "test/test_script/df667_800m_L63_Slicea_pr.nc",
        "um",
        "pp",
        "--fname_suffix",
        "test_me",
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
        "profiles",
        "--horizontal_spectra",
        "--hspectra_fname_out",
        "spectra",
        "--radial_smooth_factor",
        "1",
        "--radial_truncation",
        "--anisotropy",
        "--aniso_fname_out",
        "aniso",
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


def test_main_full_pipeline(test_args, master_output_dir):
    tmp_path = master_output_dir / Path(test_args[2])
    tmp_path.mkdir(exist_ok=False, parents=False)
    test_args[2] = str(tmp_path)
    # Execute
    args = pp_um.parse_args(test_args)
    pp_um.run(args)

    # Assert outputs exist

    assert build_output_fname(
        tmp_path / args["vprofile_fname_out"], args["fname_suffix"], pp_um.VPROF_TAG
    ).exists()
    assert build_output_fname(
        tmp_path / args["hspectra_fname_out"], args["fname_suffix"], pp_um.SPECTRA_TAG
    ).exists()
    aniso_glob = build_output_fname(
        args["aniso_fname_out"], args["fname_suffix"], "*", pp_um.ANISOTROPY_TAG
    )
    assert len(list(tmp_path.glob(str(aniso_glob)))) > 0

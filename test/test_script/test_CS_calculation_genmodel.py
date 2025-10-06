from pathlib import Path

import pytest

import sgs_tools.scripts.CS_calculation_genmodel as cs_gen


@pytest.fixture
def test_args(master_output_dir):
    return [
        "test/test_script/df667_800m_L63_Slicea_p*.nc",
        "um",
        "cs_gen",
        "--h_resolution",
        "800",
        "--z_chunk_size",
        "10",
        "--t_chunk_size",
        "1",
        "--z_range",
        "0",
        "1000",
        "--plot_path",
        str(master_output_dir / "cs_gen/plots"),
        "--filter_type",
        "box",
        "--filter_scale",
        "2",
        "4",
        "--regularize_filter_type",
        "box",
        "--regularize_filter_scale",
        "2",
        "4",
    ]


def test_main_full_pipeline(master_output_dir, test_args):
    # check test output directory is clean, so we can safely wipe it on exit
    tmp_path = master_output_dir / Path(test_args[2])
    tmp_path.mkdir(exist_ok=False, parents=False)
    test_args[2] = str(tmp_path)

    # parse clargs
    args = cs_gen.parse_args(test_args)
    # execute main
    cs_gen.run(args)
    # Assert outputs exists
    assert len(list(args["output_path"].glob("*.nc"))) > 0
    # execute plotting
    cs_gen.plot(args)
    assert len(list(args["plot_path"].glob("*.pdf"))) > 0

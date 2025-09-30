from pathlib import Path

import pytest

import sgs_tools.scripts.BasicComparisonSimAnalysis as comp


@pytest.fixture
def test_args():
    return [
        "test/test_script/df667_800m_L63_Slicea_p*.nc",
        "test/test_script/df667_800m_L63_Slicea_p*.nc",
        "800",
        "--plot_path",
        "basic_comp_sim",
        "--z_chunk_size",
        "10",
        "--t_chunk_size",
        "1",
    ]


def test_main_full_pipeline(test_args, master_output_dir):
    # check test output directory is clean, so we can safely wipe it on exit
    tmp_path = master_output_dir / Path(test_args[4])
    tmp_path.mkdir(exist_ok=False, parents=False)
    test_args[4] = str(tmp_path)

    # parse clargs
    args = comp.parse_args(test_args)
    # execute main
    comp.run(args)
    # Assert outputs exists
    assert len(list(args["plot_path"].glob("*.png"))) > 0

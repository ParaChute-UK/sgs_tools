import pytest

import sgs_tools.scripts.BasicComparisonSimAnalysis as comp


@pytest.fixture
def test_args(output_dir):
    return [
        "test/test_script/df667_800m_L63_Slicea_p*.nc",
        "test/test_script/df667_800m_L63_Slicea_p*.nc",
        "um",
        "--h_resolution",
        "800",
        "--plot_path",
        str(output_dir),
        "--z_chunk_size",
        "10",
        "--t_chunk_size",
        "1",
    ]


def test_main_full_pipeline(test_args, output_dir):
    # parse clargs
    args = comp.parse_args(test_args)
    # execute main
    comp.run(args)
    # Assert outputs exists
    assert len(list(args["plot_path"].glob("*.png"))) > 0

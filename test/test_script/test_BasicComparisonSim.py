import pytest

import sgs_tools.scripts.BasicComparisonSimAnalysis as comp


@pytest.fixture
def test_args(output_dir, testing_rootdir):
    return [
        str(testing_rootdir / "test_script/df667_800m_L63_Slicea_p*.nc"),
        str(testing_rootdir / "test_script/df667_800m_L63_Slicea_p*.nc"),
        "um_ideal",
        "--h_resolution",
        "800",
        "--plot_path",
        str(output_dir),
        "--z_chunk_size",
        "10",
        "--t_chunk_size",
        "1",
    ]


@pytest.mark.slow
@pytest.mark.integration
def test_main_full_pipeline(test_args, output_dir):
    # parse clargs
    args = comp.parse_args(test_args)
    # execute main
    comp.run(args)
    # Assert outputs exists
    assert len(list(args["plot_path"].glob("*.png"))) > 0

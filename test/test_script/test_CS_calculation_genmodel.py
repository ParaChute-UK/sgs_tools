import pytest

import sgs_tools.scripts.CS_calculation_genmodel as cs_gen


@pytest.fixture
def test_args(output_dir):
    output = output_dir
    return [
        "test/test_script/df667_800m_L63_Slicea_p*.nc",
        "um",
        str(output),
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
        str(output / "plots"),
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


def test_main_full_pipeline(test_args):
    # parse clargss
    args = cs_gen.parse_args(test_args)
    # execute main
    cs_gen.compute(args)
    # Assert outputs exists
    assert len(list(args["output_path"].glob("*.nc"))) > 0
    # execute plotting
    cs_gen.plot(args)
    assert len(list(args["plot_path"].glob("*.pdf"))) > 0

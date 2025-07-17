import shutil
from pathlib import Path

import pytest
import sgs_tools.scripts.CS_calculation_genmodel as cs_gen


@pytest.fixture
def test_args():
    return [
        "test/test_script/df667_800m_L63_Slicea_p*.nc",
        "800",
        "__test_cs_gen",
        "--z_chunk_size",
        "10",
        "--t_chunk_size",
        "1",
        "--z_range",
        "0",
        "1000",
        "--input_format",
        "um",
        "--plot_path",
        "__test_cs_gen/plots",
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
    # check test output directory is clean, so we can safely wipe it on exit
    tmp_path = Path(test_args[2])
    tmp_path.mkdir(exist_ok=False, parents=False)
    try:
        # parse clargs
        args = cs_gen.parse_args(test_args)
        # execute main
        cs_gen.main(args)
        # Assert outputs exists
        assert len(list(args["output_path"].glob("*.nc"))) > 0
        # execute plotting
        cs_gen.plot(args)
        assert len(list(args["plot_path"].glob("*.pdf"))) > 0
    finally:
        # Cleanup
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

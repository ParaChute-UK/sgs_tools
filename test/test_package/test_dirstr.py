import os

import pytest


@pytest.mark.package
@pytest.mark.fast
def test_output_dir(output_dir, testing_rootdir):
    print("CWD", os.getcwd())
    print("Output dir", output_dir.resolve())
    print("Input dir", testing_rootdir.resolve())
    assert output_dir.exists()
    assert testing_rootdir.exists()

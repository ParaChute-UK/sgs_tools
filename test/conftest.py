import shutil
from pathlib import Path

import pytest

OUTPUT_DIR = Path("__test_out")


def pytest_addoption(parser):
    parser.addoption(
        "--keep-output", action="store_true", help="Keep output directory after tests"
    )


@pytest.fixture(scope="session")
def master_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def pytest_sessionfinish(session):
    keep = session.config.getoption("--keep-output")
    if not keep and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f"Deleted output directory: {OUTPUT_DIR}")

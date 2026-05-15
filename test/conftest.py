import re
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
    print(f"==== [pytest] output dir: {OUTPUT_DIR} ====")
    return OUTPUT_DIR


def _safe_name(nodeid):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", nodeid)


@pytest.fixture
def output_dir(master_output_dir, request):
    name = _safe_name(request.node.nodeid)  # unique per test
    path = master_output_dir / name
    path.mkdir(exist_ok=False)
    return path


def pytest_sessionfinish(session):
    keep = session.config.getoption("--keep-output")
    tests_failed = session.testsfailed > 0

    if not keep and not tests_failed and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f"Deleted output directory: {OUTPUT_DIR}")
    else:
        print(f"Kept output directory: {OUTPUT_DIR} for manual clean-up.")

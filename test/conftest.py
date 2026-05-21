import re
import shutil
from pathlib import Path

import pytest


# clarg config
def pytest_addoption(parser):
    parser.addoption(
        "--keep-output", action="store_true", help="Keep output directory after tests."
    )
    parser.addoption(
        "--all", action="store_true", help="Run all tests including slow ones."
    )


# ---- output directory
OUTPUT_DIR = Path("__test_out")


@pytest.fixture(scope="session")
def testing_rootdir() -> Path:
    return Path(__file__).parent


@pytest.fixture(scope="session")
def master_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def _safe_name(nodeid):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", nodeid)


@pytest.fixture
def output_dir(master_output_dir, request):
    name = _safe_name(request.node.nodeid)  # unique per test
    path = master_output_dir / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=False)
    return path


def pytest_sessionfinish(session):
    keep = session.config.getoption("--keep-output")
    tests_failed = session.testsfailed > 0

    if not keep and not tests_failed and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR.resolve())
        print(f"Deleted output directory: {OUTPUT_DIR.resolve()}")
    else:
        print(f"Kept output directory: {OUTPUT_DIR.resolve()} for manual clean-up.")


# --- skip slow tests by default
def pytest_collection_modifyitems(config, items):
    if config.getoption("--all"):
        return

    selected = config.option.keyword  # from -k test_name
    markexpr = config.option.markexpr  # for -m slow

    skip_slow = pytest.mark.skip(reason="use --all to run")

    for item in items:
        is_slow = "slow" in item.keywords

        explicitly_selected = (selected and selected in item.nodeid) or (
            markexpr and "slow" in markexpr
        )

        if is_slow and not explicitly_selected:
            item.add_marker(skip_slow)

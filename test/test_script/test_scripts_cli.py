import subprocess
from importlib.metadata import entry_points

import pytest


@pytest.mark.parametrize(
    "script_name",
    [
        ep.name
        for ep in entry_points(group="console_scripts")
        if ep.value.startswith("sgs_tools.scripts.")
    ],
)
def test_cli_script_help(script_name):
    result = subprocess.run(
        [script_name, "--help"], capture_output=True, text=True, shell=True
    )
    assert result.returncode == 0, f"{script_name} failed with code {result.returncode}"
    assert "usage" in result.stdout.lower(), f"{script_name} missing help output"

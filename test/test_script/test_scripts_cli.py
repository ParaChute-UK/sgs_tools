import subprocess
from importlib import import_module
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
    result = subprocess.run([script_name, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, f"{script_name} failed with code {result.returncode}"
    assert "usage" in result.stdout.lower(), f"{script_name} missing help output"


MINIMAL_ARGS = {
    "CS_calculation_genmodel": ["-V", "1"],
    "post_process": ["-V", "1"],
    "version": ["-v"],
    "BasicComparisonSimAnalysis": ["-V", "1"],
    "ReferenceComparisonSimAnalysis": ["-V", "1"],
}


@pytest.mark.parametrize("script_name", MINIMAL_ARGS.keys())
def test_cli_main_minimal(script_name, capsys):
    """Import each CLI's main() and verify it runs with minimal valid arguments."""
    # Import dynamically from the script name convention, e.g. "sgs_dynamic" -> "sgs_tools.scripts.dynamic"
    module_name = f"sgs_tools.scripts.{script_name}"
    mod = import_module(module_name)
    main = getattr(mod, "main")
    assert main is not None, f"{script_name} has no main()"
    # Run with the minimal valid arguments
    try:
        main(MINIMAL_ARGS[script_name])
    except SystemExit as e:
        assert e.code == 0, f"{script_name} exited abnormally with {e.code}"

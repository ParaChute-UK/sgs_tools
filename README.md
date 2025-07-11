# Python tools for SGS analysis

## Install

  Use typical `pip` installation (preferably within a virtual environment to keep dependencies clean) e.g.

  ``` pip install git+https://github.com/dvlaykov/sgs_tools.git```

**NB** The package is in active development. No backwards compatibility is guarranteed at this time.

## Documentation
The docs are generated via [sphinx](https://www.sphinx-doc.org/en/master/) and several sphinx addons.
1. To automatically get all the dependencies, reinstall the package adding `  '[doc]' ` to the end of the install command.
1. To generate the docs run `make doc` from the top level in the repository. This will generate an html version of the documentation.
3. The resulting docs can be accessed from `<repo_directory>/doc/_build/html/index.html`

## For developers:

### Install
  1. clone the [repository](https://github.com/dvlaykov/sgs_tools)
  2. make an editable install via
    ```pip install --editable <location-of-repository>[dev]```

### Contribute
  * Please add any issues, feature requests and pull requests directly to the
    [GitHub repository](https://github.com/dvlaykov/sgs_tools) -- all are very welcome.
  * Testing, formatting and static types are handled through `tox` and for the whole repo you can run `make test` to check formatting type hints and run the tests and their coverage.

    * Tests rely on the `pytest` package and should go to `test/test_*.py`
    * Formatting is taken care of by `ruff`. Run `make format` to auto-format.
    * Type hints are handled by `mypy`. Run `make mypy` to check for any issues


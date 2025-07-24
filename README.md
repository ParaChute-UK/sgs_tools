# Python tools for SGS analysis

## Install

  Use typical `pip` installation (preferably within a virtual environment to keep dependencies clean) e.g.

  ``` pip install git+https://github.com/dvlaykov/sgs_tools.git```

**NB** The package is in active development. No backwards compatibility is guarranteed at this time.

## Documentation
The docs are generated via [sphinx](https://www.sphinx-doc.org/en/master/) and several sphinx addons.
1. To automatically get all the dependencies, reinstall the package adding `  'sgs_tools[doc]' ` to the end of the install command.
1. To generate the docs run `make doc` from the top level in the repository. This will generate an html version of the documentation.
3. The resulting docs can be accessed from `<repo_directory>/doc/_build/html/index.html`

## For developers:

### Install
  1. clone the [repository](https://github.com/dvlaykov/sgs_tools)
  2. make an editable install via
    ```pip install --editable <location-of-repository>[dev]```

### Contribute

  * We welcome, issues, feature requests, PRs, etc. directly on
    [GitHub](https://github.com/dvlaykov/sgs_tools).

#### 🧪 Testing & Formatting

  Testing is managed via [tox](https://tox.wiki/en/4.28.1/) and a set of convenience `make` targets defined in the `Makefile`.

  *	To run all checks (formatting, linting, type checks, tests, and coverage) across the repo:

  ``` console
  make test
  ```
  or

  ``` console
  tox
  ```

See `Makefile` or `tox.ini` for more grannular options.


#### 🔧 Tooling Overview

- **Unit/Integration Tests**: [`pytest`](https://docs.pytest.org/)
  - Will look for tests as `tests/test_*.py`
- **Code Style**:
  - [`ruff`](https://github.com/charliermarsh/ruff): formatting and linting
  - [`mypy`](http://mypy-lang.org/): static type checking
  - [`pre-commit`](https://pre-commit.com/): wraps up the other two and cleans-up staged files before commit and for PRs to `devel`.

> 💡 Run `pre-commit install` once to activate Git hooks in your local repo.

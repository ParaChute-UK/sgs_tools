# sgs_tools

Python tools for sub-grid scale (SGS) fluid dynamics.

🛠️ **NB** The package is in active development. Interfaces and features may change.


---
## 🚀 Install

### 🧪 Development Version (from GitHub `devel` branch)

  To install the latest development version:

  ```console
  pip install git+https://github.com/dvlaykov/sgs_tools.git@devel
  ```

⚠️ > **NB** Python requirement: requres Python **>=3.11**. Installation on older versions will fail.

## 🤝 Contributing

We welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and pull requests.

To get started follow the Developer Setup instructions below.

- Clone the repository and create a feature branch from `devel`
- Use `make pre_commit` or `tox -e pre_commit` to fix common formatting/style issues
- Use `make test` or `tox` to run checks before submitting a PR
- Submit your PR to the `devel` branch

> 💬 Feel free to open [issues](https://github.com/dvlaykov/sgs_tools) for questions, ideas, or feedback. We appreciate your input!

## 🧪 Development Setup & Tooling

The dev tools are managed using [Poetry](https://python-poetry.org/docs/).

> 🧰 If you're unfamiliar with [Poetry](https://python-poetry.org/docs/), it's a Python packaging and publishing tool for dependency management and development workflows.
You can still use `pip` for basic installs, but we recommend Poetry for contributing to this project.


### Setup
  1. clone the [repository](https://github.com/dvlaykov/sgs_tools)
  ```console
  git clone https://github.com/dvlaykov/sgs_tools.git
  cd sgs_tools
  ```

  2. [Optional] Create and activate a virtual environment in the preferred way (venv, conda, ...)

  3. [Install Poetry](https://python-poetry.org/docs/#installation) (if not already installed), e.g.
  ```console
  curl -sSL https://install.python-poetry.org | python3 -
  ```

  4. Install all dependencies including dev tools:
  ```console
  poetry install --with dev
  ```

  > 💡 This will install dev tools: `tox`, `pytest`, `ruff`, `mypy`, and `pre-commit`.

  5. Activate Git hooks to help clean up formatting etc. on commit (this may slow down the commit somewhat).
  ```console
  pre-commit install
  ```

### 🧪 Testing & Style

  Testing and code style is managed via [tox](https://tox.wiki/en/4.28.1/) or a convenience `make` targets defined in the `Makefile`.

  *	Run all checks (formatting, linting, type checks, tests, and coverage) across the repo with

  ``` console
  make test
  ```
  or

  ``` console
  tox
  ```

  * Apply standard formatting fixes and checks (that would be applied anyway for PRs) with
  ``` console
  make pre_commit
  ```
  or
  ``` console
  tox pre_commit
  ```

  See the `Makefile` or `tox.ini` for grannular options.

### 🔧 Tooling Overview
- **Virtual environment for testing**: [`tox`](https://tox.wiki/en/4.28.1/)
- **Unit/Integration Tests**: [`pytest`](https://docs.pytest.org/)
  - Will look for tests as `tests/test_*.py`
- **Code Style**:
  - [`ruff`](https://github.com/charliermarsh/ruff): formatting and linting
  - [`mypy`](http://mypy-lang.org/): static type checking
  - [`pre-commit`](https://pre-commit.com/): wraps up `ruff` and `mypy` and cleans-up staged files before commit and for PRs to `devel`.


## 📚 Documentation
The docs are generated via [sphinx](https://www.sphinx-doc.org/en/master/) and several sphinx addons.

To build the documentation locally:

1. Install the package with documentation extras:
   ```bash
   pip install git+https://github.com/dvlaykov/sgs_tools.git@devel#egg=sgs_tools[doc]
   ```
   Or, if you're using Poetry:
   ```bash
   poetry install --with doc

1. To automatically get all the dependencies, reinstall the package adding `  'sgs_tools[doc]' ` to the end of the install command.
1. To generate the docs run `make doc` from the top level in the repository. This will generate an html version of the documentation.
3. The resulting docs can be accessed from `<repo_directory>/doc/_build/html/index.html`

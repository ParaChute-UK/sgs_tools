# sgs_tools

Python tools for sub-grid scale (SGS) fluid dynamics analysis.

## 🚀 Install

> [!Caution]
> This package is under active development. Interfaces, features and dependencies may change with little notice.
>
---
> [!Important]
> The `devel` branch is the actively maintained branch containing the latest features and fixes.
>

### 🏗️ Development Version

  To install the latest development version (from GitHub `devel` branch):

  ```console
  pip install git+https://github.com/parachute-uk/sgs_tools.git@devel
  ```

  > [!Note]
   **Requires Python >=3.11**. Installation on older versions will fail with possibly unclear error messages.

## ▶️ Usage

  The package can be used both as a Python library and from the command line.

  For example, run one of the included analysis scripts directly, e.g.

  ```console
    cs_dynamic --help
  ```

  In Python, import the usual way

  ```python
    import sgs_tools
    print (sgs_tools.__version__)
  ```

  See [documentation](https://parachute-uk.github.io/sgs_tools/) for available module and functionality and CLI scripts for sample usage.

## 📚 Documentation

The documentation is hosted [on GitHub](https://parachute-uk.github.io/sgs_tools/).
(It is updated via GitHub Actions, so may be a few minutes behind the latest PR merge.)

To build the documentation locally:

  1. Install the package with suppor for building the documentation ([sphinx](https://www.sphinx-doc.org/en/master/) and addons)

       ```console
       pip install git+https://github.com/parachute-uk/sgs_tools.git@devel#egg=sgs_tools[doc]
       ```

     Or, if you're using Poetry

       ```console
       poetry install --with doc
       ```

  2. Generate

        ```console
        poetry run poe doc
        ```

  3. The rendered documentation can be accessed from `<repo_directory>/documentation/index.html`.

## 🤝 Contributing

Contributions of all kinds are wellcome — bug reports, feature requests, documentation improvements, and pull requests.

> Open an [Issue](https://github.com/parachute-uk/sgs_tools/issues) for questions, ideas, or feedback. We appreciate your input!
> See the [Development](#-development) for PR instructions.

## 🧪 Development

The dev tools are managed using [Poetry](https://python-poetry.org/docs/).

> [!Note]
> In case you are not familiar, [Poetry](https://python-poetry.org/docs/), is a Python packaging and publishing tool for dependency management and development workflows.
You can still use `pip` for user installations, but we recommend Poetry for contributing to this project.

### 🥼 Dev Installation

  1. Clone the repository

      ```console
      git clone https://github.com/parachute-uk/sgs_tools.git
      cd sgs_tools
      ```

  2. Create and activate a virtual environment in the preferred way (venv, conda, ...) **[Optional but Recommended]**

  3. [Install Poetry](https://python-poetry.org/docs/#installation) (if not already installed), e.g.

      ```console
      curl -sSL https://install.python-poetry.org | python3 -
      ```

  4. Install dependencies including dev tools. This adds dev tools including `poethepoet`, `tox`, `pytest`, `ruff`, `mypy`, and `pre-commit` to the dependencies.

      ```console
      poetry install --with dev
      ```

  5. Activate Git pre-commit hooks to help clean up formatting etc. on commit.

      ```console
      pre-commit install
      ```

        This will slow down commits somewhat. You can add `--no-verify` to the `git commit`  commands, but this is not advised, and the commit may be rejected by the remote on `push`.

### 🔬 Testing tools

- **Unit/Integration Tests**: [`pytest`](https://docs.pytest.org/)
  - Will look for tests as `test/test_*.py`
- **Code Style**:
  - [`ruff`](https://github.com/charliermarsh/ruff): formatting and linting
  - [`mypy`](http://mypy-lang.org/): static type checking
  - [`pre-commit`](https://pre-commit.com/): wraps up `ruff` and `mypy` and cleans-up staged files before commit. Automatically used in PRs to `devel`.
- **Virtual environment mamagement**: [`tox`](https://tox.wiki/en/4.28.1/)
- **Dev task orchestration**: [PoethePoet](https://poethepoet.natn.io)

> [!Note]
>
> All the tools are automatically installed with the [Dev installation](#-dev-installation).
>

### 🧷 Code check utilities

The following `poe` tasks are available (see `pyproject.toml:tool.poe.tasks`):

- `poetry run poe style`  &mdash; basic `ruff` formatting;

- `poetry run poe lint` &mdash; standard pre-commit checks run on **all files** (not just the staged ones). This  includes `ruff` formatting, linting and basic `mypy` type-hint checks. These gets automatically run before each commit on the staged files.

- `poetry run poe mypy`  &mdash; more comprehensive type checks with `--install-type`

- `poetry run poe test`  &mdash; the (unit/integration) testing suite, should take a few minutes. Should be run before a PR.

- `poetry run poe check`  &mdash; equivalent to `[lint, mypy, test]`

> [!TIP] For brevity you may want to set-up the alias
> `poe=poetry run poe`

#### Compatibility testing

Support across python versions is mamanged with [tox](https://tox.wiki/en/4.28.1/). Call

``` console
tox
```

to run the standard tests in isolated environments for each supported python version. See e.g. `tox.ini` for supported versions. This assumes that the corresponding python interpreter can be found in the `PATH`. For install with a simple source script

``` sh
# activate-dev.sh
conda activate sgs_tools # main env with all the dev tools
# only add interpreters, don't override main env
export PATH="$HOME/.conda/envs/py311/bin:$PATH"
export PATH="$HOME/.conda/envs/py312/bin:$PATH"
export PATH="$HOME/.conda/envs/py313/bin:$PATH"
export PATH="$HOME/.conda/envs/py314/bin:$PATH"
```

#### 🔀 Contirbuting & Pull Requests

> [!NOTE]
>
> PRs should be submitted to the `devel` branch.
>
> Consider adding a test for any new functinonality. Place new tests in `test/test_*.py`

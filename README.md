# sgs_tools

Python tools for sub-grid scale (SGS) fluid dynamics analysis.

## 🚀 Install

> [!Caution]
> This package is under active development. Interfaces, features and dependencies may change with little notice.
>
---
> [!Important]
> The `devel` branch is the actively maintained branch containing the latest features and fixes.
> The `main` branch contains tagged releases.

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

  See the [Documentation](https://parachute-uk.github.io/sgs_tools/) for available module and functionality and CLI scripts for sample usage.

## 📚 Documentation

The documentation is hosted [on GitHub](https://parachute-uk.github.io/sgs_tools/).
It is updated via GitHub Actions, so may be a few minutes behind the latest PR merge.

To build the documentation locally, call

```console
python build_doc.py --setup
```

for the first time, to install additional dependencies to your environment.
Thereafter, you can drop the `--setup` argument.
This will automatically pickup your regular installation (with `pip` or `poetry`).

>[!Note] The rendered documentation can be accessed from the entry point
> `<repo_directory>/documentation/index.html`

The documentation is auto-generaged with [Sphinx](https://www.sphinx-doc.org/en/master/).
The setup is found in `<repo_directory>/doc`.

>[!Note] If you are using the [dev installation](#-dev-installation) you can also call `poe doc` or `poe doc -- --setup`

## 🤝 Contributing

All contributions are wellcome — bug reports, feature requests, documentation improvements, and pull requests.

> [!Note]
> Open an [Issue](https://github.com/parachute-uk/sgs_tools/issues) for questions, ideas, or feedback.
> See  [Development](#-development) for PR instructions.

We appreciate your input!

## 🧪 Development

### 🔬 Dev tools


- **Environment management**: [Poetry](https://python-poetry.org/docs/)
- **Dev task orchestration**: [PoethePoet](https://poethepoet.natn.io), used as a Poetry plugin.
- **Unit/Integration Tests**: [`pytest`](https://docs.pytest.org/)
  - Will look for tests as `test/test_*.py`
- **Code Styling**:
  - [`ruff`](https://github.com/charliermarsh/ruff): formatting and linting
  - [`mypy`](http://mypy-lang.org/): static type checking
  - [`pre-commit`](https://pre-commit.com/): wraps up `ruff` and `mypy` and cleans-up staged files before commit. Automatically used in PRs to `devel`.
- **Multi-environment testing**: [`tox`](https://tox.wiki/en/4.28.1/)
  - use this for any changes that touch the project management, e.g. dependencies, etc.

> [!Note]
>
> All the tools apart from Poetry are automatically installed with the [Dev installation](#-dev-installation).
>

### 🥼 Dev Installation

  The dev tools are managed with [Poetry](https://python-poetry.org/docs/) and the dev tasks &mdash; with [PoethePoet](https://poethepoet.natn.io).

  > [!Note]
  > You can still use `pip` for user installations, but Poetry is prefered for codebase dev.

  1. Clone the repository

      ```console
      git clone https://github.com/parachute-uk/sgs_tools.git
      cd sgs_tools
      ```

  2. Create and activate a virtual environment in the preferred way (venv, conda, ...) **[Optional but Recommended]**

  3. [Install Poetry](https://python-poetry.org/docs/#installation), if not already installed (preferably in a separate environment.)

  4. Install dependencies including dev tools. This adds dev tools including `poethepoet`, `tox`, `pytest`, `ruff`, `mypy`, and `pre-commit` to the dependencies.

      ```console
      poetry install --with dev
      ```

  5. Activate Git pre-commit hooks to help clean up formatting etc. on commit.

      ```console
      pre-commit install
      ```

        This will slow down commits somewhat. You can add `--no-verify` to the `git commit`  commands, but this is not advised, and the commit may be rejected by the remote on `push`.

### 🧷 Code check utilities

If you don't have an independent [poe](https://poethepoet.natn.io) installation, the following assumes the alias `poe=poetry run poe`

> [!TIP]
> For brevity you may want to place that in your `$HOME/.bashrc` or analogous location.

Either way, the following `poe` code-hygiene tasks are available (see `pyproject.toml:tool.poe.tasks`):

- `poetry run poe style`  &mdash; basic `ruff` formatting;

- `poetry run poe lint` &mdash; standard pre-commit checks run on **all files** (not just the staged ones). This  includes `ruff` formatting, linting and basic `mypy` type-hint checks. These gets automatically run before each commit on the staged files.

- `poetry run poe mypy`  &mdash; more comprehensive type checks with `--install-type`

- `poetry run poe test`  &mdash; the (unit/integration) testing suite, should take a few minutes. Should be run before a PR.

- `poetry run poe check`  &mdash; equivalent to `[lint, mypy, test]`

- `poe doc` &mdash; re-generate the docs (run with `-- --setup` the first time to install dependencies)

### Multi-environment testing

Support across python versions is mamanged with [tox](https://tox.wiki/en/4.28.1/).
Call `tox ` or better yet `tox -p` to run the standard environment matrix.
See `tox.ini` for the test environments.
This assumes that the corresponding python interpreter can be found in the `PATH`.
A simple source script can help with that, e.g. if the interpreters are managed with `conda`,

``` sh
# activate-dev.sh
conda activate sgs_tools # main env with all the dev tools
# only add interpreters, don't override main env
export PATH="$HOME/.conda/envs/py311/bin:$PATH"
export PATH="$HOME/.conda/envs/py312/bin:$PATH"
export PATH="$HOME/.conda/envs/py313/bin:$PATH"
export PATH="$HOME/.conda/envs/py314/bin:$PATH"
```

followed by

``` console
source activate-dev.sh
```

### 🔀 Contributing & Pull Requests

All contributions are welcome —  pull requests, bug reports, feature requests, etc.

> [!NOTE]
> Development happens on the `devel` branch.
> Pull requests should target `devel` (not `main`).

For full development and release workflow details, see
`CONTRIBUTING.md`

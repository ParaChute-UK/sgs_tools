# sgs_tools

Python tools for sub-grid scale (SGS) fluid dynamics analysis.





## 🚀 Install

> ⚠️ This package is under active development. Interfaces, features and dependencies may change without notice.
> The `devel` branch is the actively maintained branch containing the latest features and fixes.

### 🏗️ Development Version

  To install the latest development version (from GitHub `devel` branch):

  ```console
  pip install git+https://github.com/dvlaykov/sgs_tools.git@devel
  ```

  > **Requires Python >=3.11**. Installation on older versions will fail with possibly unclear error messages.




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
  See [documentation](https://dvlaykov.github.io/sgs_tools/) for available module and functionality and CLI scripts for sample usage.

## 📚 Documentation
The documentation is hosted [here](https://dvlaykov.github.io/sgs_tools/).
(It is updated via GitHub Actions, so may be a few minutes behind the latest PR merge.)

To build the documentation locally:


  1. Install the package with documentation extras ([sphinx](https://www.sphinx-doc.org/en/master/) and addons)
       ```console
         pip install git+https://github.com/dvlaykov/sgs_tools.git@devel#egg=sgs_tools[doc]
       ```
     Or, if you're using Poetry
       ```console
          poetry install --with doc
       ```

  2. Generate
        ```console
        make doc
        ```

  3. The rendered documentation can be accessed from `<repo_directory>/documentation/index.html`.


## 🤝 Contributing

We welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and pull requests.

> Open [issues](https://github.com/dvlaykov/sgs_tools) for questions, ideas, or feedback. We appreciate your input!

> To get started on a pull request follow the Development Setup instructions below.

- Clone the repository and create a feature branch from `devel`
- Use `make pre_commit` or `tox -e pre_commit` to fix common formatting/style issues
- Use `make test` or `tox` to run checks before submitting a PR
- Submit your PR to the `devel` branch


## 🧪 Development Setup & Tooling

The dev tools are managed using [Poetry](https://python-poetry.org/docs/).

> If you're unfamiliar with [Poetry](https://python-poetry.org/docs/), it's a Python packaging and publishing tool for dependency management and development workflows.
You can still use `pip` for user installations, but we recommend Poetry for contributing to this project.


### 🥼 Setup
  1. Clone the repository
      ```console
      git clone https://github.com/dvlaykov/sgs_tools.git
      cd sgs_tools
      ```
  2. Create and activate a virtual environment in the preferred way (venv, conda, ...) [Optional, Recommended]

  3. [Install Poetry](https://python-poetry.org/docs/#installation) (if not already installed), e.g.
      ```console
      curl -sSL https://install.python-poetry.org | python3 -
      ```

  4. Install all dependencies including dev tools:
      ```console
      poetry install --with dev
      ```

      > This adds dev tools like `tox`, `pytest`, `ruff`, `mypy`, and `pre-commit` to the dependencies.

      ```console
      poetry self add poetry-dynamic-versioning@latest
      ```
      > Install Poetry Dynamic Versioning plugin.
        Make sure to run this inside the Poetry environment of the git repo, not from 'conda', 'pip', etc.
        Check with ```poetry version``` that the Git tags, commit hashes / dirty markers are detected correctly.

  5. Activate Git hooks to help clean up formatting etc. on commit (this may slow down the commit somewhat).
      ```console
      pre-commit install
      ```

### 🔬 Testing & Style

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

  See the `Makefile` or `tox.ini` for more grannular options.

### 🔧 Tooling Overview
- **Virtual environment for testing**: [`tox`](https://tox.wiki/en/4.28.1/)
- **Unit/Integration Tests**: [`pytest`](https://docs.pytest.org/)
  - Will look for tests as `test/test_*.py`
- **Code Style**:
  - [`ruff`](https://github.com/charliermarsh/ruff): formatting and linting
  - [`mypy`](http://mypy-lang.org/): static type checking
  - [`pre-commit`](https://pre-commit.com/): wraps up `ruff` and `mypy` and cleans-up staged files before commit. Automatically used in PRs to `devel`.

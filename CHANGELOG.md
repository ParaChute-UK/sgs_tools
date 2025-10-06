## Changelog

All notable changes to this project are documented in this file.

### [Under development]

#### Added
  - a central read dispatcher (#33)
  - dynamic version CL arg and tags in netcdf outputs of CLIs

#### Fixes
  - turn geometry and diagnostics into sub-packages (#34)
  - dynamic versioning (instructions and docs)
  - enhancements to post_process

### [0.1.0] - 2025-08-22

#### Added
  - Auto-deployment of docs on GitHub pages (#28, #30)
  - packaging manager setuptools -> poetry (#27)
  - Apache 2.0 license (#24)
  - Dev tools: pre-commit hooks, ruff, pytest, tox, mypy (#19)
  - Extra diagnostics tools: (anistropy, spectra, profiles) with a CLI entrypoint (#9, #15)
  - New SGS models and upgraded CS calculation script (#4)

#### Changes and Fixes
  - temporary IO depependency change to netcdf4+h5netcdf (#16)
  - Refactor core modules, use Protocol, improve dynamic model logic (#5, #9, #15)
  - CLI entry points upgrades for CS generation and analysis (#2, #3, #4, #9)
  - Improved test coverage
  - Partial update of Readme and Docs

### [0.0.1]
  - dynamic version
  - fixes to packaging and dependencies
  - upgrade IO support to xarray 2023.9
  - clean-up of main script `CS_calculations.py` (exposed as a cli)

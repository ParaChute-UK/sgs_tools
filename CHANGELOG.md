<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project are documented in this file.

## [WIP]

### Break backward compatibility

- CLI
  - in sim_comparison/ref_comparison: rename `--plot_style_file` -> `--plot_styles`, as it can take a json string or a path to a json file. (#48)
  - Read all fields when `requested_fields` is empty (#40)
  - change format names: 'um' -> 'um_ideal' and 'um_real'

### Added

- update CLI:
  - Dynamic version arg and tags in the netCDF output
    - better non-interactive backend support (#48)
    - New `--skip_clouds` and `--only_diff` arguments
- I/O:
  - support for reading real-case um sims (lat-lon grid)
  - Central read dispatcher (#33)
  - Accept string inputs (#42)
  - Custom field-names lookup for UM files with `default_field_names_dict` (#48)
- Debug utility for memory usage tracking
- Updated packaging and test environment management
- Enhanced GH Action workflows with stricter merge protections
- Updated documentation

### Fixes

- turn geometry and diagnostics into sub-packages (#34)
- dynamic versioning (instructions and docs)
- enhancements to post_process
- unify output filename convention
- ref_comparison CLI: run with non-default plot_style
- CLI plotting: add iterators for streaming figures to help memory optimization

## [0.1.0] - 2025-08-22

### Added

- Auto-deployment of docs on GitHub pages (#28, #30)
- packaging manager setuptools -> poetry (#27)
- Apache 2.0 license (#24)
- Dev tools: pre-commit hooks, ruff, pytest, tox, mypy (#19)
- Extra diagnostics tools: (anistropy, spectra, profiles) with a CLI entrypoint (#9, #15)
- New SGS models and upgraded CS calculation script (#4)

### Fixes

- temporary IO depependency change to netcdf4+h5netcdf (#16)
- Refactor core modules, use Protocol, improve dynamic model logic (#5, #9, #15)
- CLI entry points upgrades for CS generation and analysis (#2, #3, #4, #9)
- Improved test coverage
- Partial update of Readme and Docs

## [0.0.1]

- dynamic version
- fixes to packaging and dependencies
- upgrade IO support to xarray 2023.9
- clean-up of main script `CS_calculations.py` (exposed as a cli)

# Contributing Guide

## Workflow

- Work on feature branches from `devel`
- Open PRs into `devel` (not `main`)
- Ensure all checks pass before merge

## Pre-commit

Install hooks once:

```bash
    pre-commit install
```

Run manually if needed:

``` bash
    pre-commit run --all-files
```

## Releases

- Releases are made from `devel`
- Tag the release:

```bash
    git tag vX.Y.Z
    git push origin vX.Y.Z
```

- Then open a PR from `devel` → `main`

## Notes

- `main` is release-only
- CI enforces formatting, typing, and versioning

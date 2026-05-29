import glob
from collections.abc import Iterable
from pathlib import Path

import xarray as xr


def parse_fname_pattern(fname_pattern: str | Path) -> list[Path]:
    """
    parse any glob wildcards from ``fname_pattern``
    return a list of concrete Paths
    """
    print(f"Parsing {fname_pattern}")
    # return list because of incomplete typehints of xr.open_mfdataset
    return [Path(p) for p in glob.glob(str(fname_pattern), recursive=True)]


def standardize_varnames(
    ds: xr.Dataset, field_names_convention: dict[str, str]
) -> xr.Dataset:
    """rename variables in ``ds`` using ``field_names_dict``

    :param ds: input dataset
    :return: dataset with renamed variables
    """
    restricted_dict = {k: v for k, v in field_names_convention.items() if k in ds}
    return ds.rename(restricted_dict)


def restrict_ds(ds: xr.Dataset, fields: Iterable[str]) -> tuple[xr.Dataset, set[str]]:
    """restrict the dataset to fields of interest and rename using fields dict

    :param ds: input dataset
    :param fields: list of fields to restrict to, must be contained by `ds`
    :return: dataset with renamed variables
    """
    intersection = [k for k in fields if k in ds]
    missing_fields = {k for k in fields if k not in intersection}
    # print ("Missing fields:", missing_fields)
    return ds[intersection], missing_fields

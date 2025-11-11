from typing import Dict, Iterable

import xarray as xr


def standardize_varnames(
    ds: xr.Dataset, field_names_convention: Dict[str, str]
) -> xr.Dataset:
    """rename variables in ``ds`` using ``field_names_dict``

    :param ds: input dataset
    :return: dataset with renamed variables
    """
    restricted_dict = {k: v for k, v in field_names_convention.items() if k in ds}
    return ds.rename(restricted_dict)


def restrict_ds(ds: xr.Dataset, fields: Iterable[str]) -> xr.Dataset:
    """restrict the dataset to fields of interest and rename using fields dict

    :param ds: input dataset
    :param fields: list of fields to restrict to, must be contained by `ds`
    :return: dataset with renamed variables
    """
    intersection = [k for k in fields if k in ds]
    missing_fields = {k for k in fields if k not in intersection}
    # print ("Missing fields:", missing_fields)
    return ds[intersection], missing_fields

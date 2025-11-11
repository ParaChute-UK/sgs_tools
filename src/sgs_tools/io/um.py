import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr

from sgs_tools.geometry.staggered_grid import interpolate_to_grid

base_fields_dict = {
    "U_COMPNT_OF_WIND_AFTER_TIMESTEP": "u",
    "V_COMPNT_OF_WIND_AFTER_TIMESTEP": "v",
    "W_COMPNT_OF_WIND_AFTER_TIMESTEP": "w",
    "THETA_AFTER_TIMESTEP": "theta",
    #'TEMPERATURE_ON_THETA_LEVELS' : 'T',
    # 'PRESSURE_AT_THETA_LEVELS_AFTER_TS' : 'P'
}
""

Water_dict = {
    "CLD_LIQ_MIXING_RATIO__mcl__AFTER_TS": "q_l",
    "CLD_ICE_MIXING_RATIO__mcf__AFTER_TS": "q_i",
    "LARGE_SCALE_RAINFALL_RATE____KG_M2_S": "rain",
    "GRAUPEL_MIXING_RATIO__mg__AFTER_TS": "q_g",
    "SPECIFIC_HUMIDITY_AFTER_TIMESTEP": "q_v",
}
""

Smagorinsky_dict = {
    "SMAG__S__SHEAR_TERM_": "s_smag",
    "SMAG__VISC_M": "smag_visc_m",
    "SMAG__VISC_H": "smag_visc_h",
    "SHEAR_AT_SCALE_DELTA": "s",
    "MIXING_LENGTH_RNEUTML": "csDelta",
    "CS_THETA": "cs_theta",
    "TURBULENT_KINETIC_ENERGY": "tke",
    "GRADIENT_RICHARDSON_NUMBER": "Richardson",
}
""

dynamic_SGS_dict = {
    "CS_SQUARED_AT_2_DELTA": "cs2d",
    "CS_SQUARED_AT_4_DELTA": "cs4d",
    "CS_THETA_AT_SCALE_2DELTA": "cs_theta_2d",
    "CS_THETA_AT_SCALE_4DELTA": "cs_theta_4d",
}
""

dynamic_anisotropic_SGS_dict = {
    "RHOKM_DIFF_COEFF___LOCAL_SCHEME": "smag_visc_m_vert",
    "RHOKH_DIFF_COEFF___LOCAL_SCHEME": "smag_visc_h_vert",
    "CS_1": "cs_1",
    "CS_2": "cs_2",
    "CS_3": "cs_3",
    "CS_THETA_1": "cs_theta_1",
    "CS_THETA_2": "cs_theta_2",
    "CS_THETA_3": "cs_theta_3",
}
""

dynamic_SGS_diag_dict = {
    "LijMij_CONT_TENSORS": "lm",
    "QijNij_CONT_TENSORS": "qn",
    "MijMij_CONT_TENSORS": "mm",
    "NijNij_CONT_TENSORS": "nn",
    "HjTj_CONT_VECTORS": "ht",
    "TjTj_CONT_VECTORS": "tt",
    "RjFj_CONT_VECTORS": "rf",
    "FjFj_CONT_VECTORS": "ff",
    "SHEAR_AT_SCALE_2DELTA": "s2d",
    "SHEAR_AT_SCALE_4DELTA": "s4d",
    "D11_TENSOR_COMPONENT": "diag11",
    "D22_TENSOR_COMPONENT": "diag22",
    "D33_TENSOR_COMPONENT": "diag33",
    "D13_TENSOR_COMPONENT": "diag13",
    "D23_TENSOR_COMPONENT": "diag23",
    "D12_TENSOR_COMPONENT": "diag12",
    "Lagrangian_averaged_LijMij_tensors": "LM",
    "Lagrangian_averaged_MijMij_tensors": "MM",
    "Lagrangian_averaged_QijNij_tensors": "QN",
    "Lagrangian_averaged_NijNij_tensors": "NN",
    "Lagrangian_averaged_HjTj_vector": "HT",
    "Lagrangian_averaged_TjTj_vector": "TT",
    "Lagrangian_averaged_RjFj_vector": "RF",
    "Lagrangian_averaged_FjFj_vector": "FF",
    "Tdecorr_momentum": "Tdecorr_momentum",
    "Tdecorr_heat": "Tdecorr_heat",
    "Richardson": "Richardson_diag",
}
""

field_names_dict = (
    base_fields_dict
    | Water_dict
    | Smagorinsky_dict
    | dynamic_SGS_dict
    | dynamic_SGS_diag_dict
    | dynamic_anisotropic_SGS_dict
)


# IO
# open datasets
def read_stash_files(fname_pattern: Path, chunks: Any = "auto") -> xr.Dataset:
    """combine a list of output Stash files

    :param fname_pattern: filename(s) to read. Will be interpreted as a glob pattern.
    :return: `xarray.Dataset` with all available variables
    """

    print(f"Reading {fname_pattern}")
    # parse any glob wildcards in directory or filena
    # turn parsed into list because of incomplete typehints of xr.open_mfdataset
    parsed = list(
        Path(fname_pattern.root).glob(
            str(Path(*fname_pattern.parts[fname_pattern.is_absolute() :]))
        )
    )
    print(f"Reading {parsed}")
    dataset = xr.open_mfdataset(parsed, chunks="auto", parallel=True, engine="h5netcdf")
    return dataset


# Pre-process input UM arrays
def rename_variables(ds: xr.Dataset) -> xr.Dataset:
    """rename STASH variables:
       UM STASH varaibles adopt their `long_name` with special characters replaced by '_'.
       The stash code is retained as an attribute for back-searches.
       Spacial coordinates/dimesions are renamed to ``z_{theta|rho}`` and ``{x|y}_{face|centre}``.
       Time coordinate becomes ``t`` and ``t_0``.

    :param ds: input dataset
    :return: dataset with renamed variables
    """

    def varname_str(varStr: str):
        """replace special characters in varStr with "_" """
        return re.sub(r"\W|^(?=\d)", "_", varStr)

    varname_dict = {}
    for var in ds:
        lname = ds[var].attrs.get("long_name", "None").rstrip("|").rstrip()
        if "STASH" in str(var) and lname is not None:
            ds[var].attrs["original_vname"] = var
            varname_dict[var] = varname_str(lname)
    ds = ds.rename(varname_dict)

    # swap vertical dimension to height about sea level (in m) to better compare across simulations
    vertical_dim_map = {
        "thlev_eta_theta": "thlev_zsea_theta",
        "thlev_bl_eta_theta": "thlev_bl_zsea_theta",
        "rholev_eta_rho": "rholev_zsea_rho",
    }
    vertical_dim_map = {
        k: v for k, v in vertical_dim_map.items() if k in ds and v in ds
    }
    ds = ds.swap_dims(vertical_dim_map)

    # rename dimension fields by association with a primary field
    dim_names = {
        "thlev_zsea_theta": "z_theta",
        "rholev_zsea_rho": "z_rho",
        "latitude_t": "y_theta",
        "longitude_t": "x_theta",
        "latitude_cu": "y_cu",
        "longitude_cu": "x_cu",
        "latitude_cv": "y_cv",
        "longitude_cv": "x_cv",
    }
    intersection = {k: v for k, v in dim_names.items() if k in ds}
    ds = ds.rename(intersection)

    # swap to an easy time-dimension
    tname = "min15T0"
    if "min15T0_0" in ds:
        torigin = ds["min15T0_0"][0]
    else:
        torigin = ds["min15T0"][0]
    for tsuffix in "", "_0":
        if tname + tsuffix in ds:
            delta_t = np.rint(
                (ds[tname + tsuffix] - torigin) / np.timedelta64(1, "m")
            ).astype(int)
            ds = ds.assign_coords({"t" + tsuffix: (tname + tsuffix, delta_t.data)})
            ds["t" + tsuffix].attrs = {
                "standard_name": "time",
                "axis": "T",
                "unit": "min",
            }
            ds = ds.swap_dims({tname + tsuffix: "t" + tsuffix})
    return ds


def standardize_varnames(ds: xr.Dataset) -> xr.Dataset:
    """rename variables in ``ds`` using ``field_names_dict``

    :param ds: input dataset
    :return: dataset with renamed variables
    """
    restricted_dict = {k: v for k, v in field_names_dict.items() if k in ds}
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


# unify coordinate names and implement correct x-spacing for UM ideal sims
# xarray doesn't handle duplicate dimensions well, so use clunkily split-rename-merge
def unify_coords(ds: xr.Dataset, res: float) -> xr.Dataset:
    """unify coordinate names

    implement correct x-spacing using ``res``, assume ``res`` is given in the correct units
    rename coordinates with reference to a logically-cartesian grid

    :param ds: input dataset
    :param res: resolution of dataset -- to create correct x-y grid for idealised runs (the existing one is in lat-lon coords)
    :return: dataset with renamed variables

    """
    # actual coordinates
    x_face = np.linspace(
        0, (ds.x_theta.size) * res, num=ds.x_theta.size, endpoint=False
    )
    x_centre = x_face + res / 2

    # split into centered and staggered variables
    cent_vars = [x for x in ds if "x_theta" in ds[x].dims and "y_theta" in ds[x].dims]
    stag_vars = [x for x in ds if x not in cent_vars]

    # rename dimensions/coords of staggered variables
    ds_stag = ds[stag_vars]
    if ds_stag:
        ds_stag["x_centre"] = xr.DataArray(
            x_centre, coords={"x_cv": ds.x_cv}, dims="x_cv", name="x_centre"
        )
        ds_stag["y_centre"] = xr.DataArray(
            x_centre, coords={"y_cu": ds.y_cu}, dims="y_cu", name="y_centre"
        )
        ds_stag["x_face"] = xr.DataArray(
            x_face, coords={"x_cu": ds.x_cu}, dims="x_cu", name="x_face"
        )
        ds_stag["y_face"] = xr.DataArray(
            x_face, coords={"y_cv": ds.y_cv}, dims="y_cv", name="y_face"
        )

        ds_stag = ds_stag.swap_dims(
            {
                "x_cu": "x_face",
                "x_cv": "x_centre",
                "y_cu": "y_centre",
                "y_cv": "y_face",
            }
        )

    # rename dimensions/coords of centred variables
    ds_cent = ds[cent_vars]
    if ds_cent:
        ds_cent["x_centre"] = xr.DataArray(
            x_centre, coords={"x_theta": ds.x_theta}, dims="x_theta", name="x_centre"
        )
        ds_cent["y_centre"] = xr.DataArray(
            x_centre, coords={"y_theta": ds.y_theta}, dims="y_theta", name="y_centre"
        )
        ds_cent = ds_cent.swap_dims({"x_theta": "x_centre", "y_theta": "y_centre"})

    if ds_stag and ds_cent:
        ds = xr.merge([ds_stag, ds_cent])
    elif ds_stag:
        ds = ds_stag
    elif ds_cent:
        ds = ds_cent
    else:
        raise ValueError("No recocognized coordinates in ds")
    return ds


def compose_diagnostic_tensor(ds: xr.Dataset) -> xr.Dataset:
    diag_ij = xr.concat(
        [
            xr.concat([ds.diag11, ds.diag12, ds.diag13], "c1"),
            xr.concat([ds.diag12, ds.diag22, ds.diag23], "c1"),
            xr.concat([ds.diag13, ds.diag23, ds.diag33], "c1"),
        ],
        "c2",
    )
    diag_ij.name = "Diag_ij"
    diag_ij.attrs["long_name"] = "Diagnostic tensor"

    ds["Diag_ij"] = diag_ij
    ds = ds.drop_vars(["diag11", "diag22", "diag33", "diag12", "diag13", "diag23"])
    return ds


def data_ingest_UM(
    fname_pattern: Path,
    res: float,
    requested_fields: list[str] = ["u", "v", "w", "theta"],
) -> xr.Dataset:
    """read and pre-process UM data using sgs_tools naming convention.
    Any unknown fields will retain their original names.

    :param fname_pattern: UM NetCDF diagnostic file(s) to read. will be interpreted as a glob pattern. (should belong to the same simulation)
    :param res: horizontal resolution (will use to overwrite horizontal coordinates). **NB** works for ideal simulations
    :param requested_fields: list of fields to retain in ds, if falsy will retain all.
    """
    # all the fields we will need for the Cs calculations
    simulation = read_stash_files(fname_pattern)
    # parse UM stash codes into variable names
    simulation = rename_variables(simulation)

    # rename to sgs_tools naming convention
    simulation = standardize_varnames(simulation)

    # restrict to interesting fields and rename to simple names
    if requested_fields:
        simulation, _ = restrict_ds(simulation, fields=requested_fields)
    assert len(simulation) > 0, "None of the requested fields are available"
    # unify coordinates
    simulation = unify_coords(simulation, res=res)
    return simulation


def data_ingest_UM_on_single_grid(
    fname_pattern: Path,
    res: float,
    requested_fields: list[str] = ["u", "v", "w", "theta"],
) -> xr.Dataset:
    """read, pre-process UM data and interpolate to a cell-centred grid
    Any unknown fields will retain their original names.

    :param fname_pattern: UM NetCDF diagnostic file(s) to read. will be interpreted as a glob pattern. (should belong to the same simulation)
    :param res: horizontal resolution (will use to overwrite horizontal coordinates). **NB** works for ideal simulations
    :param  requested_fields: list of fields to retain in ds, if falsy will retain all.
    """
    # read, constrain fields, unify grids
    simulation = data_ingest_UM(fname_pattern, res, requested_fields)

    # interpolate all vars to a cell-centred grid
    centre_dims = ["x_centre", "y_centre", "z_theta"]
    simulation = interpolate_to_grid(simulation, centre_dims, drop_coords=True)
    # rename spatial dimensions to 'xyz'
    simple_dims = ["x", "y", "z"]
    dim_names = {d_new: d_old for d_new, d_old in zip(centre_dims, simple_dims)}
    simulation = simulation.rename(dim_names)

    return simulation

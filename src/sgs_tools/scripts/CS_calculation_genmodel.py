from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import xarray as xr
from dask.diagnostics import ProgressBar
from numpy import inf
from xarray.core.types import T_Xarray

from sgs_tools.geometry.staggered_grid import (
    compose_vector_components_on_grid,
)
from sgs_tools.geometry.vector_calculus import grad_scalar
from sgs_tools.io.read import read
from sgs_tools.physics.fields import omega_from_vel, strain_from_vel
from sgs_tools.scripts.arg_parsers import (
    add_dask_group,
    add_input_group,
    add_plotting_group,
)
from sgs_tools.sgs.CaratiCabot import DynamicCaratiCabotModel
from sgs_tools.sgs.dynamic_coefficient import (
    LillyMinimisation1Model,
    LillyMinimisation2Model,
    LillyMinimisation3Model,
)
from sgs_tools.sgs.dynamic_sgs_model import (
    DynamicModelProtcol,
)
from sgs_tools.sgs.filter import Filter, box_kernel, weight_gauss_3d, weight_gauss_5d
from sgs_tools.sgs.Kosovic import DynamicKosovicModel
from sgs_tools.sgs.Smagorinsky import (
    DynamicSmagorinskyHeatModel,
    DynamicSmagorinskyVelocityModel,
    SmagorinskyHeatModel,
    SmagorinskyVelocityModel,
)
from sgs_tools.util.timer import timer

# supported models
vel_models = ["Smag_vel", "Smag_vel_diag", "Carati", "Kosovic"]
theta_models = ["Smag_theta", "Smag_theta_diag"]
model_choices = ["all", "vel_all", "theta_all"] + vel_models + theta_models

model_name_map = {
    "Smag_vel": "Cs_isotropic",
    "Smag_vel_diag": "Cs_diagonal",
    "Carati": "Cs_CaratiCabot",
    "Kosovic": "Cs_Kosovic",
    "Smag_theta": "Ctheta_isotropic",
    "Smag_theta_diag": "Ctheta_diagonal",
}


def parse_args(arguments: Sequence[str] | None = None) -> Dict[str, Any]:
    parser = ArgumentParser(
        description="""Compute dynamic Smagorinsky coefficients as function
                        of scale from UM NetCDF output and store them in
                        a NetCDF files""",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    io = add_input_group(parser)
    io.add_argument(
        "output_path",
        type=Path,
        help="""
        Output directory, where to write netcdf output files.
        Will create/overwrite existing file and
        create any missing intermediate directories""",
    )
    io.add_argument(
        "--input_format", type=str, choices=["um", "monc", "sgs"], default="um"
    )

    add_plotting_group(parser)
    add_dask_group(parser)

    model = parser.add_argument_group("Model parameters")

    model.add_argument(
        "--sgs_model",
        type=str,
        nargs="+",
        default="all",
        choices=model_choices,
        help="Choice of models for which to compute dynamic coefficients.",
    )

    model.add_argument(
        "--filter_type",
        type=str,
        default="box",
        choices=["box", "gaussian"],
        help="Shape of filter kernel to use for scale separation.",
    )

    model.add_argument(
        "--filter_scales",
        type=int,
        default=(2,),
        nargs="+",
        help="Scales to perform filter at, in number of cells. "
        "If a single value is given, it will be used for all `regularize_filter_scales`. "
        "Otherwise, must provide as many values as for `regularize_filter_scales`",
    )

    model.add_argument(
        "--regularize_filter_type",
        type=str,
        default="box",
        choices=["box", "gaussian"],
        help="Shape of filter kernel used for coefficient regularization.",
    )

    model.add_argument(
        "--regularize_filter_scales",
        type=int,
        default=(2,),
        nargs="+",
        help="Scales to perform regularization at, in number of cells. "
        "If a single value is given, it will be used for all `filter_scale`. "
        "Otherwise, must provide as many values as for `filter_scale`",
    )

    # parse arguments into a dictionary
    args = vars(parser.parse_args(arguments))

    # check io group consistency
    if args["input_format"] == "um":
        assert args["h_resolution"] > 0, (
            "Missing required a positive h_resolution for UM datasets"
        )

    # parse negative values in the [t,z]_range
    if args["t_range"][0] < 0:
        args["t_range"][0] = -inf
    if args["t_range"][1] < 0:
        args["t_range"][1] = inf

    if args["z_range"][0] < 0:
        args["z_range"][0] = -inf
    if args["z_range"][1] < 0:
        args["z_range"][1] = inf

    # model parsing:
    if "all" in args["sgs_model"]:
        args["sgs_model"] = set(vel_models + theta_models)
    elif "vel_all" in args["sgs_model"]:
        args["sgs_model"].remove("vel_all")
        args["sgs_model"] += vel_models
    elif "theta_all" in args["sgs_model"]:
        args["sgs_model"].remove("theta_all")
        args["sgs_model"] += theta_models
    args["sgs_model"] = set(args["sgs_model"])
    # parameter consistency checks
    if args["filter_type"] == "gaussian":
        assert all([x in [2, 4] for x in args["filter_scales"]]), (
            "Gaussian filters only support scales 2 and 4 for now..."
        )

    # singleton filter_scales or regularize_filter_scales means broadcast against the other
    if len(args["filter_scales"]) == 1:
        args["filter_scales"] = args["filter_scales"] * len(
            args["regularize_filter_scales"]
        )

    if len(args["regularize_filter_scales"]) == 1:
        args["regularize_filter_scales"] = args["regularize_filter_scales"] * len(
            args["filter_scales"]
        )

    assert len(args["filter_scales"]) == len(args["regularize_filter_scales"])

    return args


def make_filter(shape: str, scale: int, dims=Sequence[str]) -> Filter:
    """make filter object. **NB** Choices are limited!!!

    :param shape: shape of filter kernel
    :param scale: length scale of filter kernel
    :param dims: dimensions along which to filter
    """
    if shape == "gaussian":
        if scale == 2:
            return Filter(weight_gauss_3d.persist(), dims)
        elif scale == 4:
            return Filter(weight_gauss_5d.persist(), dims)
        else:
            raise ValueError(f"Unsupported filter scale{scale} for gaussian filters")
    elif shape == "box":
        return Filter(box_kernel([scale, scale]).persist(), dims)
    else:
        raise ValueError(f"Unsupported filter shape {shape}")


def add_scale_coords(
    ds: T_Xarray, scales: list[float], regularization_scales: list[float]
) -> T_Xarray:
    """add scale dim and regularization_scale coordinate
    :param ds: input dataset/dataarray
    :param scales: the coordinates for the scale dimension
    :param regularization_scales: sequence of coordinates
    :return: the update dataset/dataarray
    """
    if "scale" not in ds.dims:
        ds = ds.expand_dims(scale=scales)
    else:
        ds = ds.assign_coords(scale=("scale", scales))
    ds = ds.assign_coords(regularization_scale=("scale", regularization_scales))
    ds["scale"].attrs["units"] = r"$\Delta x$"
    ds["regularization_scale"].attrs["units"] = r"$\Delta x$"
    return ds


def data_slice(
    ds: xr.Dataset, t_range: Sequence[float], z_range: Sequence[float]
) -> xr.Dataset:
    """restrit ds to the intervals inside [t,z]_range.
       Restrict a set of standard coordinate names for t and z.
    :param ds: input dataset/dataarray
    :param t_range: time interval
    :param t_range: verical interval
    """
    for z in "z", "z_rho", "z_theta":
        if z in ds:
            zslice = (z_range[0] <= ds[z]) * (ds[z] <= z_range[1])
            ds = ds.where(zslice, drop=True)
        assert not all(ds[var].size == 0 for var in ds), (
            f"Data z-slice {z_range} results in empty variables. "
            "Consider relaxing z_range"
        )
    for t in "t", "t_0":
        if t in ds:
            tslice = (t_range[0] <= ds[t]) * (ds[t] <= t_range[1])
            ds = ds.where(tslice, drop=True)
    assert not all(ds[var].size == 0 for var in ds), (
        f"Data t-slice {t_range} results in empty variables. Consider relaxing t_range"
    )
    return ds


def gather_model_inputs(simulation: xr.Dataset, req_fields: list[str]) -> xr.Dataset:
    # ensure velocity components are co-located
    simple_dims = ["x", "y", "z"]  # coordinates already exist in simulation
    ds = xr.Dataset()

    if "vel" in req_fields:
        if "vel" in simulation:
            ds["vel"] = simulation["vel"]
        else:
            ds["vel"] = compose_vector_components_on_grid(
                [simulation["u"], simulation["v"], simulation["w"]],
                simple_dims,
                name="vel",
                vector_dim="c1",
            )

    if "sij" in req_fields:
        # compute strain, rotation and potential temperature gradient
        if "sij" in simulation:
            ds["sij"] = simulation["sij"]
        else:
            ds["sij"] = strain_from_vel(
                ds["vel"],
                space_dims=simple_dims,
                vec_dim="c1",
                new_dim="c2",
                make_traceless=True,
            )

    if "omegaij" in req_fields:
        if "omegaij" in simulation:
            ds["omegaij"] = simulation["omegaij"]
        else:
            ds["omegaij"] = omega_from_vel(
                ds["vel"],
                space_dims=simple_dims,
                vec_dim="c1",
                new_dim="c2",
            )

    if "theta" in req_fields:
        ds["theta"] = simulation["theta"]

    if "grad_theta" in req_fields:
        if "grad_theta" in simulation:
            ds["grad_theta"] = simulation["grad_theta"]
        else:
            ds["grad_theta"] = grad_scalar(
                ds["theta"],
                space_dims=simple_dims,
                new_dim_name="c1",
                name="grad_theta",
            )

    return ds


def pre_process(args: dict[str, Any]) -> xr.Dataset:
    req_fields = {
        "um": ["u", "v", "w", "theta"],
        "monc": ["u", "v", "w", "theta"],
        "sgs": ["vel", "theta", "sij", "omegaij", "theta", "grad_theta"],
    }

    simulation = read(
        args["input_files"],
        args["input_format"],
        requested_fields=req_fields[args["input_format"]],
        resolution=args["h_resolution"],
    )
    # overwrite resolution if recorded during read
    if "h_resolution" in simulation.attrs:
        args["h_resolution"] = simulation.attrs["h_resolution"]

    with ProgressBar():
        with timer("Extract grid-based fields", "s"):
            # slice to the requested sub-domain
            simulation = data_slice(simulation, args["t_range"], args["z_range"])
            model_req_fields = []
            if args["sgs_model"].intersection(vel_models):
                model_req_fields += ["vel", "sij", "omegaij"]
            if args["sgs_model"].intersection(theta_models):
                model_req_fields += ["theta", "grad_theta"]
            simulation = gather_model_inputs(simulation, model_req_fields)
    # chunk
    chunks = {
        "z": args["z_chunk_size"],
        "t_0": args["t_chunk_size"],
        "x": -1,
        "y": -1,
        "c1": -1,
        "c2": -1,
    }
    # add caveate for degenerate t or z-slice that may drop a coordinate
    simulation = simulation.chunk(
        chunks={x: y for x, y in chunks.items() if x in simulation.dims}
    )
    simulation = simulation.persist()
    return simulation


def model_selection(
    model: str, simulation: xr.Dataset, h_resolution: float
) -> DynamicModelProtcol:
    if model == "Smag_vel":
        return DynamicSmagorinskyVelocityModel(
            SmagorinskyVelocityModel(
                vel=simulation.vel,
                strain=simulation.sij,
                cs=1.0,
                dx=h_resolution,
                tensor_dims=("c1", "c2"),
            ),
            LillyMinimisation1Model(contraction_dims=["c1", "c2"], coeff_dim="cdim"),
        )
    if model == "Smag_vel_diag":
        return DynamicSmagorinskyVelocityModel(
            SmagorinskyVelocityModel(
                vel=simulation.vel,
                strain=simulation.sij,
                cs=1.0,
                dx=h_resolution,
                tensor_dims=("c1", "c2"),
            ),
            LillyMinimisation1Model(contraction_dims=["c2"], coeff_dim="cdim"),
        )
    if model == "Carati":
        return DynamicCaratiCabotModel(
            simulation.sij,
            res=h_resolution,
            compoment_coeff=[1.0, 1.0, 1.0],
            n=[0.0, 0.0, 1.0],  # gravity direction
            vel=simulation.vel,
            tensor_dims=("c1", "c2"),
            minimisation=LillyMinimisation3Model(
                contraction_dims=["c1", "c2"], coeff_dim="cdim"
            ),
        )
    if model == "Kosovic":
        return DynamicKosovicModel(
            simulation.sij,
            simulation.omegaij,
            res=h_resolution,
            compoment_coeff=[1.0, 1.0],
            vel=simulation.vel,
            tensor_dims=("c1", "c2"),
            minimisation=LillyMinimisation2Model(
                contraction_dims=["c1", "c2"], coeff_dim="cdim"
            ),
        )
    if model == "Smag_theta":
        return DynamicSmagorinskyHeatModel(
            SmagorinskyHeatModel(
                vel=simulation.vel,
                grad_theta=simulation.grad_theta,
                strain=simulation.sij,
                ctheta=1.0,
                dx=h_resolution,
                tensor_dims=("c1", "c2"),
            ),
            simulation["theta"],
            LillyMinimisation1Model(contraction_dims=["c1"], coeff_dim="cdim"),
        )
    if model == "Smag_theta_diag":
        return DynamicSmagorinskyHeatModel(
            SmagorinskyHeatModel(
                vel=simulation.vel,
                grad_theta=simulation.grad_theta,
                strain=simulation.sij,
                ctheta=1.0,
                dx=h_resolution,
                tensor_dims=("c1", "c2"),
            ),
            simulation["theta"],
            LillyMinimisation1Model(contraction_dims=[], coeff_dim="cdim"),
        )
    else:
        raise ValueError(f"Unknown model {model}, choose from {model_choices}")


def compute_cs(
    dyn_model: DynamicModelProtcol,
    test_filters: list[Filter],
    reg_filters: list[Filter],
) -> xr.DataArray:
    cs_at_scale_ls = []
    for test_filter, reg_filter in zip(test_filters, reg_filters):
        # compute Cs
        cs = dyn_model.compute_coeff(test_filter, reg_filter)
        # force execution for timer logging
        cs_at_scale_ls.append(cs)
    cs_at_scale = xr.concat(cs_at_scale_ls, dim="scale")
    cs_at_scale = add_scale_coords(
        cs_at_scale,
        [f.scale() for f in test_filters],
        [f.scale() for f in reg_filters],
    )
    return cs_at_scale


def plot(args: dict[str, Any]) -> None:
    row_lbl = "scale"

    def wrap_label(text: str, width: int = 20) -> str:
        """
        Inserts `\n` at word boundaries so that no line exceeds `width` characters.
        """
        import textwrap

        return "\n".join(
            textwrap.wrap(
                text,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )

    figures = {}
    for model in args["sgs_model"]:
        mpath = args["output_path"] / f"{model_name_map[model]}.nc"

        with timer(f"Plotting {model}", "s"):
            model_data = xr.open_mfdataset(mpath)
            model_data = model_data[model]
            mean = model_data.mean(["x", "y"])
            if "cdim" in mean.dims:
                mean = mean.rename(cdim="c1")

            if "c1" in mean.dims:
                figures[model] = mean.plot(y="z", row=row_lbl, col="c1").fig  # type: ignore
            else:
                figures[model] = mean.plot(y="z", row=row_lbl).fig  # type: ignore
                # figures[model].axes[0].set_title("")
            for ax in figures[model].axes:
                ax.set_title(wrap_label(ax.get_title()))
            figures[model].suptitle(str(model).replace("_", " "), fontsize=14, y=1)
        if args["plot_path"] is not None:
            args["plot_path"].mkdir(parents=True, exist_ok=True)
            figures[model].savefig(args["plot_path"] / f"{model}.pdf", dpi=180)
    # interactive plotting out of time
    if args["plot_show"]:
        for name, fig in figures.items():
            fig.canvas.manager.set_window_title(name)  # set window title
            fig.show()
        plt.show()


def run(args: Dict[str, Any]) -> None:
    # read and pre-process simulation
    # read UM stasth files: data
    with timer("Read Dataset", "s"):
        simulation = pre_process(args)

    # check scales make sense
    nhoriz = min(simulation["x"].shape[0], simulation["y"].shape[0])
    for scale in args["filter_scales"]:
        assert scale in range(1, nhoriz), (
            f"scale {scale} must be less than horizontal number of grid cells {nhoriz}"
        )
    for scale in args["regularize_filter_scales"]:
        assert scale in range(1, nhoriz), (
            f"regularization_scale {scale} must be less than horizontal number of grid cells {nhoriz}"
        )

    with timer("Setup filtering operators"):
        test_filters = []
        regularization_filters = []
        for scale, regularization_scale in zip(
            args["filter_scales"], args["regularize_filter_scales"]
        ):
            test_filters.append(make_filter(args["filter_type"], scale, ["x", "y"]))
            regularization_filters.append(
                make_filter(
                    args["regularize_filter_type"], regularization_scale, ["x", "y"]
                )
            )

    for m in args["sgs_model"]:
        # setup dynamic model
        with timer(f"Coeff calculation SETUP for {model_name_map[m]} model", "s"):
            dynamic_model = model_selection(m, simulation, args["h_resolution"])

            coeff = compute_cs(
                dynamic_model,
                test_filters,
                regularization_filters,
            )
            # for multi-coefficient models
            if "cdim" in coeff.dims:
                coeff = coeff.rename({"cdim": "c1"})
            coeff.name = m

        out_fname = args["output_path"] / f"{model_name_map[m]}.nc"

        # trigger computation -- split for time logging
        with timer(f"Coeff calculation compute for {model_name_map[m]} model", "s"):
            with ProgressBar():
                coeff.compute()
        # write to disk
        with timer(f"Coeff calculation write for {model_name_map[m]} model", "s"):
            with ProgressBar():
                args["output_path"].mkdir(parents=True, exist_ok=True)
                coeff.to_netcdf(
                    out_fname,
                    mode="w",
                    compute=True,
                    unlimited_dims=["scale"],
                    engine="h5netcdf",
                )


def main():
    args = parse_args()
    print(args)
    with timer("Total execution time", "min"):
        run(args)

    # plot
    if args["plot_show"] or args["plot_path"] is not None:
        with timer("Plotting", "s"):
            # try:
            plot(args)
        # except:
        #   print("Failed in generating plots")


if __name__ == "__main__":
    main()

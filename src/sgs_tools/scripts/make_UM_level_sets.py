from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import f90nml  # type: ignore
from numpy import linspace

from sgs_tools.scripts.arg_parsers import add_version_group
from sgs_tools.util.gitinfo import print_git_state
from sgs_tools.util.path_utils import add_extension


def parser() -> dict[str, Any]:
    parser = ArgumentParser(
        description="Create a constant level set namelist for the UM from given model top height and number of levels"
    )
    add_version_group(parser)

    parser.add_argument(
        "output_file",
        type=Path,
        help="output path, will create/overwrite existing file and create any missing intermediate directories",
    )

    parser.add_argument(
        "z_top_of_model",
        type=float,
        help="top of model a.k.a. last theta level in meters",
    )

    parser.add_argument(
        "n_z_rho",
        type=int,
        help="number of rho-levels, will have one more theta level",
    )

    args = vars(parser.parse_args())

    assert args["n_z_rho"] > 0, f"Need a positive n_z, got {args['n_z_rho']}"
    assert args["z_top_of_model"] > 0, (
        f"Need a positive z_top_of_model, got {args['z_top_of_model']}"
    )

    args["output_file"] = add_extension(args["output_file"], ".nml")
    return args


def const_level_sets(n_rho):
    """create constant level sets"""
    eta_theta = linspace(0, 1, n_rho + 1)
    rho_init = (eta_theta[1] + eta_theta[0]) / 2
    rho_last = (eta_theta[-1] + eta_theta[-2]) / 2
    eta_rho = linspace(rho_init, rho_last, n_rho)
    return eta_theta, eta_rho


def main():
    args = parser()
    if args["version"]:
        print_git_state(args["version"])
        exit()

    # create namelist
    nml = f90nml.Namelist()
    nml["VERTLEVS"] = {}
    nml["VERTLEVS"]["z_top_of_model"] = [
        args["z_top_of_model"],
    ]
    nml["VERTLEVS"]["first_constant_r_rho_level"] = [
        args["n_z_rho"],
    ]
    nml["VERTLEVS"]["eta_theta"], nml["VERTLEVS"]["eta_rho"] = const_level_sets(
        args["n_z_rho"]
    )

    # keep close to the input format (decorative)
    nml.column_width = 80
    nml.end_comma = True
    nml.float_format = ".8f"

    # write
    nml.write(args["output_file"], force=True)


if __name__ == "__main__":
    main()

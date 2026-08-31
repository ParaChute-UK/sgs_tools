# yourpkg/cli_main.py
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence

from sgs_tools.scripts.arg_parsers import add_output_group
from sgs_tools.util.gitinfo import print_version_info


def main(arguments: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(
        description="Show package version",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    add_output_group(parser)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="""
        Output directory, where to write netcdf output files.
        Will create/overwrite existing file and
        create any missing intermediate directories""",
    )

    args = parser.parse_args(arguments)

    print_version_info(args.verbosity, args.raw)


if __name__ == "__main__":
    main()

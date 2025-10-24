# yourpkg/cli_main.py
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from sgs_tools.scripts.arg_parsers import add_output_group
from sgs_tools.util.gitinfo import print_version_info


def main():
    parser = ArgumentParser(
        description="Show package version",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    add_output_group(parser)
    args = parser.parse_args()
    print_version_info(args.verbosity)


if __name__ == "__main__":
    main()

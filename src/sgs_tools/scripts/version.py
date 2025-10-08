# yourpkg/cli_main.py
import argparse
from importlib.metadata import version as get_poetry_version

from sgs_tools.scripts.arg_parsers import add_version_group
from sgs_tools.util.gitinfo import print_git_state


def main():
    # Poetry dynamic version
    poetry_ver = get_poetry_version("sgs_tools")  # dynamic version from PDV
    print(f"SGS_tools: {poetry_ver}")

    parser = argparse.ArgumentParser()
    add_version_group(parser)
    args = parser.parse_args()
    print_git_state(max(args.git_version, 1))


if __name__ == "__main__":
    main()

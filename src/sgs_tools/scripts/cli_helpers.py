from pprint import pprint
from typing import Any

from sgs_tools.util.gitinfo import print_version_info

separator = "=" * 60


def print_header(title: str, version_verbosity=1) -> None:
    """Print a consistent, centered CLI header."""
    print(separator)
    print(f"SGS Tools: {title}")
    print_version_info(version_verbosity)
    print(separator)


def print_args_dict(args: dict[str, Any]) -> None:
    print("Arguments:")
    pprint(args, indent=2, width=80, sort_dicts=False)
    print(separator)

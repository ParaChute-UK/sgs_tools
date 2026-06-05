import sys
from collections.abc import Iterable

import matplotlib


def configure_matplotlib_backend(argv: Iterable | None = None, strict: bool = False):
    """
    Configure matplotlib backend based on presence of ``--plot_show``
    in argv or sys.argv

    Must be called before any import of matplotlib.pyplot.

    :param argv: argument list. If None will look at ``sys.argv`` instead
    :param strict: if True will raise a ``RuntimeError`` if it detects pyplot
        in ``sys.modules``. Otherwise just prints a Warning.
    """

    if "matplotlib.pyplot" in sys.modules:
        msg = "pyplot already imported before backend configuration"
        if strict:
            raise RuntimeError(msg)
        else:
            print("WARNING:", msg)

    argv = argv or sys.argv[1:]
    if "--plot_show" in argv:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")

import sys

import matplotlib


def configure_matplotlib_backend(argv=None, strict=False):
    """
    Configure matplotlib backend based on presence of --plot_show.

    Must be called before any import of matplotlib.pyplot.
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
    print("Backend:", matplotlib.get_backend())

from pathlib import Path


def render_figure(fig, path: Path, filename: str, show: bool) -> None:
    if path:
        fig.savefig(path / filename, dpi=180)
    if show:
        # pyplot imported lazily so it does not load
        # before configure_matplotlib_backend()
        import matplotlib.pyplot as plt

        plt.show()
    fig.clf()

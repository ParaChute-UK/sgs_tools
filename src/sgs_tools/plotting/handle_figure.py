from pathlib import Path

import matplotlib.pyplot as plt


def render_figure(fig, path: Path, filename: str, show: bool) -> None:
    if path:
        fig.savefig(path / filename, dpi=180)
    if show:
        plt.show()
    plt.close(fig)

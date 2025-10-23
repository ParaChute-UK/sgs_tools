import sys

from dask.diagnostics import ProgressBar


class TermimalProgressBar:
    def __init__(self):
        self.enabled = sys.stdout.isatty()
        self._pbar = ProgressBar() if self.enabled else None

    # Context manager support
    def __enter__(self):
        if self.enabled:
            self._pbar.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self._pbar.__exit__(exc_type, exc_val, exc_tb)

    # Proxy register() to global registration
    def register(self):
        if self.enabled:
            self._pbar.register()

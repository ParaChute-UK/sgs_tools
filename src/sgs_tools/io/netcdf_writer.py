from dataclasses import dataclass
from pathlib import Path

import xarray as xr

from ..util.timer import timer


@dataclass
class NetCDFWriter:
    overwrite: bool = False

    def check_filename(self, filename: Path) -> bool:
        return filename.exists()

    def write(self, array: xr.Dataset, filename: Path) -> None:
        if not self.overwrite and self.check_filename(filename):
            raise OSError(f"{filename} already exists. Won't overwrite.")
        filename.parent.mkdir(exist_ok=True, parents=True)
        array.to_netcdf(
            filename,
            mode="w",
            compute=True,
            engine="h5netcdf",
        )

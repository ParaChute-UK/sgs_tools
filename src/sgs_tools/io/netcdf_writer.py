from dataclasses import dataclass
from pathlib import Path

import xarray as xr


@dataclass
class NetCDFWriter:
    """A class to write xarray datasets to NetCDF files.

    :ivar overwrite: overwrite existing files if set to True. If False, raises an OSError if the file already exists.
    """

    overwrite: bool = False
    verbose: bool = False

    def check_filename(self, filename: Path) -> bool:
        """Check if the file exists.

        :param filename: Path to the NetCDF file.
        """
        return filename.exists()

    def write(self, array: xr.Dataset, filename: Path) -> None:
        """Write an xarray Dataset to a NetCDF file.

        :param array: xarray Dataset to write.
        :param filename: Path to the output NetCDF file.
        """

        if not self.overwrite and self.check_filename(filename):
            raise OSError(f"{filename} already exists. Won't overwrite.")
        filename.parent.mkdir(exist_ok=True, parents=True)
        array.to_netcdf(
            filename,
            mode="w",
            compute=True,
            engine="netcdf4",
            auto_complex=True,
        )
        if self.verbose:
            print(f"Wrote: {filename}")

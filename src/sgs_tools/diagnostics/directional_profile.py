from typing import Sequence

from pathlib import Path
import dask
import xarray as xr

from ..io.netcdf_writer import NetCDFWriter

def directional_profile(
    simulation: xr.Dataset, red_dims: Sequence[str], writer: NetCDFWriter, output_path: Path
) -> None:
    
    s = simulation.chunk({x:-1 for x in red_dims})
    mean = s.mean(dim=red_dims)
    std = s.std(dim=red_dims)
    # mean, std = dask.persist(mean, std)
    profile = xr.concat(
        [mean, std], dim=xr.DataArray(["mean", "std"], dims=["statistic"])
    )
    
    # rechunk for IO optimisation??
    # have to do explicit rechunking because UM date-time coordinate is an object
    #profile = profile.chunk({})
    writer.write(profile, output_path)
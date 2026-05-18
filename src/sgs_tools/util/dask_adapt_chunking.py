import math
import warnings
from collections.abc import Hashable, Mapping
from typing import Any

from xarray import Dataset


def adaptive_chunks(
    domain: Mapping[Hashable, int],
    min_chunks: dict[Hashable, int] | None = None,
    fixed_chunks: dict[Hashable, int] | None = None,
    itemsize: int = 8,
    target_mem_MB: float = 256.0,
) -> dict[Hashable, int]:
    """
    Compute adaptive chunk sizes for a given domain and memory target.

    :param domain : Full domain sizes per dimension, e.g.
      {'x': 1024, 'y': 1024, 'z': 70}.
    :param min_chunks : Minimum chunk sizes per adaptinve dimension;
      acts as lower bound and rounding base. If None, uses an empty dict.
    :param fixed_chunks : Chunk sizes that must remain fixed (subset of domain).
      If None, uses an empty dict.
    :param itemsize : Bytes per element, e.g. 8 for single float64 (default).
    :param target_mem_MB : Approximate desired memory per chunk in MB (default: 200).

    :return: Mapping of dimension names to computed chunk sizes.
    """
    if min_chunks is None:
        min_chunks = {}
    if fixed_chunks is None:
        fixed_chunks = {}
    if not all(x in domain for x in fixed_chunks):
        raise ValueError(
            f"fixed_chunks has dimensions missing from domain:{fixed_chunks.keys()}"
            f"vs. {domain.keys()}"
        )
    if not all(x in domain for x in min_chunks):
        raise ValueError(
            f"min_chunks has dimensions missing from domain:{min_chunks.keys()}"
            f"vs. {domain.keys()}"
        )
    if not all(x not in fixed_chunks for x in min_chunks):
        raise ValueError(
            f"min_chunks has dimensions missing from domain:{min_chunks.keys()}"
            f" vs. {domain.keys()}"
        )

    chunks = fixed_chunks
    adaptive_dims = [d for d in domain if d not in fixed_chunks and domain[d] > 1]
    if adaptive_dims:
        # memory already used by fixed dimensions
        used_elems = math.prod(fixed_chunks.values()) if fixed_chunks else 1
        target_bytes = target_mem_MB * 1024**2
        used_bytes = used_elems * itemsize
        # can't have less than one element in block
        remaining_bytes = max(target_bytes / used_bytes, itemsize)

        # aim for roughly same size for all adaptive dimensions
        total_elems_target = remaining_bytes / itemsize
        n = len(adaptive_dims)
        per_dim_target = total_elems_target ** (1 / n)

        (print(f"Targer Mem MB {target_mem_MB:.2f}"),)
        print(f"Target el size {total_elems_target:.2f}")
        print(f"Dim size {per_dim_target:.2f}")
        print(f"Adaptive dims: {adaptive_dims}")
        print(f"Fixed dims: {fixed_chunks.keys()}")
        for d in adaptive_dims:
            full = domain[d]
            base = min_chunks.get(d, 1)
            # rough chunksize between domain > raw > base
            raw = max(base, min(full, int(per_dim_target)))
            # round to nearest multiple of base, capped by full domain
            chunk = min(full, max(base, int(round(raw / base)) * base))
            chunks[d] = chunk

    # compute resulting memory footprint
    est_mem_MB = math.prod(chunks.values()) * itemsize / 1024**2
    if est_mem_MB > target_mem_MB * 1.2:  # allow 20% slack
        warnings.warn(
            f"Estimated chunk memory {est_mem_MB:.1f} MB"
            f"exceeds target {target_mem_MB:.1f} MB "
            "by 20% after enforcing minimum chunk sizes and rounding.",
            RuntimeWarning,
            stacklevel=2,
        )

    return chunks


def chunk_ds(ds: Dataset, chunks: dict[Hashable | str, Any]) -> Dataset:
    """rechunk the dataset according to the provided chunks
    needed for variables with different dimensions

    :param ds: input dataset
    :param chunks: chunks a dictionary lie
    :return: chunked dataset
    """
    chunk_dict = {}
    for d in ds.dims:
        if d in chunks:
            chunk_dict[d] = chunks[d]
        else:
            chunk_dict[d] = ds.chunks[d]

    return ds.chunk(chunk_dict)

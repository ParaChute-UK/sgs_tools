import contextlib
import gc
import threading
import time

import psutil
from dask.distributed import default_client, get_worker  # type: ignore[attr-defined]

unit_map = {"GB": 3, "MB": 2, "KB": 1, "B": 0}


@contextlib.contextmanager
def memory_delta(label, units="GB", title=""):
    """simple rss memory profiler for fine-grain code blocks"""
    if units not in unit_map:
        raise ValueError(f"Unrecognised units '{units}', choose from {unit_map.keys()}")
    scale = 1024 ** unit_map[units]

    if title:
        print(f"{title}")
    client = get_client()
    proc = psutil.Process()
    # before
    gc.collect()
    rss_before = proc.memory_info().rss / scale
    print(f"DEBUG {label}: rss_before = {rss_before:.3f} {units}")

    # Cluster RSS before (if client provided)
    if get_client() is not None:

        def _worker_rss():
            return psutil.Process().memory_info().rss

        cluster_before = client.run(_worker_rss)
        cluster_before_gb = {k: v / scale for k, v in cluster_before.items()}
        print(f"[{label}] Cluster RSS before per worker: {cluster_before_gb}")

    # compute
    try:
        yield
    finally:
        # after
        gc.collect()
        rss_after = proc.memory_info().rss / scale
        print(f"DEBUG {label}: rss_after = {rss_after:.3f} {units}")
        print(f"DEBUG {label}: rss_diff  = {rss_after - rss_before:.3f} {units}")

        if get_client() is not None:
            cluster_after = client.run(_worker_rss)
            cluster_after_gb = {k: v / scale for k, v in cluster_after.items()}
            cluster_diff_gb = {
                k: cluster_after_gb[k] - cluster_before_gb[k] for k in cluster_after_gb
            }
            print(f"[{label}] Cluster RSS after per worker: {cluster_after_gb}")
            print(f"[{label}] Cluster RSS diff  per worker: {cluster_diff_gb}")


@contextlib.contextmanager
def memory_watch(label, units="GB", interval=1.0):
    """Monitor peak resident memory (RSS).
    :param label: Identifier printed in the logs
    :param interval: Polling interval in seconds. Keep above 1 for low overheads.
    """
    if units not in unit_map:
        raise ValueError(f"Unrecognised units '{units}', choose from {unit_map.keys()}")
    scale = 1024 ** unit_map[units]
    client = get_client()
    # choose memory getter depending of dask cluster client or not
    if get_client() is not None:

        def get_mem():
            try:
                workers = client.scheduler_info()["workers"].values()
                return sum(w["memory"]["rss"] for w in workers) / scale
            except Exception:
                return float("nan")
    else:
        proc = psutil.Process()

        def get_mem():
            return proc.memory_info().rss / scale

    # monitor function
    def monitor():
        nonlocal peak
        while not stop_flag:
            mem = get_mem()
            if not mem or mem != mem:  # nan or invalid
                break
            peak = max(peak, mem)
            time.sleep(interval)

    # actual tracking
    # before
    rss_before = get_mem()
    peak = rss_before
    stop_flag = False
    print(f"DEBUG {label}: rss_before = {rss_before:.3f} {units}")
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        yield  # code block runs here
    finally:
        stop_flag = True
        thread.join(timeout=interval * 1.5)
        rss_after = get_mem()
        print(f"[{label}] End  memory: {rss_after:.2f} {units}")
        print(f"[{label}] Peak memory: {peak:.2f} {units}")
        print(f"[{label}] Diff memory: {rss_after - rss_before:+.2f} {units}")


def get_client():
    try:
        return default_client()
    except ValueError:
        return None


def get_mem_limit_MB(
    worker_fraction: float = 0.2, system_fraction: float = 0.1
) -> float:
    """Return a safe target memory in MB, based on Dask worker or system memory.

    Uses a fraction of available memory to avoid oversubscription.
    - If inside a Dask worker: uses `worker_fraction` of its memory limit.
    - Otherwise: uses `system_fraction` of total system memory.
    """
    try:
        worker = get_worker()
        limit_MB = worker.memory_limit / 1024**2
        print("worker MB", limit_MB, worker_fraction)
        return limit_MB * worker_fraction
    except (ValueError, RuntimeError):
        # Not inside a Dask worker
        limit_MB = psutil.virtual_memory().total / 1024**2
        print("system MB", limit_MB, system_fraction)
        return limit_MB * system_fraction

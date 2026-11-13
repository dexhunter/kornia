"""Peak traced bytes of one ASCII PLY write of a 640x480 depth cloud.

`save_pointcloud_ply` materialises every coordinate with `Tensor.tolist()`
before formatting lines. tracemalloc's peak is that live Python list (and the
line being written). A chunked write of the same shortest-round-trip text
drops the peak by the size of the list.
"""

from __future__ import annotations

import gc
import tracemalloc

import kornia

from ply_workload import depth_cloud, temporary_ply


def peak_bytes(path: str, points) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        tracemalloc.clear_traces()
        tracemalloc.reset_peak()
        kornia.geometry.save_pointcloud_ply(path, points)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def main() -> None:
    points = depth_cloud()
    directory, path = temporary_ply("-ascii")
    try:
        peak_bytes(path, points)  # warmup: imports and the file cache
        print(peak_bytes(path, points))
    finally:
        directory.cleanup()


if __name__ == "__main__":
    main()

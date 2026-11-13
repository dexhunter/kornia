"""Peak new Python blocks while writing one binary PLY of a 640x480 depth cloud.

`save_pointcloud_ply_binary` passes `Tensor.tolist()` into `array.array`,
which builds one Python float per coordinate (307200 * 3). The metric is the
high-water mark of `sys.getallocatedblocks()` during that call, minus the
count just before it. Copying the float64 buffer with `frombytes` removes
those floats and lowers this count.
"""

from __future__ import annotations

import gc
import sys
import threading

import kornia

from ply_workload import depth_cloud, temporary_ply


def peak_new_blocks(fn) -> int:
    gc.collect()
    stop = threading.Event()
    base = sys.getallocatedblocks()
    peak = base

    def sample() -> None:
        nonlocal peak
        while not stop.is_set():
            current = sys.getallocatedblocks()
            if current > peak:
                peak = current

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    try:
        fn()
    finally:
        stop.set()
        watcher.join()
    return peak - base


def main() -> None:
    points = depth_cloud()
    directory, path = temporary_ply("-binary")
    try:
        peak_new_blocks(lambda: kornia.geometry.save_pointcloud_ply_binary(path, points))
        print(peak_new_blocks(lambda: kornia.geometry.save_pointcloud_ply_binary(path, points)))
    finally:
        directory.cleanup()


if __name__ == "__main__":
    main()

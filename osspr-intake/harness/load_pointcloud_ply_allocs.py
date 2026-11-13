"""Peak new Python blocks while reading one ASCII PLY of a 640x480 depth cloud.

`load_pointcloud_ply` splits every vertex line and appends three Python floats
to one list, then copies that list into a tensor. The metric is the high-water
mark of `sys.getallocatedblocks()` during the read, minus the count just
before it. A bulk parse into a preallocated buffer returns the same float32
`(N, 3)` tensor with far fewer live objects.
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
    directory, path = temporary_ply("-ascii")
    try:
        kornia.geometry.save_pointcloud_ply(path, points)
        peak_new_blocks(lambda: kornia.geometry.load_pointcloud_ply(path))
        print(peak_new_blocks(lambda: kornia.geometry.load_pointcloud_ply(path)))
    finally:
        directory.cleanup()


if __name__ == "__main__":
    main()

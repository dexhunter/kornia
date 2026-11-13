"""Shared VGA depth-cloud workload for the PLY intake harnesses.

A 640x480 map is one frame from a typical depth camera. Coordinates are
deterministic so repeated runs allocate the same Python objects.
"""

from __future__ import annotations

import os
import tempfile

import torch

# 640 * 480 = 307_200 points, channels-last (H, W, 3), the layout the writers require.
HEIGHT = 480
WIDTH = 640


def depth_cloud() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    points = torch.rand(HEIGHT, WIDTH, 3, generator=generator, dtype=torch.float32)
    # Depth in metres, not a uniform cube, so the text writer emits realistic magnitudes.
    points[..., 2] = points[..., 2] * 5.0 + 0.3
    return points


def temporary_ply(suffix: str) -> tuple[tempfile.TemporaryDirectory[str], str]:
    directory = tempfile.TemporaryDirectory(prefix="kornia-ply-intake-")
    return directory, os.path.join(directory.name, f"cloud{suffix}.ply")

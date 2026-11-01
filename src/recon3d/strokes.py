"""User strokes -> per-view control mask `umask`.

Ports the numeric part of regiuster_uinput.m / interact_to.m / erase_from.m, without the
MATLAB figure window. Convention (resolved from the code): positive = inside, negative =
outside.
  - erase_from ('inside'):   umask += 3 * change
  - interact_to ('outside'): umask -= change
where `change` is 2 on the stroke (thickened by one pixel in 8 directions, clamped to the
image, then closed with disk(4)) and 0 elsewhere.
"""
from __future__ import annotations

import numpy as np

from .silhouette import disk, imclose

INSIDE_GAIN = 3.0
OUTSIDE_GAIN = -1.0
STROKE_VALUE = 2.0


def stroke_change(points, rows, cols):
    """`points` is a sequence of (x, y) pixel positions (column, row), 0-based floats."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros((rows, cols))
    # densify so a fast drag has no gaps (a mouse reports sparse points)
    dense = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(np.ceil(np.hypot(*(b - a)))), 1)
        t = np.arange(1, n + 1)[:, None] / n
        dense.extend(a + (b - a) * t)
    dense = np.round(np.asarray(dense))
    m = np.zeros((rows, cols), dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            x = np.clip(dense[:, 0] + dx, 0, cols - 1).astype(int)
            y = np.clip(dense[:, 1] + dy, 0, rows - 1).astype(int)
            m[y, x] = True
    return imclose(m, disk(4)) * STROKE_VALUE


def build_umask(strokes, rows, cols):
    """`strokes`: iterable of dicts {type: 'in'|'out', pts: [[x, y], ...]}."""
    u = np.zeros((rows, cols))
    for s in strokes:
        change = stroke_change(s["pts"], rows, cols)
        u += change * (INSIDE_GAIN if s["type"] == "in" else OUTSIDE_GAIN)
    return u

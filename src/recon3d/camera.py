"""Cameras and projection.

Ports select_camera_auto.m, create_projection_operator.m and image_processor.m.
Pixel arrays are 0-based here; MATLAB's 1-based clamping to [1, size] becomes [0, size-1].
"""
from __future__ import annotations

import numpy as np


def select_camera_auto(index: int, dist: float = 4.0, z_offset: float = 0.0):
    """Camera `index` (1-based, as in MATLAB) on a circle, looking at the origin.

    Returns (cam_x, cam_y, cam_z, camera_loc) as length-3 arrays. The loaders scale
    camera_loc by 10 afterwards (see `Solver`).
    """
    res_multiplier = 4.0
    ang = (index - 1) * 10.0 + 3.0
    x = dist * np.cos(np.deg2rad(ang)) * res_multiplier
    y = dist * np.sin(np.deg2rad(ang)) * res_multiplier
    z = z_offset
    loc = np.array([x, y, z], dtype=float)
    cam_z = -loc / np.sqrt(x * x + y * y + z * z)
    cam_y = np.array([0.0, 0.0, 1.0])
    cx = np.cross(cam_y, cam_z)
    cam_x = cx / np.sqrt(np.sum(cx * cx))
    return cam_x, cam_y, cam_z, loc


def create_projection_operator(cam_x, cam_y, cam_z, camera_loc, img_sz, f):
    """3x4 operator K @ [R | -R c]. `img_sz` is (rows, cols)."""
    R = np.zeros((3, 3))
    for r, v in enumerate((cam_x, cam_y, cam_z)):
        R[r] = np.asarray(v, float) / np.sqrt(np.sum(np.asarray(v, float) ** 2))
    T = np.eye(4)
    T[:3, 3] = -np.asarray(camera_loc, float)
    R4 = np.eye(4)
    R4[:3, :3] = R
    persp = (R4 @ T)[:3, :]
    K = np.array([[f, 0.0, img_sz[1] / 2.0], [0.0, f, img_sz[0] / 2.0], [0.0, 0.0, 1.0]])
    return K @ persp


def matlab_round(a):
    """MATLAB rounds half away from zero (numpy rounds half to even)."""
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


def project_points(coords, P, sz):
    """image_processor.m. `coords` is (n, 4) homogeneous; `sz` is (rows, cols).

    Returns (px, py, z): 0-based integer pixel columns/rows (clamped into the image)
    and the depth z of every point.
    """
    tv = coords @ P.T
    z = tv[:, 2]
    xy = tv[:, :2] / (z + 1e-5)[:, None]
    xcap = matlab_round(xy)
    rows, cols = sz[0], sz[1]
    x = np.clip(xcap[:, 0], 1, cols)  # MATLAB: >=col_max -> col_max, <=1 -> 1
    y = np.clip(xcap[:, 1], 1, rows)
    return (x.astype(np.int64) - 1), (y.astype(np.int64) - 1), z

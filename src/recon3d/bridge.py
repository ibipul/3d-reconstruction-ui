"""Glue between the solver and a UI that talks bytes and JSON (the Pyodide web worker).

Everything here takes and returns plain bytes/str so the JS side needs no NumPy knowledge.
"""
from __future__ import annotations

import json

import numpy as np

from .data import gt_mask_from_gray
from .solver import Config, Solver, imnoise_salt_pepper


def init_solver(gray_bytes, n, rows, cols, grid, noise, seed):
    """`gray_bytes`: n * rows * cols uint8 ground-truth silhouettes (row-major).

    As in loader_wt_control_rec_error_trace.m, the solver's observed image is the ground
    truth plus salt & pepper noise of density `noise`; the closed binary mask is for metrics.
    """
    g = np.frombuffer(bytes(gray_bytes), dtype=np.uint8).reshape(n, rows, cols)
    rng = np.random.default_rng(int(seed))
    images, masks = [], []
    for i in range(n):
        masks.append(gt_mask_from_gray(g[i]))
        images.append(imnoise_salt_pepper(g[i], float(noise), rng) if noise > 0 else g[i].copy())
    solver = Solver(images, masks, Config(grid=int(grid)))
    solver._bridge_masks = masks
    return solver


def inputs_bytes(solver):
    """(observed images, ground-truth masks) as uint8 bytes, shape (n, rows, cols)."""
    imgs = np.stack(solver.images).astype(np.uint8)
    gts = np.stack(solver._bridge_masks).astype(np.uint8)
    return imgs.tobytes(), gts.tobytes()


def set_strokes(solver, view, strokes_json):
    solver.set_strokes(int(view), json.loads(strokes_json))


def step(solver, k):
    """Run k iterations; returns JSON list of per-iteration metrics."""
    out = []
    for _ in range(int(k)):
        r = solver.step()
        r["voxels"] = int((solver.phi >= 0).sum())
        out.append(r)
    return json.dumps(out)


def snapshot(solver, max_points=2500):
    """(outlines, phi points, phi_hat points) as bytes.

    outlines is int32: for each view [n_phi, idx..., n_hat, idx...] where idx are
    column-major flat pixel indices (y + rows * x).
    """
    parts = []
    for v in range(solver.n_views):
        a, b = solver.outline(v), solver.outline(v, hat=True)
        parts += [np.array([a.size], np.int32), a, np.array([b.size], np.int32), b]
    outlines = np.concatenate(parts).astype(np.int32)
    return (outlines.tobytes(),
            solver.surface_points(False, max_points).tobytes(),
            solver.surface_points(True, max_points // 2, seed=1).tobytes())

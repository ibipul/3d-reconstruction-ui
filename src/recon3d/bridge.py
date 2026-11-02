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


def set_param(solver, name, value):
    """Change a solver setting while it runs (applies from the next iteration)."""
    if name not in ("curvature_weight", "sussman_dt", "sussman_every", "narrow_band_high"):
        raise ValueError("unknown parameter: " + name)
    setattr(solver.cfg, name, int(value) if name == "sussman_every" else float(value))


def _mesh_bytes(mesh):
    verts, normals, tris = mesh
    head = np.array([len(verts), len(tris)], np.int32)
    return head.tobytes() + verts.tobytes() + normals.tobytes() + tris.tobytes()


def snapshot(solver):
    """(outlines, mesh, mesh_hat) as bytes.

    outlines is float32: for each view [n_phi, x1 y1 x2 y2 ..., n_hat, ...], smooth sub-pixel
    segments in pixel-centre coordinates. A mesh is int32 [n_vertices, n_triangles] followed by
    float32 vertices (n,3), float32 normals (n,3) and uint32 triangles (m,3).
    """
    parts = []
    for v in range(solver.n_views):
        for hat in (False, True):
            seg = solver.outline_segments(v, hat)
            parts += [np.array([len(seg)], np.float32), seg.ravel()]
    outlines = np.concatenate(parts).astype(np.float32)
    return (outlines.tobytes(), _mesh_bytes(solver.surface_mesh(False)), _mesh_bytes(solver.surface_mesh(True)))

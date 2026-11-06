"""Glue between the solver and a UI that talks bytes and JSON (the Pyodide web worker).

Everything here takes and returns plain bytes/str so the JS side needs no NumPy knowledge.
"""
from __future__ import annotations

import json

import numpy as np

from . import meshio
from .data import gt_mask_from_gray
from .solver import Config, ShapeCollapsed, Solver, imnoise_salt_pepper


def init_solver(gray_bytes, n, rows, cols, grid, noise, seed, photo_bytes=None, focal=None, distance=None, box=None, box_angle=0.0, camera_height=0.0,
                energy="bhattacharyya"):
    """`gray_bytes`: n * rows * cols uint8 ground-truth silhouettes (row-major), used for the metrics
    and, by default, as the solver's input. None for a dataset without ground truth: `photo_bytes` is then
    required and the metrics that compare against the ground truth are None.

    `focal` (pixels) and `distance` (camera distance from the object in world units, default 160) are free
    parameters of the camera ring. The distance sets how large the object looks compared with the initial surface.
    `box` is the initial surface, an (x, y, z) size in voxels at grid 150 (scaled with the grid); default: the MATLAB box.
    `energy` is the region energy: "bhattacharyya" (histograms inside vs outside) or "chan_vese" (means, as in the MATLAB code).
    `box_angle` turns that box about the vertical axis (degrees). `camera_height` raises the cameras above the ring
    plane (world units, positive = above the object).

    As in loader_wt_control_rec_error_trace.m, the observed image is the ground truth plus salt &
    pepper noise of density `noise`. If `photo_bytes` (n * rows * cols * 3 uint8 RGB) is given, the
    solver runs on the photos instead, on all three channels (the `length(size(img))==3` branch of
    the MATLAB loaders); noise is then applied to each channel independently.
    """
    rng = np.random.default_rng(int(seed))
    if gray_bytes is None:
        if photo_bytes is None:
            raise ValueError("a dataset without ground truth needs photos")
        g = masks = None
    else:
        g = np.frombuffer(bytes(gray_bytes), dtype=np.uint8).reshape(n, rows, cols)
        masks = [gt_mask_from_gray(g[i]) for i in range(n)]
    if photo_bytes is not None:
        src = np.frombuffer(bytes(photo_bytes), dtype=np.uint8).reshape(n, rows, cols, 3)
    else:
        src = g
    images = [imnoise_salt_pepper(src[i], float(noise), rng) if noise > 0 else src[i].copy() for i in range(n)]
    if energy not in ("bhattacharyya", "chan_vese"):
        raise ValueError("unknown energy: %s" % energy)
    cfg = Config(grid=int(grid), energy=energy)
    if focal is not None:
        cfg.focal = float(focal)
    if box is not None:
        cfg.init_box = tuple(float(b) for b in box)
        cfg.init_box_angle = float(box_angle)
    cfg.camera_height = float(camera_height)
    if distance is not None:
        cfg.camera_dist = float(distance) / (4.0 * cfg.loc_scale)     # camera_dist 4 <-> 160 world units
    solver = Solver(images, masks, cfg)
    solver._bridge_masks = masks
    return solver


def inputs_bytes(solver):
    """(observed images, ground-truth masks or b"" without ground truth, channels). Images are uint8, shape (n, rows, cols) or
    (n, rows, cols, 3); masks are 0/1, shape (n, rows, cols)."""
    imgs = np.stack(solver.images).astype(np.uint8)
    masks = solver._bridge_masks
    gts = b"" if masks is None else np.stack(masks).astype(np.uint8).tobytes()      # empty: no ground truth
    return imgs.tobytes(), gts, (3 if imgs.ndim == 4 else 1)


def set_strokes(solver, view, strokes_json):
    solver.set_strokes(int(view), json.loads(strokes_json))


def step(solver, k):
    """Run k iterations. Returns JSON {"metrics": [per-iteration dicts], "collapsed": null or a message}.

    If a level set loses its whole interior the run stops and `collapsed` says so, instead of failing."""
    out, collapsed = [], None
    for _ in range(int(k)):
        try:
            r = solver.step()
        except ShapeCollapsed as e:
            collapsed = str(e)
            break
        r["voxels"] = int((solver.phi >= 0).sum())
        out.append(r)
    return json.dumps({"metrics": out, "collapsed": collapsed})


def set_param(solver, name, value):
    """Change a solver setting while it runs (applies from the next iteration)."""
    if name not in ("curvature_weight", "sussman_dt", "sussman_every", "narrow_band_high"):
        raise ValueError("unknown parameter: " + name)
    setattr(solver.cfg, name, int(value) if name == "sussman_every" else float(value))


def _mesh_bytes(mesh):
    verts, normals, tris = mesh
    head = np.array([len(verts), len(tris)], np.int32)
    return head.tobytes() + verts.tobytes() + normals.tobytes() + tris.tobytes()


def export_model(solver, fmt, hat=False):
    """The current shape as an OBJ, STL or PLY file (bytes), in world units (see Solver.export_mesh)."""
    verts, normals, tris = meshio.drop_degenerate(*solver.export_mesh(bool(hat)))
    if len(tris) == 0:
        raise ValueError("there is nothing to export: the shape has no surface")
    comment = ("recon3d reconstruction (%s) after %d iterations, grid %d\n"
               "units: world units of the camera setup (camera distance 160), centred on the object, z up\n"
               "%d vertices, %d triangles" % ("phi_hat" if hat else "phi", solver.iter, solver.cfg.grid, len(verts), len(tris)))
    return meshio.write(fmt, verts, normals, tris, comment)


def _outline_parts(solver):
    parts = []
    for v in range(solver.n_views):
        for hat in (False, True):
            seg = solver.outline_segments(v, hat)
            parts += [np.array([len(seg)], np.float32), seg.ravel()]
    return np.concatenate(parts).astype(np.float32).tobytes()


def preview(n, rows, cols, grid, focal, distance, box, box_angle, camera_height):
    """The starting surface as outlines in every view plus its mesh, without photos or a run: a cheap look at the
    camera and initial-box settings while they are being tuned. Done at a coarse `grid`; the box has the same size
    in world units at any grid, so it matches the real run up to voxel rounding. Returns (outlines, mesh) bytes."""
    blank = np.zeros(n * rows * cols * 3, np.uint8).tobytes()
    s = init_solver(None, n, rows, cols, grid, 0.0, 0, blank, focal, distance, box, box_angle, camera_height)
    return _outline_parts(s), _mesh_bytes(s.surface_mesh(False))


def snapshot(solver):
    """(outlines, mesh, mesh_hat) as bytes.

    outlines is float32: for each view [n_phi, x1 y1 x2 y2 ..., n_hat, ...], smooth sub-pixel
    segments in pixel-centre coordinates. A mesh is int32 [n_vertices, n_triangles] followed by
    float32 vertices (n,3), float32 normals (n,3) and uint32 triangles (m,3).
    """
    return (_outline_parts(solver), _mesh_bytes(solver.surface_mesh(False)), _mesh_bytes(solver.surface_mesh(True)))

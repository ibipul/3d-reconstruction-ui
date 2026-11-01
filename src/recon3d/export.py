"""Turn a saved strokes file (from the web app) into per-view control masks.

The masks are exactly what the solver uses (`strokes.build_umask`): +6 on inside strokes, -2 on
outside strokes, 0 elsewhere. The MATLAB output mirrors `get_uis.m`, so a loader's control
injection (`load(...); camera_view(k).umask = uinput_bank(k).umask; ...`) can use it directly.
"""
from __future__ import annotations

import json

import numpy as np

from .strokes import build_umask

FORMAT = "recon3d-strokes"


def load_session(path_or_obj):
    obj = path_or_obj
    if isinstance(path_or_obj, str):
        with open(path_or_obj) as f:
            obj = json.load(f)
    if obj.get("format") != FORMAT:
        raise ValueError("not a recon3d strokes file")
    return obj


def masks_from_session(session, n_views=36):
    """Returns (umask array (n_views, rows, cols) float32, per-view stroke lists)."""
    rows, cols = session["image"]["rows"], session["image"]["cols"]
    per_view = [[] for _ in range(n_views)]
    for st in session["strokes"]:
        per_view[int(st["view"])].append(st)
    umask = np.zeros((n_views, rows, cols), np.float32)
    for v, lst in enumerate(per_view):
        if lst:
            umask[v] = build_umask(lst, rows, cols)
    return umask, per_view


def uinput_bank(umask, per_view):
    """A (1, n) MATLAB struct array with the fields of get_uis.m."""
    n = umask.shape[0]
    dt = [("umask", "O"), ("smask", "O"), ("umask_indices", "O"), ("utype", "O"), ("npts", "O")]
    bank = np.empty((1, n), dtype=dt)
    for v in range(n):
        lst = per_view[v]
        m = umask[v].astype(np.float64)
        bank[0, v] = (m, m, np.array([[s.get("iter", 0) for s in lst]], dtype=np.float64),
                      "".join("i" if s["type"] == "in" else "o" for s in lst),
                      np.ones((1, len(lst))))
    return bank


def save_masks(session, out_path, n_views=36):
    """Write .npz (NumPy: umask, plus iter/type per stroke) or .mat (MATLAB: uinput_bank)."""
    umask, per_view = masks_from_session(session, n_views)
    if out_path.endswith(".mat"):
        from scipy.io import savemat
        savemat(out_path, {"uinput_bank": uinput_bank(umask, per_view)})
    else:
        np.savez_compressed(out_path, umask=umask, strokes_per_view=np.array([len(p) for p in per_view]))
    return umask

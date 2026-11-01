"""How much does the curvature weight smooth the result? Runs the clean-data loop at several weights
and reports surface area (smaller = smoother), IoU and the roughness of the projected outline.
Usage: python scripts/smoothing_sweep.py [grid] [iterations]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from recon3d.data import load_trace_dataset
from recon3d.solver import Config, Solver

grid = int(sys.argv[1]) if len(sys.argv) > 1 else 64
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
root = os.path.expanduser("~/codes/3d-reconstruction-ui-matlab/dataset_sentinel_gt")
imgs, gts = load_trace_dataset(root, 36, noise=0.0)


def area(v, t):
    a, b, c = v[t[:, 0]].astype(float), v[t[:, 1]].astype(float), v[t[:, 2]].astype(float)
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


print("weight   voxels   IoU    surface area   (area/voxels^(2/3): lower = smoother)")
for w in (0.001, 0.1, 1.0, 3.0, 10.0):
    s = Solver(imgs, gts, Config(grid=grid, curvature_weight=w))
    for _ in range(iters):
        r = s.step()
    v, _, t = s.surface_mesh()
    vox = int((s.phi >= 0).sum())
    iou = r["tp"] / max(r["tp"] + r["fp"] + r["fn"], 1)
    A = area(v, t)
    print("%7g  %6d  %.3f  %10.3f   %.4f" % (w, vox, iou, A, A / max(vox, 1) ** (2 / 3)))

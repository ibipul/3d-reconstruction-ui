import json

import numpy as np

from recon3d import bridge


def _gt_bytes(n=36, rows=80, cols=50):
    g = np.zeros((n, rows, cols), np.uint8)
    g[:, 20:60, 15:35] = 255
    return g.tobytes(), n, rows, cols


def test_init_step_snapshot_roundtrip():
    gt, n, rows, cols = _gt_bytes()
    s = bridge.init_solver(gt, n, rows, cols, grid=40, noise=0.0, seed=0)
    inp, gts = bridge.inputs_bytes(s)
    assert len(inp) == n * rows * cols and len(gts) == n * rows * cols
    assert set(np.unique(np.frombuffer(gts, np.uint8))) <= {0, 1}

    metrics = json.loads(bridge.step(s, 2))
    assert [m["iter"] for m in metrics] == [1, 2]
    assert {"energy", "tp", "fp", "fn", "rms", "voxels"} <= set(metrics[0])

    outlines, pts, pts_hat = bridge.snapshot(s)
    o = np.frombuffer(outlines, np.int32)
    p = 0
    for _ in range(n):  # n_phi, idx..., n_hat, idx...
        n1 = o[p]; p += 1 + n1
        n2 = o[p]; p += 1 + n2
    assert p == o.size
    assert np.frombuffer(pts, np.float32).size % 3 == 0 and len(pts) > 0


def test_noise_is_seeded_and_applied():
    gt, n, rows, cols = _gt_bytes()
    a = bridge.init_solver(gt, n, rows, cols, 40, 0.9, 7)
    b = bridge.init_solver(gt, n, rows, cols, 40, 0.9, 7)
    c = bridge.init_solver(gt, n, rows, cols, 40, 0.9, 8)
    assert (a.images[0] == b.images[0]).all() and not (a.images[0] == c.images[0]).all()
    assert (a.images[0] != np.frombuffer(gt, np.uint8).reshape(n, rows, cols)[0]).mean() > 0.4


def test_set_strokes_json():
    gt, n, rows, cols = _gt_bytes()
    s = bridge.init_solver(gt, n, rows, cols, 40, 0.0, 0)
    bridge.set_strokes(s, 3, json.dumps([{"type": "in", "size": 6, "pts": [[10, 30], [40, 30]]}]))
    assert s.views[3]["umask"].max() == 6.0
    bridge.set_strokes(s, 3, "[]")
    assert s.views[3]["umask"].max() == 0.0

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
    inp, gts, channels = bridge.inputs_bytes(s)
    assert channels == 1 and len(inp) == n * rows * cols and len(gts) == n * rows * cols
    assert set(np.unique(np.frombuffer(gts, np.uint8))) <= {0, 1}

    metrics = json.loads(bridge.step(s, 2))
    assert [m["iter"] for m in metrics] == [1, 2]
    assert {"energy", "tp", "fp", "fn", "rms", "voxels"} <= set(metrics[0])

    outlines, mesh, mesh_hat = bridge.snapshot(s)
    o = np.frombuffer(outlines, np.float32)
    p = 0
    for _ in range(n * 2):  # per view: phi then phi_hat: n_seg, then 4 floats per segment
        k = int(o[p]); p += 1 + 4 * k
    assert p == o.size
    for blob in (mesh, mesh_hat):
        nv, nt = np.frombuffer(blob[:8], np.int32)
        assert nv > 0 and nt > 0 and len(blob) == 8 + nv * 24 + nt * 12


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


def test_set_param_changes_smoothing_live():
    gt, n, rows, cols = _gt_bytes()
    s = bridge.init_solver(gt, n, rows, cols, 40, 0.0, 0)
    assert s.cfg.curvature_weight == 0.001
    bridge.set_param(s, "curvature_weight", 0.25)
    assert s.cfg.curvature_weight == 0.25
    bridge.set_param(s, "sussman_every", 10.0)
    bridge.set_param(s, "sussman_dt", 0.05)
    assert s.cfg.sussman_every == 10 and isinstance(s.cfg.sussman_every, int) and s.cfg.sussman_dt == 0.05
    import pytest
    with pytest.raises(ValueError):
        bridge.set_param(s, "grid", 10)


def test_photos_as_solver_input():
    gt, n, rows, cols = _gt_bytes()
    photos = np.random.default_rng(1).integers(40, 220, (n, rows, cols, 3), dtype=np.uint8)
    s = bridge.init_solver(gt, n, rows, cols, 40, 0.0, 0, photos.tobytes())
    inp, gts, ch = bridge.inputs_bytes(s)
    assert ch == 3 and len(inp) == n * rows * cols * 3 and len(gts) == n * rows * cols
    assert (np.frombuffer(inp, np.uint8).reshape(n, rows, cols, 3) == photos).all()
    metrics = json.loads(bridge.step(s, 2))
    assert len(metrics) == 2 and all(np.isfinite(m["energy"]) for m in metrics)
    # the default input is still the silhouettes
    s2 = bridge.init_solver(gt, n, rows, cols, 40, 0.0, 0)
    assert bridge.inputs_bytes(s2)[2] == 1
    # noise hits each channel independently
    noisy = bridge.init_solver(gt, n, rows, cols, 40, 0.5, 3, photos.tobytes())
    changed = [(noisy.images[0][:, :, c] != photos[0][:, :, c]) for c in range(3)]
    assert all(0.4 < c.mean() < 0.6 for c in changed)
    assert not (changed[0] == changed[1]).all()

import json

import numpy as np
import pytest

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

    res = json.loads(bridge.step(s, 2))
    metrics = res["metrics"]
    assert res["collapsed"] is None and [m["iter"] for m in metrics] == [1, 2]
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
    metrics = json.loads(bridge.step(s, 2))["metrics"]
    assert len(metrics) == 2 and all(np.isfinite(m["energy"]) for m in metrics)
    # the default input is still the silhouettes
    s2 = bridge.init_solver(gt, n, rows, cols, 40, 0.0, 0)
    assert bridge.inputs_bytes(s2)[2] == 1
    # noise hits each channel independently
    noisy = bridge.init_solver(gt, n, rows, cols, 40, 0.5, 3, photos.tobytes())
    changed = [(noisy.images[0][:, :, c] != photos[0][:, :, c]) for c in range(3)]
    assert all(0.4 < c.mean() < 0.6 for c in changed)
    assert not (changed[0] == changed[1]).all()


def test_a_vanished_shape_is_reported_not_raised():
    gt, n, rows, cols = _gt_bytes()
    s = bridge.init_solver(gt, n, rows, cols, 40, 0.0, 0)
    json.loads(bridge.step(s, 1))
    s.phi[...] = -1.0                                  # nothing is inside any more
    res = json.loads(bridge.step(s, 3))
    assert res["metrics"] == [] and "no interior left" in res["collapsed"]
    s2 = bridge.init_solver(gt, n, rows, cols, 40, 0.0, 0)
    json.loads(bridge.step(s2, 1))
    s2.phi_hat[...] = -1.0                             # the estimate vanishing is reported too
    assert "phi_hat" in json.loads(bridge.step(s2, 2))["collapsed"]


def test_dataset_without_ground_truth():
    """No ground-truth silhouettes (e.g. a real photo set): needs photos, runs, reports no GT metrics."""
    n, rows, cols = 36, 54, 80                                        # landscape, like the zebra photos
    photos = np.random.default_rng(2).integers(40, 220, (n, rows, cols, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        bridge.init_solver(None, n, rows, cols, 40, 0.0, 0)
    s = bridge.init_solver(None, n, rows, cols, 40, 0.0, 0, photos.tobytes(), focal=130.0)
    inp, gts, ch = bridge.inputs_bytes(s)
    assert ch == 3 and len(inp) == n * rows * cols * 3 and gts == b""
    res = json.loads(bridge.step(s, 2))
    assert res["collapsed"] is None and len(res["metrics"]) == 2
    for r in res["metrics"]:
        assert all(r[k] is None for k in ("rms", "tp", "tn", "fp", "fn"))
        assert np.isfinite(r["energy"]) and np.isfinite(r["lyapunov"])
    assert s.cfg.focal == 130.0


def test_distance_moves_the_cameras():
    n, rows, cols = 36, 54, 80
    photos = np.random.default_rng(3).integers(40, 220, (n, rows, cols, 3), dtype=np.uint8)
    a = bridge.init_solver(None, n, rows, cols, 40, 0.0, 0, photos.tobytes())
    b = bridge.init_solver(None, n, rows, cols, 40, 0.0, 0, photos.tobytes(), distance=240.0)
    assert np.isclose(np.linalg.norm(a.views[0]["loc"]), 160.0) and np.isclose(np.linalg.norm(b.views[0]["loc"]), 240.0)


def _box_extent(s):
    inside = np.argwhere(s.phi >= 0)
    return tuple(int(v) for v in (inside.max(0) - inside.min(0) + 1))


def test_default_initial_box_matches_the_matlab_box_within_rounding():
    n, rows, cols = 36, 54, 80
    photos = np.random.default_rng(4).integers(40, 220, (n, rows, cols, 3), dtype=np.uint8).tobytes()
    for grid in (64, 100, 150):
        a = bridge.init_solver(None, n, rows, cols, grid, 0.0, 0, photos)
        b = bridge.init_solver(None, n, rows, cols, grid, 0.0, 0, photos, box=(20, 60, 80))
        assert max(abs(p - q) for p, q in zip(_box_extent(a), _box_extent(b))) <= 2, (grid, _box_extent(a), _box_extent(b))


def test_initial_box_size_is_adjustable():
    n, rows, cols = 36, 54, 80
    photos = np.random.default_rng(5).integers(40, 220, (n, rows, cols, 3), dtype=np.uint8).tobytes()
    small = bridge.init_solver(None, n, rows, cols, 64, 0.0, 0, photos, box=(10, 10, 20))
    wide = bridge.init_solver(None, n, rows, cols, 64, 0.0, 0, photos, box=(50, 30, 20))
    assert _box_extent(wide)[0] > _box_extent(small)[0] and _box_extent(wide)[1] > _box_extent(small)[1]
    assert _box_extent(wide)[2] == _box_extent(small)[2]                 # same height
    assert json.loads(bridge.step(wide, 2))["collapsed"] is None


def test_rotated_box_is_the_same_box_at_angle_zero():
    from recon3d import levelset as ls
    for args in ((64, 34, 13, 4), (100, 54, 20, 7), (150, 80, 30, 10)):
        a = ls.initial_box_phi(*args)
        b = ls.rotated_box_phi(*args, 0.0)
        assert a.shape == b.shape and ((a >= 0) == (b >= 0)).all()
        assert np.allclose(a[a >= 0], b[a >= 0])                      # inside: distance to the faces, to the voxel


def test_box_angle_turns_the_box():
    n, rows, cols = 36, 54, 80
    photos = np.random.default_rng(6).integers(40, 220, (n, rows, cols, 3), dtype=np.uint8).tobytes()
    a = bridge.init_solver(None, n, rows, cols, 64, 0.0, 0, photos, box=(20, 60, 30))
    b = bridge.init_solver(None, n, rows, cols, 64, 0.0, 0, photos, box=(20, 60, 30), box_angle=90.0)
    ea, eb = _box_extent(a), _box_extent(b)
    assert abs(ea[0] - eb[1]) <= 1 and abs(ea[1] - eb[0]) <= 1 and ea[2] == eb[2]
    assert abs(int((a.phi >= 0).sum()) - int((b.phi >= 0).sum())) <= 0.12 * int((a.phi >= 0).sum())   # same volume


def test_camera_height_lifts_the_cameras():
    n, rows, cols = 36, 54, 80
    photos = np.random.default_rng(7).integers(40, 220, (n, rows, cols, 3), dtype=np.uint8).tobytes()
    flat = bridge.init_solver(None, n, rows, cols, 40, 0.0, 0, photos)
    up = bridge.init_solver(None, n, rows, cols, 40, 0.0, 0, photos, camera_height=60.0)
    assert flat.views[0]["loc"][2] == 0.0 and np.allclose(up.views[0]["loc"][:2], flat.views[0]["loc"][:2])
    assert np.isclose(up.views[0]["loc"][2], -60.0)                  # world z points down, so above is negative
    assert json.loads(bridge.step(up, 2))["collapsed"] is None


def test_preview_shows_the_starting_surface_without_a_run():
    n, rows, cols = 36, 54, 80
    outlines, mesh = bridge.preview(n, rows, cols, 40, 130.0, 160.0, (20, 60, 80), 30.0, 10.0)
    o = np.frombuffer(outlines, np.float32)
    assert o.size > 2 * n and np.isfinite(o).all() and o[0] > 0          # a first view with a phi outline
    assert np.frombuffer(mesh[:8], np.int32)[1] > 0                      # and some triangles


def _world_centre(s):
    inside = np.argwhere(s.phi >= 0).astype(float)
    return ((inside - s.m).mean(0)) * s.voxel_scale


def test_box_offset_moves_the_box_in_world_coordinates():
    n, rows, cols = 36, 54, 80
    photos = np.random.default_rng(8).integers(40, 220, (n, rows, cols, 3), dtype=np.uint8).tobytes()
    for grid in (48, 64):
        a = bridge.init_solver(None, n, rows, cols, grid, 0.0, 0, photos, box=(20, 60, 80))
        b = bridge.init_solver(None, n, rows, cols, grid, 0.0, 0, photos, box=(20, 60, 80), box_offset=(15.0, -20.0, 30.0))
        d = _world_centre(b) - _world_centre(a)
        assert np.allclose(d, (15.0, -20.0, 30.0), atol=2.5), (grid, d)
        assert json.loads(bridge.step(b, 2))["collapsed"] is None


def test_dragging_right_in_a_view_moves_the_box_right_in_that_view():
    """The page turns a drag in view i into a world shift along the camera's right/down axes; check against the
    solver's own projection that those axes really are image-right and image-down."""
    n, rows, cols = 36, 54, 80
    photos = np.random.default_rng(9).integers(40, 220, (n, rows, cols, 3), dtype=np.uint8).tobytes()
    for height in (0.0, 40.0):
        s = bridge.init_solver(None, n, rows, cols, 40, 0.0, 0, photos, focal=130.0, camera_height=height)
        for i in (0, 9, 35):                                    # the page's axes, written the same way
            a = np.deg2rad(3 + 10 * i)
            loc = np.array([160 * np.cos(a), 160 * np.sin(a), -height]); cz = -loc / np.linalg.norm(loc)
            cx = np.cross([0, 0, 1], cz); cx /= np.linalg.norm(cx)
            cy = np.cross(cz, cx)
            P = s.views[i]["P"]
            def proj(w):
                t = P @ np.append(w, 1.0); return t[:2] / t[2]
            base = proj(np.zeros(3))
            right, down = proj(10 * cx) - base, proj(10 * cy) - base
            assert right[0] > 5 and abs(right[1]) < 1.0, (i, height, right)      # right: +x in the image, no y
            assert down[1] > 5 and abs(down[0]) < 1.0, (i, height, down)

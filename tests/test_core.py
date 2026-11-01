import numpy as np
import pytest
from scipy import ndimage

from recon3d import camera as cam
from recon3d import levelset as ls
from recon3d import silhouette as sil
from recon3d.solver import Config, Solver, imnoise_salt_pepper
from recon3d.strokes import build_umask, stroke_change


# ---------------------------------------------------------------- morphology
@pytest.mark.parametrize("radius", [3, 4, 6])
def test_disk_octagon_shape(radius):
    d = sil.disk(radius).arr
    n = 2 * radius + 1
    assert d.shape == (n, n)
    assert d[radius, radius] and d[radius, 0] and d[0, radius]
    assert not d[0, 0]
    assert (d == d[::-1]).all() and (d == d[:, ::-1]).all() and (d == d.T).all()


def test_disk3_and_disk4_match_matlab_neighbourhoods():
    d3 = np.array([[0, 0, 1, 1, 1, 0, 0],
                   [0, 1, 1, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1, 0, 0]], bool)
    assert (sil.disk(3).arr == d3).all()
    d4 = sil.disk(4).arr
    assert d4[0].sum() == 5 and d4[1].sum() == 7 and d4[2].sum() == 9 and d4[4].sum() == 9


@pytest.mark.parametrize("radius", [3, 4, 6])
def test_fast_morphology_equals_generic(radius):
    rng = np.random.default_rng(0)
    d = sil.disk(radius)
    assert d.segments is not None
    for trial in range(6):
        m = ndimage.binary_dilation(rng.random((60, 45)) < 0.01, iterations=trial % 3)
        if trial % 2:
            m[:, :6] = True  # shape touching the border exercises the border rules
        full_e = ndimage.binary_erosion(m, structure=d.arr, border_value=1)
        full_d = ndimage.binary_dilation(m, structure=d.arr, border_value=0)
        assert (sil.imerode(m, d) == full_e).all()
        assert (sil.imdilate(m, d) == full_d).all()
        full_c = ndimage.binary_erosion(full_d, structure=d.arr, border_value=1)
        assert (sil.imclose(m, d) == full_c).all()


def test_erosion_does_not_eat_shapes_at_the_border():
    m = np.zeros((20, 20), bool)
    m[:, :10] = True
    e = sil.imerode(m, sil.disk(4))
    assert e[:, 0].all()  # MATLAB pads erosion with 1s


# ---------------------------------------------------------------- camera
def test_camera_ring_and_projection_centre():
    cx, cy, cz, loc = cam.select_camera_auto(1)
    assert np.allclose(loc, [16 * np.cos(np.deg2rad(3)), 16 * np.sin(np.deg2rad(3)), 0])
    assert np.isclose(np.linalg.norm(cz), 1) and np.isclose(np.dot(cx, cz), 0)
    loc = loc * 10
    P = cam.create_projection_operator(cx, cy, cz, loc, (320, 200), 520.0)
    px, py, z = cam.project_points(np.array([[0.0, 0.0, 0.0, 1.0]]), P, (320, 200))
    assert (px[0], py[0]) == (100 - 1, 160 - 1)  # principal point, 0-based
    assert np.isclose(z[0], 160.0)


def test_matlab_round_half_away_from_zero():
    assert (cam.matlab_round(np.array([0.5, 1.5, 2.5, -0.5, -2.5])) == [1, 2, 3, -1, -3]).all()


# ---------------------------------------------------------------- level sets
def test_shifts_repeat_the_edge():
    a = np.arange(5.0)[:, None, None] * np.ones((1, 3, 3))
    assert (ls.shiftB(a)[:, 0, 0] == [1, 2, 3, 4, 4]).all()
    assert (ls.shiftF(a)[:, 0, 0] == [0, 0, 1, 2, 3]).all()
    b = np.arange(5.0)[None, :, None] * np.ones((3, 1, 3))
    assert (ls.shiftD(b)[0, :, 0] == [1, 2, 3, 4, 4]).all()
    assert (ls.shiftU(b)[0, :, 0] == [0, 0, 1, 2, 3]).all()
    c = np.arange(5.0)[None, None, :] * np.ones((3, 3, 1))
    assert (ls.shiftL(c)[0, 0, :] == [1, 2, 3, 4, 4]).all()
    assert (ls.shiftR(c)[0, 0, :] == [0, 0, 1, 2, 3]).all()


def test_initial_box_matches_matlab_dimensions():
    mask = ls.initial_box(150, 80, 30, 10)
    assert mask.shape == (150, 150, 153)  # the 3 extra slices come from MATLAB's cat() loops
    assert mask.sum() == 21 * 61 * 81
    phi = ls.signed_distance(mask)
    assert (phi[mask > 0] >= 0).all() and (phi[mask == 0] < 0).all()
    assert phi.max() <= 11  # centre of a 21-wide box


def test_signed_distance_values():
    mask = np.zeros((9, 9, 9)); mask[2:7, 2:7, 2:7] = 1
    phi = ls.signed_distance(mask)
    assert phi[4, 4, 4] == 2.0       # distance to nearest outside voxel (3) minus 1
    assert phi[2, 4, 4] == 0.0       # surface voxel
    assert phi[1, 4, 4] == -1.0


def test_sussman_restores_unit_gradient():
    n = 41
    ax = np.arange(n) - n // 2
    r = np.sqrt(ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)
    sd = 12.0 - r                                   # signed distance, positive inside
    distorted = sd * (1 + 0.8 * np.exp(-((r - 12) / 6) ** 2))
    def err(p):
        g = np.sqrt(sum(np.square(np.gradient(p, axis=a)) for a in range(3)))
        band = np.abs(sd) < 4
        return np.abs(g[band] - 1).mean()
    p = distorted
    before = err(p)
    for _ in range(20):
        p = ls.sussman(p, 0.5)
    assert err(p) < before


def test_mean_curvature_of_a_sphere():
    n = 41
    ax = np.arange(n) - n // 2
    r = np.sqrt(ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)
    phi = np.asfortranarray(r)  # distance from the centre: mean curvature should be 2 / r
    idx = np.ravel_multi_index((n // 2 + 10, n // 2, n // 2), phi.shape, order="F")
    k = ls.mean_curvature(phi, np.array([idx]))
    assert np.isclose(k[0], 2 / 10, rtol=0.1)
    assert np.isclose(-ls.mean_curvature(phi, np.array([idx]))[0], -k[0])  # odd in phi
    assert np.isclose(ls.mean_curvature(-phi, np.array([idx]))[0], -k[0])


def test_get_normals_keeps_the_matlab_xy_swap():
    phi = np.zeros((9, 9, 9))
    phi += np.arange(9)[:, None, None] * 1.0   # varies along dim 1 only
    idx = np.array([np.ravel_multi_index((4, 4, 4), phi.shape, order="F")])
    n = ls.get_normals(phi, idx)[0]
    assert abs(n[1]) > 0.99 and abs(n[0]) < 1e-3   # MATLAB's FY (d/d dim 1) lands in column 2


# ---------------------------------------------------------------- strokes
def test_stroke_mask_and_signs():
    ch = stroke_change([(10, 10), (30, 10)], 40, 50)
    assert ch.shape == (40, 50) and set(np.unique(ch)) == {0.0, 2.0}
    assert ch[10, 20] == 2.0 and ch[30, 20] == 0.0
    u = build_umask([{"type": "in", "pts": [(10, 10), (30, 10)]},
                     {"type": "out", "pts": [(10, 30), (30, 30)]}], 40, 50)
    assert u[10, 20] == 6.0     # erase_from: +3 * 2  -> inside is positive
    assert u[30, 20] == -2.0    # interact_to: -2     -> outside is negative


def test_stroke_is_clamped_to_the_image():
    ch = stroke_change([(-5, -5), (100, 100)], 20, 20)
    assert ch.shape == (20, 20) and ch.max() == 2.0


def test_salt_and_pepper_density():
    img = np.full((200, 200), 128, np.uint8)
    out = imnoise_salt_pepper(img, 0.9, np.random.default_rng(1))
    changed = (out != 128).mean()
    assert 0.88 < changed < 0.92
    assert 0.4 < (out == 255).sum() / (out != 128).sum() < 0.6


# ---------------------------------------------------------------- solver
def _synthetic(n_views=36, rows=80, cols=50):
    gt = np.zeros((rows, cols)); gt[20:60, 15:35] = 1
    return [(gt * 255).astype(np.uint8)] * n_views, [gt] * n_views


def test_solver_runs_and_carves_on_clean_data():
    imgs, gts = _synthetic()
    s = Solver(imgs, gts, Config(grid=40, focal=130.0))
    assert s.phi.shape[0] == 40
    v0 = int((s.phi >= 0).sum())
    out = [s.step() for _ in range(6)]
    assert out[-1]["iter"] == 6 and len(s.tracks["energy"]) == 6
    assert all(np.isfinite(o["energy"]) for o in out)
    assert out[-1]["fp"] + out[-1]["fn"] <= out[0]["fp"] + out[0]["fn"]
    assert (s.phi >= 0).sum() != v0   # the surface moved


def test_solver_exports_for_the_ui():
    imgs, gts = _synthetic(n_views=36)
    s = Solver(imgs, gts, Config(grid=40, focal=130.0))
    s.step()
    o = s.outline(0)
    assert o.dtype == np.int32 and o.size > 0 and o.max() < 80 * 50
    pts = s.surface_points(max_points=500)
    assert pts.shape[1] == 3 and pts.shape[0] <= 500 and np.abs(pts).max() < 1.5


def test_strokes_steer_phi_hat():
    imgs, gts = _synthetic()
    a = Solver(imgs, gts, Config(grid=40, focal=130.0))
    b = Solver(imgs, gts, Config(grid=40, focal=130.0))
    # outside marks across every view along the projected silhouette region
    for v in range(36):
        b.set_strokes(v, [{"type": "out", "pts": [(x, y) for x in range(5, 45) for y in (10, 25, 40, 55, 70)]}])
    for _ in range(4):
        a.step(); b.step()
    assert not np.allclose(a.phi_hat, b.phi_hat)

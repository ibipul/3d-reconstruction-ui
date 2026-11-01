import numpy as np

from recon3d.surface import surface_mesh


def _sphere(n=41, r=12.0, center=(20.0, 20.0, 20.0)):
    ax = np.arange(n)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    return r - np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)


def _volume(v, t):
    a, b, c = v[t[:, 0]].astype(float), v[t[:, 1]].astype(float), v[t[:, 2]].astype(float)
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def test_sphere_mesh_geometry():
    v, nrm, t = surface_mesh(_sphere(), origin=(20, 20, 20))
    r = np.linalg.norm(v, axis=1)
    assert abs(r.mean() - 12.0) < 0.2 and r.std() < 0.35
    assert (np.einsum("ij,ij->i", v / r[:, None], nrm) > 0.95).all()      # normals point outward
    assert abs(_volume(v, t) - 4 / 3 * np.pi * 12 ** 3) / (4 / 3 * np.pi * 12 ** 3) < 0.03  # + => outward winding


def test_mesh_is_closed_and_consistently_oriented():
    v, _, t = surface_mesh(_sphere(r=9.5))
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]).astype(np.int64)
    key_fwd = e[:, 0] * len(v) + e[:, 1]
    key_rev = e[:, 1] * len(v) + e[:, 0]
    # every directed edge must appear exactly once and have its reverse present
    assert len(np.unique(key_fwd)) == len(key_fwd)
    assert np.isin(key_fwd, key_rev).all()


def test_box_and_empty_and_scaling():
    phi = -np.ones((12, 12, 12)); phi[3:8, 4:9, 2:10] = 1.0
    v, _, t = surface_mesh(phi, origin=(0, 0, 0), scale=2.0)
    assert len(t) > 0 and v[:, 0].min() > 3 * 2 - 2 and v[:, 0].max() < 8 * 2
    assert _volume(v, t) > 0
    ev, en, et = surface_mesh(-np.ones((5, 5, 5)))
    assert len(ev) == 0 and len(et) == 0

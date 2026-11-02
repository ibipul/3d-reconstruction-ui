import json
import struct

import numpy as np
import pytest

from recon3d import bridge, meshio
from recon3d.solver import Config, Solver
from recon3d.surface import surface_mesh


def _sphere_mesh(r=12.0, scale=1.0):
    n = 41
    ax = np.arange(n)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    phi = r - np.sqrt((x - 20.0) ** 2 + (y - 20.0) ** 2 + (z - 20.0) ** 2)
    return surface_mesh(phi, origin=(20, 20, 20), scale=scale)


def _volume(v, t):
    v = np.asarray(v, float)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


# ---- independent readers -----------------------------------------------------------------------
def read_obj(data):
    v, vn, f, comments = [], [], [], []
    for line in data.decode("ascii").splitlines():
        if line.startswith("#"): comments.append(line)
        elif line.startswith("vn "): vn.append([float(x) for x in line.split()[1:]])
        elif line.startswith("v "): v.append([float(x) for x in line.split()[1:]])
        elif line.startswith("f "): f.append([int(p.split("//")[0]) for p in line.split()[1:]])
    return np.array(v), np.array(vn), np.array(f), comments


def read_stl(data):
    count = struct.unpack("<I", data[80:84])[0]
    assert len(data) == 84 + 50 * count                       # 50 bytes per triangle
    rec = np.frombuffer(data[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return data[:80], rec


def read_ply(data):
    head, body = data.split(b"end_header\n", 1)
    h = head.decode("ascii").splitlines()
    assert h[0] == "ply" and "format binary_little_endian 1.0" in h
    nv = int([x for x in h if x.startswith("element vertex")][0].split()[2])
    nf = int([x for x in h if x.startswith("element face")][0].split()[2])
    vr = np.frombuffer(body[:nv * 24], dtype=[("p", "<f4", 3), ("n", "<f4", 3)])
    fr = np.frombuffer(body[nv * 24:], dtype=[("k", "u1"), ("i", "<i4", 3)])
    assert len(fr) == nf and len(body) == nv * 24 + nf * 13
    return h, vr, fr


# ---- the writers ---------------------------------------------------------------------------------
def test_obj_roundtrip():
    v, n, t = _sphere_mesh()
    rv, rn, rf, comments = read_obj(meshio.to_obj(v, n, t, "line one\nline two"))
    assert comments == ["# line one", "# line two"]
    assert rv.shape == v.shape and rn.shape == n.shape and rf.shape == t.shape
    assert np.abs(rv - v).max() < 1e-4 and np.abs(rn - n).max() < 1e-3
    assert rf.min() == 1 and rf.max() == len(v)                    # 1-based indices
    assert (rf - 1 == t).all()


def test_drop_degenerate_removes_only_zero_area_triangles():
    v, n, t = _sphere_mesh()
    area = lambda tt: 0.5 * np.linalg.norm(np.cross(v[tt[:, 1]] - v[tt[:, 0]], v[tt[:, 2]] - v[tt[:, 0]]), axis=1)
    assert (area(t) < 1e-9).sum() > 0                          # the raw mesh does contain some
    _, _, clean = meshio.drop_degenerate(v, n, t)
    assert (area(clean) > 1e-9).all() and len(clean) == len(t) - (area(t) <= 1e-9).sum()
    assert abs(_volume(v, clean.astype(np.int64)) - _volume(v, t.astype(np.int64))) < 1e-3      # same solid


def test_stl_roundtrip_and_normals():
    v, n, t = _sphere_mesh()
    v, n, t = meshio.drop_degenerate(v, n, t)
    header, rec = read_stl(meshio.to_stl(v, t, "recon3d test"))
    assert header.startswith(b"recon3d test") and len(header) == 80
    assert len(rec) == len(t)
    assert np.abs(rec["v"] - v[t]).max() < 1e-6
    assert np.allclose(np.linalg.norm(rec["n"], axis=1), 1, atol=1e-3)
    centre = rec["v"].mean(axis=1)                                  # on a sphere, outward = away from the centre
    assert (np.einsum("ij,ij->i", rec["n"], centre) > 0).mean() > 0.99


def test_ply_roundtrip():
    v, n, t = _sphere_mesh()
    h, vr, fr = read_ply(meshio.to_ply(v, n, t, "hello"))
    assert "comment hello" in h
    assert np.abs(vr["p"] - v).max() < 1e-6 and np.abs(vr["n"] - n).max() < 1e-6
    assert (fr["k"] == 3).all() and (fr["i"] == t).all()


def test_all_formats_describe_the_same_solid():
    v, n, t = _sphere_mesh(r=12.0)
    expected = 4 / 3 * np.pi * 12 ** 3
    vol_obj = _volume(read_obj(meshio.to_obj(v, n, t))[0], read_obj(meshio.to_obj(v, n, t))[2] - 1)
    rec = read_stl(meshio.to_stl(v, t))[1]
    vol_stl = float(np.einsum("ij,ij->i", rec["v"][:, 0], np.cross(rec["v"][:, 1], rec["v"][:, 2])).sum() / 6)
    _, vr, fr = read_ply(meshio.to_ply(v, n, t))
    vol_ply = _volume(vr["p"], fr["i"])
    for vol in (vol_obj, vol_stl, vol_ply):
        assert vol > 0 and abs(vol - expected) / expected < 0.03   # positive = outward winding


def test_unknown_format_is_rejected():
    v, n, t = _sphere_mesh()
    with pytest.raises(ValueError):
        meshio.write("fbx", v, n, t)
    assert meshio.write("OBJ", v, n, t).startswith(b"o reconstruction")      # case-insensitive


# ---- the exported reconstruction ------------------------------------------------------------------
def _solver(steps=3):
    gt = np.zeros((80, 50)); gt[20:60, 15:35] = 1
    s = Solver([(gt * 255).astype(np.uint8)] * 36, [gt] * 36, Config(grid=40, focal=130.0))
    for _ in range(steps):
        s.step()
    return s


def test_exported_model_is_in_world_units_of_the_camera_setup():
    s = _solver()
    v, n, t = s.export_mesh()
    assert len(t) > 0 and np.isfinite(v).all()
    # same frame as the projected voxels (world units, centred on the object): centroids agree to ~2 voxels
    inside = s.coords[:, :3]
    assert np.abs(v.mean(axis=0) - inside.mean(axis=0)).max() < 2 * s.voxel_scale * 2
    # and the extent matches the interior's extent (the surface sits half a voxel outside the voxel centres)
    for ax in range(3):
        assert abs(v[:, ax].max() - inside[:, ax].max()) < 1.5 * s.voxel_scale
        assert abs(v[:, ax].min() - inside[:, ax].min()) < 1.5 * s.voxel_scale
    assert _volume(v, t.astype(np.int64)) > 0


def test_bridge_export_model_formats_and_metadata():
    s = _solver()
    obj = bridge.export_model(s, "obj")
    rv, rn, rf, comments = read_obj(obj)
    assert any("after 3 iterations, grid 40" in c for c in comments) and any("z up" in c for c in comments)
    assert len(rf) > 0
    assert bridge.export_model(s, "stl")[:80].startswith(b"recon3d reconstruction")
    assert bridge.export_model(s, "ply").startswith(b"ply\n")
    assert bridge.export_model(s, "obj", hat=True).startswith(b"# recon3d")        # the estimate exports too


def test_nothing_to_export_is_an_error_not_an_empty_file():
    s = _solver(1)
    s.phi[...] = -1.0
    with pytest.raises(ValueError, match="nothing to export"):
        bridge.export_model(s, "obj")

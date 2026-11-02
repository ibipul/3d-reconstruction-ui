"""Write a triangle mesh as OBJ, STL or PLY (bytes; no dependencies beyond NumPy).

All three keep the triangle winding of the input (counter-clockwise seen from outside, so normals point
out of the shape). Vertices are written as float32.
"""
from __future__ import annotations

import numpy as np

FORMATS = {"obj": "model/obj", "stl": "model/stl", "ply": "application/octet-stream"}


def face_normals(verts, tris):
    v = np.asarray(verts, np.float64)
    t = np.asarray(tris, np.int64)
    n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30)


def drop_degenerate(verts, normals, tris, min_area=1e-9):
    """Remove zero-area triangles (surface nets can make a few where vertices coincide). They render fine but give
    STL a zero normal, which some tools reject. Vertices are kept as they are."""
    v = np.asarray(verts, np.float64)
    t = np.asarray(tris, np.int64)
    area = 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
    return verts, normals, tris[area > min_area]


def to_obj(verts, normals, tris, comment=""):
    """Wavefront OBJ (ASCII) with vertex normals; faces are 1-based `f a//a b//b c//c`."""
    lines = ["# " + c for c in comment.splitlines()] if comment else []
    lines.append("o reconstruction")
    lines += ["v %.5f %.5f %.5f" % tuple(p) for p in np.asarray(verts, np.float32)]
    lines += ["vn %.4f %.4f %.4f" % tuple(n) for n in np.asarray(normals, np.float32)]
    t = np.asarray(tris, np.int64) + 1
    lines += ["f %d//%d %d//%d %d//%d" % (a, a, b, b, c, c) for a, b, c in t]
    return ("\n".join(lines) + "\n").encode("ascii")


def to_stl(verts, tris, comment=""):
    """Binary STL: an 80-byte header, the triangle count, then per triangle its normal and 3 vertices."""
    v = np.asarray(verts, np.float32)
    t = np.asarray(tris, np.int64)
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = face_normals(v, t)
    rec["v"] = v[t]
    header = (comment.replace("\n", " ")[:79]).encode("ascii", "replace").ljust(80, b" ")
    return header + np.uint32(len(t)).tobytes() + rec.tobytes()


def to_ply(verts, normals, tris, comment=""):
    """Binary little-endian PLY with per-vertex position and normal."""
    v = np.asarray(verts, np.float32)
    n = np.asarray(normals, np.float32)
    t = np.asarray(tris, np.int32)
    head = ["ply", "format binary_little_endian 1.0"]
    head += ["comment " + c for c in comment.splitlines()] if comment else []
    head += ["element vertex %d" % len(v), "property float x", "property float y", "property float z",
             "property float nx", "property float ny", "property float nz",
             "element face %d" % len(t), "property list uchar int vertex_indices", "end_header"]
    vr = np.zeros(len(v), dtype=[("p", "<f4", 3), ("n", "<f4", 3)])
    vr["p"], vr["n"] = v, n
    fr = np.zeros(len(t), dtype=[("k", "u1"), ("i", "<i4", 3)])
    fr["k"], fr["i"] = 3, t
    return ("\n".join(head) + "\n").encode("ascii") + vr.tobytes() + fr.tobytes()


def write(fmt, verts, normals, tris, comment=""):
    fmt = fmt.lower()
    if fmt == "obj":
        return to_obj(verts, normals, tris, comment)
    if fmt == "stl":
        return to_stl(verts, tris, comment)
    if fmt == "ply":
        return to_ply(verts, normals, tris, comment)
    raise ValueError("unknown mesh format: %s (use one of %s)" % (fmt, ", ".join(FORMATS)))

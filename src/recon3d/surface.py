"""Zero level set -> triangle mesh, for display.

Naive surface nets: one vertex per grid cell the surface passes through (the mean of the
edge crossings), one quad per sign-changing grid edge. Table-free, vectorised, and the mesh
is closed and consistently oriented (normals point out of the {phi >= level} region).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def empty_mesh():
    return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32))


def surface_mesh(phi, origin=(0.0, 0.0, 0.0), scale=1.0, level=0.0):
    """Returns (vertices, normals, triangles). Vertices are (voxel_index - origin) * scale;
    voxel index i is the centre of voxel i, so they line up with the solver's coordinates."""
    inside = phi >= level
    if not inside.any():
        return empty_mesh()
    lo, hi = [], []
    for ax in range(3):                      # bounding box without materialising all coordinates
        idx = np.flatnonzero(inside.any(axis=tuple(a for a in range(3) if a != ax)))
        lo.append(max(idx[0] - 2, 0)); hi.append(min(idx[-1] + 3, phi.shape[ax]))
    lo, hi = np.array(lo), np.array(hi)
    f = np.ascontiguousarray(phi[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=float) - level
    S = f >= 0
    cdims = np.array(f.shape) - 1
    ncell = int(np.prod(cdims))
    sums = np.zeros((3, ncell))
    cnt = np.zeros(ncell)
    quads = []

    for a in range(3):
        s0 = [slice(None)] * 3; s1 = [slice(None)] * 3
        s0[a] = slice(0, -1); s1[a] = slice(1, None)
        s0, s1 = tuple(s0), tuple(s1)
        cross = S[s0] != S[s1]
        p = np.argwhere(cross)
        if not p.size:
            continue
        v0, v1 = f[s0][cross], f[s1][cross]
        pos = p.astype(float)
        pos[:, a] += v0 / (v0 - v1)                 # where along the edge phi crosses the level
        lower_inside = S[s0][cross]
        b, c = (a + 1) % 3, (a + 2) % 3             # (a, b, c) is a right-handed triple
        cells = []
        for db, dc in ((-1, -1), (0, -1), (0, 0), (-1, 0)):   # counter-clockwise seen from +a
            q = p.copy(); q[:, b] += db; q[:, c] += dc
            ok = (q >= 0).all(axis=1) & (q < cdims).all(axis=1)
            flat = np.where(ok, np.ravel_multi_index(np.clip(q, 0, cdims - 1).T, cdims), -1)
            cells.append(flat)
            w = ok.astype(float)
            for d in range(3):
                sums[d] += np.bincount(np.maximum(flat, 0), weights=pos[:, d] * w, minlength=ncell)
            cnt += np.bincount(np.maximum(flat, 0), weights=w, minlength=ncell)
        q4 = np.stack(cells, axis=1)
        q4[~lower_inside] = q4[~lower_inside][:, ::-1]        # outward = from inside to outside
        quads.append(q4)

    if not quads:
        return empty_mesh()
    q4 = np.concatenate(quads)
    q4 = q4[(q4 >= 0).all(axis=1)]
    active = cnt > 0
    vid = np.full(ncell, -1, dtype=np.int64)
    vid[active] = np.arange(int(active.sum()))
    verts = (sums[:, active] / cnt[active]).T                   # crop-local voxel coordinates
    vq = vid[q4]
    tris = np.concatenate([vq[:, [0, 1, 2]], vq[:, [0, 2, 3]]])

    grads = [ndimage.map_coordinates(np.gradient(f, axis=a), verts.T, order=1, mode="nearest")
             for a in range(3)]
    n = -np.stack(grads, axis=1)                                # phi is larger inside: outward = -grad
    n /= np.sqrt((n * n).sum(axis=1))[:, None] + 1e-9

    world = (verts + lo - np.asarray(origin, float)) * scale
    return world.astype(np.float32), n.astype(np.float32), tris.astype(np.uint32)

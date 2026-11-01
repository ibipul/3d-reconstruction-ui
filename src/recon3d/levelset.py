"""Level-set machinery on a 3D grid.

Ports shift*.m, sussman.m, sussman_sign.m, get_normals.m and mean_curvature.m.
Arrays are indexed [i, j, k] = MATLAB dims (1, 2, 3). Flat indices are Fortran-order
(MATLAB's column-major), so `find`/`ind2sub`/`intersect` results line up.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


# ---- shift operators (edge value repeated). Names follow the MATLAB files. ----
def _shift(M, axis, step):
    n = M.shape[axis]
    idx = np.clip(np.arange(n) + step, 0, n - 1)
    return np.take(M, idx, axis=axis)


def shiftB(M): return _shift(M, 0, +1)   # M(i+1,:,:)
def shiftF(M): return _shift(M, 0, -1)   # M(i-1,:,:)
def shiftD(M): return _shift(M, 1, +1)   # M(:,j+1,:)
def shiftU(M): return _shift(M, 1, -1)   # M(:,j-1,:)
def shiftL(M): return _shift(M, 2, +1)   # M(:,:,k+1)
def shiftR(M): return _shift(M, 2, -1)   # M(:,:,k-1)


def sussman_sign(D):
    return D / np.sqrt(D * D + 1.0)


def sussman(D, dt):
    """One re-initialisation step (|grad D| -> 1). Mirrors sussman.m term for term,
    including its sign choices for the j-direction (c, d)."""
    a = D - shiftR(D)
    b = shiftL(D) - D
    c = D - shiftD(D)
    d = shiftU(D) - D
    e = D - shiftF(D)
    f = shiftB(D) - D

    def pos(x): return np.where(x < 0, 0.0, x)
    def neg(x): return np.where(x > 0, 0.0, x)

    a_p, a_n, b_p, b_n = pos(a), neg(a), pos(b), neg(b)
    c_p, c_n, d_p, d_n = pos(c), neg(c), pos(d), neg(d)
    e_p, e_n, f_p, f_n = pos(e), neg(e), pos(f), neg(f)

    dD = np.zeros_like(D)
    P = D > 0
    N = D < 0
    dD[P] = np.sqrt(np.maximum(a_p[P] ** 2, b_n[P] ** 2) + np.maximum(c_p[P] ** 2, d_n[P] ** 2)
                    + np.maximum(e_p[P] ** 2, f_n[P] ** 2)) - 1.0
    dD[N] = np.sqrt(np.maximum(a_n[N] ** 2, b_p[N] ** 2) + np.maximum(c_n[N] ** 2, d_p[N] ** 2)
                    + np.maximum(e_n[N] ** 2, f_p[N] ** 2)) - 1.0
    return D - dt * sussman_sign(D) * dD


def get_normals(phi, flat_idx):
    """Unit normals at `flat_idx`, ordered like MATLAB: [FX FY FZ] = gradient(phi) returns
    d/d(dim 2), d/d(dim 1), d/d(dim 3). The x/y swap relative to the (i, j, k) voxel
    coordinates is a quirk of the original and is kept on purpose."""
    g0 = np.gradient(phi, axis=0)
    g1 = np.gradient(phi, axis=1)
    g2 = np.gradient(phi, axis=2)
    n = np.stack([g1.ravel(order="F")[flat_idx], g0.ravel(order="F")[flat_idx],
                  g2.ravel(order="F")[flat_idx]], axis=1)
    return n / (np.sqrt(np.sum(n * n, axis=1)) + 1e-5)[:, None]


def mean_curvature(phi, flat_idx):
    """Mean curvature at the given voxels (mean_curvature.m)."""
    eps = 1e-10
    dx_, dy_, dz_ = phi.shape
    x, y, z = np.unravel_index(flat_idx, phi.shape, order="F")
    xm1, ym1, zm1 = np.maximum(x - 1, 0), np.maximum(y - 1, 0), np.maximum(z - 1, 0)
    xp1, yp1, zp1 = np.minimum(x + 1, dx_ - 1), np.minimum(y + 1, dy_ - 1), np.minimum(z + 1, dz_ - 1)
    P = lambda a, b, c: phi[a, b, c].astype(np.float64)
    c0 = P(x, y, z)
    dx = (P(xm1, y, z) - P(xp1, y, z)) / 2
    dxx = P(xm1, y, z) - 2 * c0 + P(xp1, y, z)
    dy = (P(x, ym1, z) - P(x, yp1, z)) / 2
    dyy = P(x, ym1, z) - 2 * c0 + P(x, yp1, z)
    dz = (P(x, y, zm1) - P(x, y, zp1)) / 2
    dzz = P(x, y, zm1) - 2 * c0 + P(x, y, zp1)
    dxy = (P(xm1, ym1, z) + P(xp1, yp1, z) - P(xp1, ym1, z) - P(xm1, yp1, z)) / 4
    dxz = (P(xm1, y, zp1) + P(xp1, y, zm1) - P(xp1, y, zp1) - P(xm1, y, zm1)) / 4
    dyz = (P(x, ym1, zp1) + P(x, yp1, zm1) - P(x, yp1, zp1) - P(x, ym1, zm1)) / 4
    dx2, dy2, dz2 = dx * dx, dy * dy, dz * dz
    num = (dxx * (dy2 + dz2) + dyy * (dx2 + dz2) + dzz * (dx2 + dy2)
           - 2 * dx * dy * dxy - 2 * dx * dz * dxz - 2 * dy * dz * dyz)
    return num / (dx2 + dy2 + dz2 + eps)


def initial_box(space_length: int, rec_length: int, off_a: int, off_b: int):
    """The initial rectangular mask built in the loaders.

    MATLAB builds it by concatenating slices, which gives a z extent of
    (nb+1) + (rec+1) + (nb+1) = space_length + 3 when space-rec = 2*nb. That quirk is kept.
    `off_a`/`off_b` are the loader's `start+30 .. end-30` and `start+10 .. end-10` insets.
    """
    nb = (space_length - rec_length) // 2
    start, end = nb, rec_length + nb
    nz = np.zeros((space_length, space_length))
    # MATLAB ranges are 1-based inclusive: i = start+a .. end-a
    nz[start + off_a - 1: end - off_a, start + off_b - 1: end - off_b] = 1.0
    depth = (nb + 1) + (rec_length + 1) + (nb + 1)
    mask = np.zeros((space_length, space_length, depth))
    mask[:, :, nb + 1: nb + 1 + rec_length + 1] = nz[:, :, None]
    return mask


def signed_distance(mask):
    """phi = -(bwdist(mask) - bwdist(1-mask) + mask): positive inside."""
    inside = ndimage.distance_transform_edt(mask)          # = bwdist(1 - mask)
    outside = ndimage.distance_transform_edt(1.0 - mask)   # = bwdist(mask)
    return -(outside - inside + mask)


# ----------------------------------------------------------------------------------------
# Memory-lean variants for large grids (200^3, 300^3). Same results as the full-grid
# functions above wherever the result matters (near the surface and inside the object).
# ----------------------------------------------------------------------------------------
def initial_box_extent(space_length: int, rec_length: int, off_a: int, off_b: int):
    """Inclusive 0-based voxel ranges (i0, i1, j0, j1, k0, k1) of the initial box, and the volume depth.
    Same box as `initial_box`."""
    nb = (space_length - rec_length) // 2
    start, end = nb, rec_length + nb
    i0, i1 = start + off_a - 1, end - off_a - 1
    j0, j1 = start + off_b - 1, end - off_b - 1
    k0, k1 = nb + 1, nb + 1 + rec_length
    depth = (nb + 1) + (rec_length + 1) + (nb + 1)
    return (i0, i1, j0, j1, k0, k1), depth


def initial_box_phi(space_length: int, rec_length: int, off_a: int, off_b: int, dtype=np.float64, slab: int = 24):
    """Signed distance of the initial box, computed analytically (no mask volume, no EDT).

    For an axis-aligned box of voxels the Euclidean distance transform has a closed form:
    outside, the distance to the nearest box voxel is the norm of the per-axis overshoot;
    inside, the distance to the nearest background voxel is the smallest per-axis distance to a
    face. This equals `signed_distance(initial_box(...))` exactly, and is built slab by slab so
    the peak memory stays near the size of the result.
    """
    (i0, i1, j0, j1, k0, k1), depth = initial_box_extent(space_length, rec_length, off_a, off_b)
    nx = ny = space_length
    out = np.empty((nx, ny, depth), dtype=dtype, order="F")
    j = np.arange(ny, dtype=np.float64)[None, :, None]
    k = np.arange(depth, dtype=np.float64)[None, None, :]
    oy = np.maximum(np.maximum(j0 - j, j - j1), 0.0)
    oz = np.maximum(np.maximum(k0 - k, k - k1), 0.0)
    fy = np.minimum(j - j0 + 1, j1 - j + 1)
    fz = np.minimum(k - k0 + 1, k1 - k + 1)
    for s in range(0, nx, slab):
        i = np.arange(s, min(s + slab, nx), dtype=np.float64)[:, None, None]
        ox = np.maximum(np.maximum(i0 - i, i - i1), 0.0)
        fx = np.minimum(i - i0 + 1, i1 - i + 1)
        outside = np.sqrt(ox * ox + oy * oy + oz * oz)
        inside = (ox == 0) & (oy == 0) & (oz == 0)
        depth_in = np.minimum(np.minimum(fx, fy), fz)
        out[s:s + slab] = np.where(inside, depth_in - 1.0, -outside)
    return out


def get_normals_at(phi, I, J, K):
    """get_normals for voxels given by 0-based subscripts. Gradients are taken on a crop around
    the voxels (with a 1-voxel margin, which central differences need), so the cost follows the
    object, not the whole grid. Same x/y swap as `get_normals`."""
    lo = np.array([I.min(), J.min(), K.min()])
    lo = np.maximum(lo - 1, 0)
    hi = np.minimum(np.array([I.max(), J.max(), K.max()]) + 2, phi.shape)
    c = phi[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    ii, jj, kk = I - lo[0], J - lo[1], K - lo[2]
    g0 = np.gradient(c, axis=0)[ii, jj, kk]
    g1 = np.gradient(c, axis=1)[ii, jj, kk]
    g2 = np.gradient(c, axis=2)[ii, jj, kk]
    n = np.stack([g1, g0, g2], axis=1).astype(np.float64)
    return n / (np.sqrt(np.sum(n * n, axis=1)) + 1e-5)[:, None]


def sussman_local(phi, dt, floor=-12.0, margin=2):
    """`sussman` applied in place on the box around {phi > floor}.

    Re-initialisation only changes the sign-relevant neighbourhood of the surface; voxels deeper
    than `-floor` outside never come near it, and the sign (which is all that is used there) is
    preserved. Returns phi."""
    near = phi > floor
    if not near.any():
        return phi
    sl = []
    for ax in range(3):
        other = tuple(a for a in range(3) if a != ax)
        idx = np.flatnonzero(near.any(axis=other))
        sl.append(slice(max(idx[0] - margin, 0), min(idx[-1] + margin + 1, phi.shape[ax])))
    sl = tuple(sl)
    phi[sl] = sussman(phi[sl].astype(np.float64), dt).astype(phi.dtype)
    return phi

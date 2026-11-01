"""Silhouettes, visibility and image statistics.

Ports get_mask.m, project_phi_to_image.m and get_image_stat.m, plus the MATLAB-style
morphology they rely on (imclose / imerode border handling, strel('disk', n)).
Pixel flat indices are Fortran-order (y + rows * x), like MATLAB's sub2ind([row col], y, x).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


class Disk:
    """Octagonal structuring element standing in for MATLAB's strel('disk', R).

    MATLAB builds its disk from line segments, which makes it an octagon. Here it is the set
    {|dx| <= R, |dy| <= R, |dx|+|dy| <= L} with L = round(R * sqrt(2)); that reproduces the
    known neighbourhoods for R=3 and R=4. Whether R=6 matches exactly has to be confirmed
    against a MATLAB fixture (see docs/matlab-architecture.md).

    When the octagon is a Minkowski sum of a horizontal, a vertical and two diagonal
    segments, `segments` holds them, and dilation/erosion run as a few array shifts
    instead of a scan of the whole neighbourhood. Results are identical (tested).
    """

    def __init__(self, radius: int):
        r = int(radius)
        L = int(round(r * np.sqrt(2.0)))
        d = np.abs(np.arange(-r, r + 1))
        self.arr = (d[:, None] <= r) & (d[None, :] <= r) & (d[:, None] + d[None, :] <= L)
        self.reach = r
        p, twoq = L - r, 2 * r - L
        self.segments = None
        if p >= 0 and twoq >= 0 and twoq % 2 == 0:
            q = twoq // 2
            self.segments = [(0, 1, p), (1, 0, p), (1, 1, q), (1, -1, q)]


def disk(radius: int) -> Disk:
    return Disk(radius)


def _arr(nhood):
    return nhood.arr if isinstance(nhood, Disk) else nhood


def _reach(nhood):
    return nhood.reach if isinstance(nhood, Disk) else max(nhood.shape) // 2 + 1


def _shifted(a, sy, sx, fill):
    out = np.full_like(a, fill)
    h, w = a.shape
    if abs(sy) >= h or abs(sx) >= w:
        return out
    out[max(sy, 0):h + min(sy, 0), max(sx, 0):w + min(sx, 0)] = \
        a[max(-sy, 0):h + min(-sy, 0), max(-sx, 0):w + min(-sx, 0)]
    return out


def _segment_op(a, dy, dx, half, dilate):
    res = a.copy()
    fill = not dilate  # dilation pads with 0, erosion with 1 (MATLAB border rules)
    for t in range(1, half + 1):
        for sgn in (1, -1):
            sh = _shifted(a, sgn * t * dy, sgn * t * dx, fill)
            if dilate:
                res |= sh
            else:
                res &= sh
    return res


def _crop(mask, pad):
    """Bounding box of the True pixels, grown by `pad` and clipped to the image.

    Running morphology on the crop gives the same result as on the whole image whenever
    pad > the structuring element's reach. Where the crop touches the image edge the MATLAB
    border rule still applies, because the crop edge is the image edge.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    r, c = mask.shape
    return (slice(max(rows[0] - pad, 0), min(rows[-1] + pad + 1, r)),
            slice(max(cols[0] - pad, 0), min(cols[-1] + pad + 1, c)))


def _morph(a, nhood, dilate):
    if isinstance(nhood, Disk) and nhood.segments is not None:
        # Pad by the full reach with the border value first: intermediate steps must see
        # the true values just outside the image, not a constant, or edges come out wrong.
        r = nhood.reach
        a = np.pad(a, r, mode="constant", constant_values=not dilate)
        for dy, dx, half in nhood.segments:
            a = _segment_op(a, dy, dx, half, dilate)
        return a[r:-r, r:-r]
    if dilate:
        return ndimage.binary_dilation(a, structure=_arr(nhood), border_value=0)
    return ndimage.binary_erosion(a, structure=_arr(nhood), border_value=1)


def imerode(mask, nhood):
    # MATLAB pads erosion with 1s, so shapes touching the border are not eroded away.
    sl = _crop(mask, _reach(nhood) + 1)
    out = np.zeros_like(mask)
    if sl is not None:
        out[sl] = _morph(mask[sl], nhood, dilate=False)
    return out


def imdilate(mask, nhood):
    sl = _crop(mask, _reach(nhood) + 1)
    out = np.zeros_like(mask)
    if sl is not None:
        out[sl] = _morph(mask[sl], nhood, dilate=True)
    return out


def imclose(mask, nhood):
    sl = _crop(mask, 2 * _reach(nhood) + 2)
    out = np.zeros_like(mask)
    if sl is not None:
        out[sl] = _morph(_morph(mask[sl], nhood, dilate=True), nhood, dilate=False)
    return out


def get_mask(rows, cols, px, py, nhood):
    """Scatter projected voxels into an image and close it. Returns (mask, flat_pixel_idx)."""
    m = np.zeros((rows, cols), dtype=bool)
    m[py, px] = True
    return imclose(m, nhood), (py + rows * px)


def silhouette_band(mask, nhood):
    """|mask - erode(mask)| (the thick edge band used by project_phi_to_image)."""
    return mask != imerode(mask, nhood)


def project_phi_to_image(mask, pixel_idx, camera_loc, coords, normals, nhood, eps=120.0):
    """Pick the silhouette voxels that actually face the camera.

    Returns (occluding_voxel_idx, silhouette_pixel_idx): positions into the voxel list and
    flat pixel indices, aligned element by element.
    """
    edge = np.flatnonzero(silhouette_band(mask, nhood).ravel(order="F"))
    _, ie, ii = np.intersect1d(edge, pixel_idx, return_indices=True)  # first occurrences
    cand = ii
    if cand.size == 0:
        return cand, edge[ie]
    pts = coords[cand, :3]
    vv = pts - np.asarray(camera_loc, float)
    vv = vv / np.sqrt(np.sum(vv * vv, axis=1))[:, None]
    dt = np.sum(vv * normals[cand], axis=1)
    v_plus = np.flatnonzero(dt > -0.15)
    v_minus = np.flatnonzero(dt < 0.15)
    flag = np.zeros(cand.size, dtype=bool)
    if v_plus.size and v_minus.size:
        # A point is kept if a counterpart from the other set lies within eps.
        d_pm, _ = cKDTree(pts[v_minus]).query(pts[v_plus])
        flag[v_plus[d_pm < eps]] = True
        d_mp, _ = cKDTree(pts[v_plus]).query(pts[v_minus])
        flag[v_minus[d_mp < eps]] = True
    return cand[flag], edge[ie[flag]]


def get_image_stat(img, mask):
    """Mean intensity inside/outside the mask: (in, out) each a length-3 array.

    Colour images average only non-zero products (MATLAB `mean(nonzeros(...))`), so
    black pixels are ignored. Grayscale fills only the first channel.
    """
    m = mask.astype(np.uint8)
    a_in = float(m.sum())
    a_out = float((1 - m).sum())
    mu_in = np.zeros(3)
    mu_out = np.zeros(3)
    if a_out == 0:
        return mu_in, mu_out
    if img.ndim == 3:
        for c in range(3):
            pin = (m * img[:, :, c])[m * img[:, :, c] != 0]
            pout = ((1 - m) * img[:, :, c])[(1 - m) * img[:, :, c] != 0]
            mu_in[c] = pin.mean() if pin.size else 0.0
            mu_out[c] = pout.mean() if pout.size else 0.0
    else:
        mu_in[0] = float((m * img).sum(dtype=np.float64)) / a_in if a_in else 0.0
        mu_out[0] = float(((1 - m) * img).sum(dtype=np.float64)) / a_out
    return mu_in, mu_out

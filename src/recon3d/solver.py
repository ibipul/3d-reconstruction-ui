"""The reconstruction loop of loader_wt_control_rec_error_trace.m as a pure, steppable object.

No UI, no file I/O: images come in as arrays so the same code runs locally and in Pyodide.
Differences from the MATLAB script are limited to:
  * 0-based arrays (centred coordinates are identical),
  * a configurable grid size. The MATLAB constants are defined at 150^3; at other sizes
    the box insets scale with grid/150 and voxel coordinates are multiplied by 150/grid so
    the projection geometry (camera distance 160, focal length) is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import camera as cam
from . import levelset as ls
from . import silhouette as sil
from .strokes import build_umask
from .surface import surface_mesh

REF_GRID = 150


class ShapeCollapsed(RuntimeError):
    """A level set has no interior left (phi >= 0 nowhere), so there is nothing to project.

    Happens when the evolution shrinks the shape away, e.g. with very strong smoothing or a long run
    on heavily noisy input. `which` is 'phi' or 'phi_hat'."""

    def __init__(self, which, iteration):
        super().__init__("the %s level set has no interior left at iteration %d" % (which, iteration))
        self.which, self.iteration = which, iteration
TRACE_SQ_DT = 0.5


@dataclass
class Config:
    grid: int = 100               # voxels per side (MATLAB: 150)
    focal: float = 520.0          # rec_error_trace: f = 520
    rec_length_ref: int = 80      # at grid 150
    init_box_angle: float = 0.0   # degrees: the initial box turned about the vertical axis (only with init_box)
    camera_height: float = 0.0    # world units: the cameras raised above the ring plane (the MATLAB setups are level)
    init_box: tuple | None = None  # initial surface: (x, y, z) size in voxels at grid 150; None = the MATLAB box (20, 60, 80)
    narrow_band_high: float = 5.0
    sussman_every: int = 5
    sussman_dt: float = 0.5
    curvature_weight: float = 0.001
    eps_visibility: float = 120.0
    camera_dist: float = 4.0
    loc_scale: float = 10.0       # loaders: camera_loc = cam_mat(4,:) * 10

    def scale(self) -> float:
        return self.grid / REF_GRID


def imnoise_salt_pepper(img, density, rng):
    """MATLAB imnoise(img, 'salt & pepper', d): a fraction d of pixels set to 0 or max."""
    out = img.copy()
    r = rng.random(img.shape)
    hi = np.iinfo(img.dtype).max
    out[r < density / 2] = 0
    out[(r >= density / 2) & (r < density)] = hi
    return out


class Solver:
    def __init__(self, images, gt_masks=None, config: Config | None = None):
        """images: list of observed images (2D gray or 3D RGB uint8).
        gt_masks: list of 2D {0,1} ground-truth silhouettes (for metrics only), or None for a dataset
        without ground truth; the tp/tn/fp/fn/rms numbers are then None."""
        self.cfg = cfg = config or Config()
        self.images = [np.ascontiguousarray(im) for im in images]
        self.gt = None if gt_masks is None else [np.asarray(g, dtype=float) for g in gt_masks]
        self.n_views = len(self.images)
        self.nhood = sil.disk(6)
        self.nhood_small = sil.disk(4)
        self.iter = 0

        # cameras
        self.views = []
        for i in range(self.n_views):
            img = self.images[i]
            sz = img.shape[:2]
            cx, cy, cz, loc = cam.select_camera_auto(i + 1, cfg.camera_dist, -cfg.camera_height / cfg.loc_scale)   # world z points down
            loc = loc * cfg.loc_scale
            P = cam.create_projection_operator(cx, cy, cz, loc, sz, cfg.focal)
            self.views.append({
                "sz": sz, "loc": loc, "P": P,
                "umask": np.zeros(sz), "strokes": [],
            })

        # level sets
        s = cfg.scale()
        space = cfg.grid
        if cfg.init_box is None:
            rec = int(round(cfg.rec_length_ref * s))
            if (space - rec) % 2:
                rec += 1
            off_a, off_b = max(int(round(30 * s)), 1), max(int(round(10 * s)), 1)
            rec_z = None
        else:
            # a box of the requested x / y / z size, centred in the volume (the solver re-centres on it anyway)
            dx, dy, dz = (min(max(int(round(float(d) * s)), 2), space - 4) for d in cfg.init_box)
            rec = space                                       # footprint inset from the edges of the volume
            off_a, off_b = max((space - dx + 1) // 2, 1), max((space - dy + 1) // 2, 1)   # +1: the surface voxels (phi = 0) belong to the box
            rec_z = dz + ((space - dz) % 2)
        # float32 above the MATLAB size keeps 200^3 / 300^3 within browser memory
        dtype = np.float64 if cfg.grid <= REF_GRID else np.float32
        if cfg.init_box is not None:
            self.phi = ls.rotated_box_phi(space, rec, off_a, off_b, cfg.init_box_angle, dtype, rec_z=rec_z)
        else:
            self.phi = ls.initial_box_phi(space, rec, off_a, off_b, dtype, rec_z=rec_z)
        self.phi_hat = self.phi.copy(order="F")
        self.shape = self.phi.shape
        self.voxel_scale = REF_GRID / cfg.grid

        # centring offset m (MATLAB: mean of the initial interior, then m(3) += 1)
        in_pt = np.flatnonzero(self.phi.ravel(order="F") >= 0)
        I, J, K = np.unravel_index(in_pt, self.shape, order="F")
        m = np.array([I.mean(), J.mean(), K.mean()])
        m[2] += 1.0
        self.m = m

        self.tracks = {k: [] for k in
                       ("energy", "ienergy", "rms", "lyapunov", "lyap1", "lyap2", "tp", "tn", "fp", "fn")}
        self.cache = [None] * self.n_views
        self.project_all()

    # ------------------------------------------------------------------ geometry
    def _voxels(self, phi, which="phi"):
        pf = phi.ravel(order="F")
        in_pt = np.flatnonzero(pf >= 0)
        if in_pt.size == 0:
            raise ShapeCollapsed(which, self.iter)
        I, J, K = np.unravel_index(in_pt, self.shape, order="F")
        normals = -ls.get_normals_at(phi, I, J, K)    # MATLAB: get_normals(-phi, in_pt)
        coords = np.stack([I - self.m[0], J - self.m[1], K - self.m[2], np.ones(in_pt.size)], axis=1)
        coords[:, :3] *= self.voxel_scale
        return in_pt, normals, coords

    def project_all(self):
        """Pass 1 of each iteration: project both shapes into every view."""
        cfg = self.cfg
        self.in_pt, self.normals, self.coords = self._voxels(self.phi)
        self.in_pt_hat, self.normals_hat, self.coords_hat = self._voxels(self.phi_hat, "phi_hat")
        in_sum = np.zeros(3); out_sum = np.zeros(3)
        iin_sum = np.zeros(3); iout_sum = np.zeros(3)
        for v, view in enumerate(self.views):
            rows, cols = view["sz"]
            px, py, z = cam.project_points(self.coords, view["P"], view["sz"])
            pxh, pyh, _ = cam.project_points(self.coords_hat, view["P"], view["sz"])
            mask, in_img = sil.get_mask(rows, cols, px, py, self.nhood)
            mask_hat, in_img_hat = sil.get_mask(rows, cols, pxh, pyh, self.nhood)
            occ, sp = sil.project_phi_to_image(mask, in_img, view["loc"], self.coords, self.normals,
                                               self.nhood, cfg.eps_visibility)
            occ_h, sp_h = sil.project_phi_to_image(mask_hat, in_img_hat, view["loc"], self.coords_hat,
                                                   self.normals_hat, self.nhood, cfg.eps_visibility)
            mi, mo = sil.get_image_stat(self.images[v], mask)
            imi, imo = sil.get_image_stat(self.images[v], mask_hat)
            in_sum += mi; out_sum += mo; iin_sum += imi; iout_sum += imo
            self.cache[v] = {"mask": mask, "mask_hat": mask_hat, "occ": occ, "sil": sp,
                             "occ_hat": occ_h, "sil_hat": sp_h, "depth": z[occ]}
        n = self.n_views
        self.in_mean, self.out_mean = in_sum / n, out_sum / n
        self.iin_mean, self.iout_mean = iin_sum / n, iout_sum / n

    # ------------------------------------------------------------------ strokes
    def set_strokes(self, view: int, strokes):
        v = self.views[view]
        v["strokes"] = list(strokes)
        v["umask"] = build_umask(v["strokes"], *v["sz"])

    # ------------------------------------------------------------------ iteration
    def step(self):
        """One iteration (MATLAB `for iter`). Returns a dict of this iteration's numbers."""
        cfg = self.cfg
        self.iter += 1
        it = self.iter
        if it > 1:
            self.project_all()
        in_m, out_m = self.in_mean, self.out_mean

        pf = self.phi.ravel(order="F")
        pf_hat = self.phi_hat.ravel(order="F")
        band = np.flatnonzero((pf >= 0) & (pf <= cfg.narrow_band_high))
        band_hat = np.flatnonzero((pf_hat >= 0) & (pf_hat <= cfg.narrow_band_high))

        rms = lya = lya1 = lya2 = 0.0
        tp = tn = fp = fn = 0
        for v, view in enumerate(self.views):
            c = self.cache[v]
            img_idx = c["sil"]
            if img_idx.size == 0:
                continue
            rows, cols = view["sz"]
            tot_px = rows * cols
            occ, occ_h, img_idx_h = c["occ"], c["occ_hat"], c["sil_hat"]

            # curvature (uses the *current* phi: views update it one after another)
            # MATLAB calls mean_curvature(-phi, ...); the numerator is odd in phi, so that is -H(phi)
            curvature = -ls.mean_curvature(self.phi, self.in_pt[occ])
            z_i = c["depth"]
            _, ibi, ioi = np.intersect1d(band, self.in_pt[occ], return_indices=True)
            _, ibi_h, ioi_h = np.intersect1d(band_hat, self.in_pt_hat[occ_h], return_indices=True)

            u = view["umask"]
            Hp = (u > 0).astype(float)
            Hn = (u < 0).astype(float)
            mask = c["mask"].astype(float)
            mask_hat = c["mask_hat"].astype(float)
            Ea = mask - mask_hat
            eu = np.sign(mask_hat - (Hn - Hp))

            if self.gt is not None:
                gt = self.gt[v]
                err = gt - mask
                fp += int((err < 0).sum()); fn += int((err > 0).sum())
                tp += int(((mask * gt) == 1).sum()); tn += int((((1 - mask) * (1 - gt)) == 1).sum())
                rms += np.sqrt((err ** 2).sum())
            l1 = 0.5 * (Ea ** 2).sum() / tot_px
            l2 = 0.5 * (eu ** 2).sum() / tot_px
            lya += l1 + l2; lya1 += l1; lya2 += l2

            # image force (Chan-Vese style, per channel), flipped, depth-weighted
            img = self.images[v]
            if img.ndim == 3:
                t = sum(-1.0 * (in_m[k] - out_m[k]) *
                        (2.0 * img[:, :, k].ravel(order="F")[img_idx].astype(float) - (in_m[k] + out_m[k]))
                        for k in range(3))
            else:
                t = -1.0 * (in_m[0] - out_m[0]) * (2.0 * img.ravel(order="F")[img_idx].astype(float)
                                                   - (in_m[0] + out_m[0]))
            t = -t
            grad_i = t / (z_i ** 3)
            grad_max = np.max(np.abs(grad_i + 1e-5))
            F = -grad_max * Ea.ravel(order="F")[img_idx]
            grad = grad_i + F
            ng = grad / np.max(np.abs(grad + 1e-5))
            nc = -cfg.curvature_weight * curvature / np.max(np.abs(curvature + 1e-5))
            dphi = ng + nc
            ndphi = 0.5 * dphi / np.max(np.abs(dphi + 1e-5))
            pf[band[ibi]] += ndphi[ioi]

            # phi_hat: driven by the error and the user strokes
            ea_h = Ea.ravel(order="F")[img_idx_h]
            fu_h = u.ravel(order="F")[img_idx_h]
            if ea_h.size:
                d_hat = ea_h / np.max(np.abs(ea_h + 1e-5)) + fu_h / np.max(np.abs(fu_h + 1e-5))
                n_hat = 0.5 * d_hat / np.max(np.abs(d_hat + 1e-5))
                pf_hat[band_hat[ibi_h]] += n_hat[ioi_h]

        d = self.in_mean - self.out_mean
        di = self.iin_mean - self.iout_mean
        energy = -float((d ** 2).sum())
        ienergy = -float((di ** 2).sum())
        if it % cfg.sussman_every == 1:
            ls.sussman_local(self.phi, cfg.sussman_dt)        # in place, on the box around the object
            ls.sussman_local(self.phi_hat, cfg.sussman_dt)

        out = {"iter": it, "energy": energy, "ienergy": ienergy, "rms": rms, "lyapunov": lya,
               "lyap1": lya1, "lyap2": lya2, "tp": tp, "tn": tn, "fp": fp, "fn": fn}
        if self.gt is None:                                   # no ground truth: these numbers do not exist
            for k in ("rms", "tp", "tn", "fp", "fn"):
                out[k] = None
        for k in self.tracks:
            self.tracks[k].append(out[k])
        return out

    # ------------------------------------------------------------------ views for the UI
    def outline(self, v: int, hat: bool = False):
        """Flat (Fortran-order) pixel indices of the 1-px boundary of view v's projected mask."""
        m = self.cache[v]["mask_hat" if hat else "mask"]
        edge = m & ~sil.imerode(m, np.ones((3, 3), bool))
        return np.flatnonzero(edge.ravel(order="F")).astype(np.int32)

    def outline_segments(self, v: int, hat: bool = False, sigma: float = 1.5):
        """Smooth sub-pixel outline of view v's projected mask, as (n, 4) line segments."""
        return sil.smooth_outline(self.cache[v]["mask_hat" if hat else "mask"], sigma)

    def surface_mesh(self, hat: bool = False):
        """Triangle mesh of the zero level set, in the same normalised frame as surface_points."""
        phi = self.phi_hat if hat else self.phi
        return surface_mesh(phi, origin=self.m, scale=2.0 / self.cfg.grid)

    def export_mesh(self, hat: bool = False):
        """The zero level set as (vertices, normals, triangles) in world units: the units of the camera
        positions (camera distance 160), centred on the object, z up. Full resolution, as in the viewer."""
        phi = self.phi_hat if hat else self.phi
        return surface_mesh(phi, origin=self.m, scale=self.voxel_scale)

    def surface_points(self, hat: bool = False, max_points: int = 3000, seed: int = 0):
        """Voxels on the zero level set, centred and scaled to roughly [-1, 1]."""
        phi = self.phi_hat if hat else self.phi
        idx = np.flatnonzero((phi.ravel(order="F") >= 0) & (phi.ravel(order="F") < 1.0))
        if idx.size > max_points:
            idx = np.random.default_rng(seed).choice(idx, max_points, replace=False)
        I, J, K = np.unravel_index(idx, self.shape, order="F")
        pts = np.stack([I - self.m[0], J - self.m[1], K - self.m[2]], axis=1) / (self.cfg.grid / 2.0)
        return pts.astype(np.float32)

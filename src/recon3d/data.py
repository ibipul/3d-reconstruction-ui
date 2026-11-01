"""Local dataset loading (needs Pillow; not used in the browser, where JS decodes the PNGs)."""
from __future__ import annotations

import numpy as np

from .silhouette import imclose
from .solver import imnoise_salt_pepper

# the 6x6 structuring element from the loader's ground-truth closing
GT_CLOSE = np.array([[0, 0, 1, 1, 0, 0],
                     [0, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1],
                     [0, 1, 1, 1, 1, 0],
                     [0, 0, 1, 1, 0, 0]], dtype=bool)


def gt_mask_from_gray(gray):
    """`im_gt(im_gt>0)=1; imclose(im_gt, SE)`: closed binary mask as float."""
    return imclose(gray > 0, GT_CLOSE).astype(float)


def load_gray(path):
    """MATLAB imread of a gray(+alpha) PNG returns just the gray plane."""
    from PIL import Image  # only needed for local loading, not in the browser

    im = Image.open(path)
    a = np.asarray(im)
    return a[:, :, 0] if a.ndim == 3 else a


def load_trace_dataset(gt_root, n=36, noise=0.9, seed=0):
    """What loader_wt_control_rec_error_trace.m does: the observed image is the ground-truth
    silhouette with salt & pepper noise; the closed GT mask is kept for the metrics."""
    rng = np.random.default_rng(seed)
    images, masks = [], []
    for i in range(1, n + 1):
        gray = load_gray(f"{gt_root}/{i}.png")
        masks.append(gt_mask_from_gray(gray))
        images.append(imnoise_salt_pepper(gray, noise, rng) if noise > 0 else gray)
    return images, masks

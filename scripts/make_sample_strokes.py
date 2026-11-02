"""Generate sample stroke sets for the sentinel dataset (dataset25 photos / dataset_sentinel_gt).

The strokes imitate what a careful user would draw, using the ground-truth silhouettes only to
decide where:
  - inside strokes follow the object's long axes, kept at least 4 px away from its edge;
  - outside strokes run across clear background, at least 14 px away from the object.

In these sample files `iter` is an offset: 0 means "at the moment the sample is plugged in".

  python scripts/make_sample_strokes.py [gt_dir] [out_dir]
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from scipy import ndimage

from recon3d.data import gt_mask_from_gray, load_gray

HERE = os.path.dirname(os.path.abspath(__file__))
GT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "web", "app", "data", "gt")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "web", "app", "data", "strokes")
N, SIZE = 36, 6
EDGE_IN, EDGE_OUT, STEP = 4, 14, 7


def runs(flags):
    """Contiguous True runs of a 1D boolean array as (start, stop)."""
    d = np.diff(np.concatenate([[0], flags.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def polyline(points):
    return [[round(float(x), 1), round(float(y), 1)] for x, y in points]


def inside_strokes(mask, din):
    out = []
    lab, n = ndimage.label(mask)
    comps = sorted(range(1, n + 1), key=lambda k: -(lab == k).sum())[:2]
    for k in comps:
        ys, xs = np.nonzero(lab == k)
        if len(xs) < 300:
            continue
        c = np.array([xs.mean(), ys.mean()])
        _, vecs = np.linalg.eigh(np.cov(np.stack([xs, ys])))
        for axis in (vecs[:, 1], vecs[:, 0]):               # major axis, then minor axis
            t = np.arange(-200, 201)
            p = c[None, :] + t[:, None] * axis[None, :]
            ix, iy = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
            ok = (ix >= 0) & (ix < mask.shape[1]) & (iy >= 0) & (iy < mask.shape[0])
            good = np.zeros(len(t), bool)
            good[ok] = (lab[iy[ok], ix[ok]] == k) & (din[iy[ok], ix[ok]] >= EDGE_IN)
            best = max(runs(good), key=lambda r: r[1] - r[0], default=None)
            if best and best[1] - best[0] >= 25:
                out.append(polyline(p[best[0]:best[1]:STEP]))
    return out


def outside_strokes(mask, dout):
    out = []
    h, w = mask.shape
    clear = dout >= EDGE_OUT
    clear[:6], clear[-6:], clear[:, :6], clear[:, -6:] = False, False, False, False
    for fy in (0.12, 0.5, 0.88):                            # horizontal scan lines
        y = int(fy * h)
        best = max(runs(clear[y]), key=lambda r: r[1] - r[0], default=None)
        if best and best[1] - best[0] >= 30:
            xs = np.arange(best[0] + 4, best[1] - 4, STEP)
            out.append(polyline([(x, y) for x in xs]))
    for fx in (0.1, 0.9):                                   # vertical scan lines
        x = int(fx * w)
        best = max(runs(clear[:, x]), key=lambda r: r[1] - r[0], default=None)
        if best and best[1] - best[0] >= 30:
            ys = np.arange(best[0] + 4, best[1] - 4, STEP)
            out.append(polyline([(x, y) for y in ys]))
    return out


def session(title, description, strokes):
    return {"format": "recon3d-strokes", "version": 1, "image": {"rows": 320, "cols": 200},
            "dataset": "sentinel (dataset25 photos, dataset_sentinel_gt)", "title": title, "description": description,
            "note": "iter is an offset from the moment the sample is plugged in. view is 0-based; pts are [x, y] in image pixels; "
                    "type in = inside (+), out = outside (-)",
            "strokes": strokes}


ins, outs = [], []
for i in range(N):
    mask = gt_mask_from_gray(load_gray(os.path.join(GT, "%d.png" % (i + 1)))) > 0
    din, dout = ndimage.distance_transform_edt(mask), ndimage.distance_transform_edt(~mask)
    for pts in inside_strokes(mask, din):
        ins.append({"view": i, "type": "in", "size": SIZE, "iter": 0, "pts": pts})
    for pts in outside_strokes(mask, dout):
        outs.append({"view": i, "type": "out", "size": SIZE, "iter": 0, "pts": pts})

staged = [dict(s, iter=0) for s in outs] + [dict(s, iter=10) for s in ins]
sets = {
    "sentinel-outside": session("Background markers", "Outside strokes across clear background in every view, applied immediately.", outs),
    "sentinel-inside": session("Object core markers", "Inside strokes along the long axes of the object in every view, applied immediately.", ins),
    "sentinel-both": session("Inside and outside markers", "Both sets, applied immediately.", outs + ins),
    "sentinel-staged": session("Staged guidance", "Outside strokes first, then the inside strokes 10 iterations later.", staged),
}
os.makedirs(OUT, exist_ok=True)
manifest = {"dataset": "sentinel", "note": "Plug these into the web app's Strokes panel. iter in each file is an offset.",
            "samples": []}
for key, sess in sets.items():
    path = os.path.join(OUT, key + ".json")
    with open(path, "w") as f:
        json.dump(sess, f, separators=(",", ":"))
    manifest["samples"].append({"file": key + ".json", "title": sess["title"], "description": sess["description"],
                                "strokes": len(sess["strokes"]), "bytes": os.path.getsize(path)})
    print("%-18s %3d strokes, %6d bytes" % (key, len(sess["strokes"]), os.path.getsize(path)))
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)

# Web app (live solver)

The real reconstruction loop from `loader_wt_control_rec_error_trace.m`, running in the browser.
The Python package `recon3d` runs in [Pyodide](https://pyodide.org) inside a Web Worker; this
page is the UI. The static layout mock lives in `../mock/`.

## Run it

```sh
# once: copy the MATLAB repo's datasets into web/app/data/ (git-ignored)
scripts/fetch_data.sh ~/codes/3d-reconstruction-ui-matlab

# from the repo root (so the page can fetch src/recon3d/*.py)
python3 -m http.server 8000
```

Open http://localhost:8000/web/app/. It must be served over HTTP: as a `file://` page the browser
blocks the worker and image decoding. The first load downloads Pyodide, NumPy and SciPy from the
jsDelivr CDN (about 30 MB, cached afterwards), so it needs internet access.

## What you see

- **Grid:** the 36 views. *Show* switches between the photo, what the solver sees (ground-truth
  silhouette plus salt & pepper noise), and the ground truth. Yellow outlines are the projected
  reconstruction φ, cyan the estimate φ̂.
- **Click a card** to focus it. Pick **Inside** or **Outside** and draw. Strokes are sent to the
  solver and steer φ̂ from the next iteration (positive = inside, negative = outside, as in
  `erase_from.m` / `interact_to.m`).
- **Sidebar:** the zero level set of φ as a lit surface (drag to orbit, scroll to zoom, **Enlarge** for a wider
  panel; optional φ̂ wireframe), a compact ring marking the camera angles, and traces (energy, IoU against
  the ground truth, FP+FN, RMS error, Lyapunov). Outlines on the cards are smoothed sub-pixel contours.

## Settings

| Setting | Default | Notes |
|---|---|---|
| Grid | 100³ | The MATLAB code uses 150³. 64³ is fast; 200³ and 300³ are slower (about 1 s and 3 s per iteration natively; the browser is a bit slower) and need roughly 0.6 GB and 1 GB. The camera geometry is the same at every size (voxels are rescaled). |
| Smoothing | 0.001 | Weight of the mean-curvature term (the MATLAB loader's value). Values around 0.1 to 1 smooth more; above about 3 the shape collapses. Applies from the next iteration, no re-initialise needed. |
| Noise | 0.90 | `imnoise(gt, 'salt & pepper', 0.90)`, as in the loader. At 0 the carving converges visibly within tens of iterations. |
| Seed | 0 | Noise random seed. |

Change a setting, then press **Re-initialise** (strokes you drew are kept and re-sent).

## Not ported

The MATLAB figure windows, per-iteration PNG/`.mat` saving, the commented-out control injection
(`uinput_bank` load) and `diffuse2d` smoothing of strokes (`smask`, unused by this loader).

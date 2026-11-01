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
- **Sidebar:** live point cloud of φ and φ̂ with the camera ring, and traces (energy, IoU against
  the ground truth, FP+FN, RMS error, Lyapunov).

## Settings

| Setting | Default | Notes |
|---|---|---|
| Grid | 100³ | The MATLAB code uses 150³. Smaller grids keep the same camera geometry (voxels are rescaled) and run faster. |
| Noise | 0.90 | `imnoise(gt, 'salt & pepper', 0.90)`, as in the loader. At 0 the carving converges visibly within tens of iterations. |
| Seed | 0 | Noise random seed. |

Change a setting, then press **Re-initialise** (strokes you drew are kept and re-sent).

## Not ported

The MATLAB figure windows, per-iteration PNG/`.mat` saving, the commented-out control injection
(`uinput_bank` load) and `diffuse2d` smoothing of strokes (`smask`, unused by this loader).

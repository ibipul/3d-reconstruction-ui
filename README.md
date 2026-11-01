# Interactive 3D Reconstruction

Python port of [ibipul/3d-reconstruction-ui-matlab](https://github.com/ibipul/3d-reconstruction-ui-matlab):
silhouette-based 3D reconstruction by level-set evolution, with user strokes (inside / outside) that
steer the result. The solver is a port of `loader_wt_control_rec_error_trace.m`.

## Layout
- `src/recon3d/` – the solver (pure NumPy/SciPy; no UI, no file I/O except `data.py`)
  - `camera.py` cameras and projection · `levelset.py` Sussman re-initialisation, normals, curvature
  - `silhouette.py` masks, visibility, image statistics · `strokes.py` user strokes
  - `solver.py` the iteration loop · `surface.py` level-set mesh · `export.py` strokes to masks
  - `bridge.py` bytes/JSON glue for the web worker
- `web/app/` – the live browser app (Pyodide in a Web Worker). See [web/app/README.md](web/app/README.md)
- `web/mock/` – the static layout mock
- `tests/` – unit tests · `scripts/` – `fetch_data.sh`, `smoke.py`, `export_masks.py`, `smoothing_sweep.py` · `docs/` – porting notes

## Live app
https://ibipul.github.io/3d-reconstruction-ui/ (GitHub Pages; the first load downloads Pyodide, NumPy and SciPy, about 30 MB)

## Quick start

```sh
python3 -m venv .venv && .venv/bin/pip install numpy scipy pytest pillow
PYTHONPATH=src .venv/bin/python -m pytest -q tests

# the web app (the two datasets are committed in web/app/data/; scripts/fetch_data.sh refreshes them)
python3 -m http.server 8000      # then open http://localhost:8000/web/app/

# or run the solver natively and print metrics
PYTHONPATH=src .venv/bin/python scripts/smoke.py 100 30 0.0   # grid, iterations, noise
```

## Status
- Ported: cameras, level-set machinery, silhouette extraction and visibility, the update rule,
  metrics (energy, RMS error, TP/TN/FP/FN, Lyapunov terms), user strokes.
- **Not yet verified against MATLAB.** The tests check invariants (e.g. the fast morphology equals
  scipy's, shift and projection conventions, sphere curvature). Parity with MATLAB output needs
  fixtures from a MATLAB run, and one known approximation: `strel('disk', 6)` is modelled as an
  octagon (R=3 and R=4 match MATLAB's neighbourhoods; R=6 is unconfirmed).
- The grid size is configurable (default 100³; MATLAB uses 150³), so results differ from MATLAB's
  unless run at 150.

## Docs
- [MATLAB codebase map](docs/matlab-architecture.md) – how the original scripts and functions fit together

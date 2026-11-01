# Interactive 3D Reconstruction

Python port of [ibipul/3d-reconstruction-ui-matlab](https://github.com/ibipul/3d-reconstruction-ui-matlab).

## Layout
- `src/recon3d/` – core numerics (pure NumPy/SciPy; no UI/file I/O)
- `tests/` – tests, validated against saved MATLAB outputs (`tests/fixtures/`)
- `web/` – browser UI (Pyodide)
- `docs/` – porting notes

## Port plan
1. Numerics: sussman, shift ops, curvature, normals, projection
2. Parity tests vs MATLAB outputs (dataset25, dataset_sentinel_gt)
3. Local GUI (matplotlib)
4. Web UI via Pyodide

## Docs
- [MATLAB codebase map](docs/matlab-architecture.md) – how the original scripts and functions fit together

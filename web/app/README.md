# Web app (live solver)

The real reconstruction loop from `loader_wt_control_rec_error_trace.m`, running in the browser.
The Python package `recon3d` runs in [Pyodide](https://pyodide.org) inside a Web Worker; this
page is the UI. The static layout mock lives in `../mock/`.

## Run it

```sh
# the datasets are committed in web/app/data/; to refresh them from the MATLAB repo:
# scripts/fetch_data.sh ~/codes/3d-reconstruction-ui-matlab

# from the repo root (so the page can fetch src/recon3d/*.py)
python3 -m http.server 8000
```

Open http://localhost:8000/web/app/. It must be served over HTTP: as a `file://` page the browser
blocks the worker and image decoding. The first load downloads Pyodide, NumPy and SciPy from the
jsDelivr CDN (about 30 MB, cached afterwards), so it needs internet access.

## Update rule (CFL)

How the forces of the 36 views move the surface is the **Update** setting (change it, then *Re-initialise*).

φ is kept close to a distance function, so one iteration may move it by at most half a voxel (the CFL bound) or the
explicit update becomes unstable.

- **CFL-limited (summed)**, the default: each view's force is added up per voxel, and the total is limited as a whole. The
  total force at the *Force percentile* of the moving voxels is scaled to the full half-voxel step, and anything stronger is
  clipped to it. So the bound holds exactly, and a typical silhouette voxel uses most of the step. A lower percentile moves
  more voxels at the full step; 100 scales by the strongest voxel. It can be changed while the run goes.
- **MATLAB (per view)**: each view is scaled to a half-voxel step on its own and the results are added. A voxel seen by
  many views can then move several voxels in one iteration (measured: up to 3.9 on sentinel_polar, 2,000+ voxels above
  0.5 in the first iteration), and because the scale is set by one outlier pixel per view, most voxels move only
  hundredths of a voxel (median 0.01 for Bhattacharyya).

On sentinel_polar (RGB photos, grid 100, 200 iterations) the best IoU against the ground truth was 0.40 for Bhattacharyya
under the MATLAB rule and 0.69 under the CFL rule (percentile 25), and for Chan-Vese 0.83 and 0.85. The shipped replay was recorded
under the MATLAB rule, so Play switches to it.

## Energy

The **Energy** setting chooses the region energy the surface descends (change it, then *Re-initialise*):

- **Bhattacharyya** (default): the energy is ln B, with B = Σ √(p_in · p_out) over the smoothed colour histograms of the
  pixels inside and outside the surface (32 bins per channel, pooled over all views; with RGB the channels are summed).
  B is 1 when the two histograms are equal and falls toward 0 as they separate, so the energy goes down as the inside and
  outside look more different. The force on each boundary pixel is the gradient of B (Michailovich, Rathi and
  Tannenbaum 2007): a pixel is pushed in if its colour is common outside and rare inside, and out in the opposite case.
  One departure from the exact gradient: it weighs inside and outside equally instead of by 1/area each. With the area
  weights the force just outside the true edge is much stronger than just inside it (the inside is the smaller region),
  and the surface drifts inward. The forces are also scaled to the range of the Chan-Vese ones, because the MATLAB
  normalisations carry fixed stabilising constants.
- **Chan-Vese**: the MATLAB code's energy, which compares only the mean colour inside and outside.

The *Energy* trace shows ln B (at most 0) for Bhattacharyya and minus the squared difference of the means for Chan-Vese,
so the two are not comparable with each other.

## Datasets

The **Dataset** picker (or `?dataset=zebra` in the URL) switches between the photo sets listed in `data/datasets.json`:

| Dataset | Photos | Ground truth |
|---|---|---|
| `sentinel_polar` | 36 synthetic views, 200×320 | yes: IoU, FP/FN and RMS error are shown |
| `zebra` | 36 real photos, 320×216 | no: the *Silhouettes + noise* input, the *Ground truth* view and the IoU / FP+FN / RMS traces are disabled; energy and Lyapunov remain |

Every dataset starts from the same camera ring as the MATLAB code (level cameras, 10° steps). These settings are free
parameters, taken from `datasets.json` when you pick a dataset and editable before you press Run:

- **Distance**: camera distance in world units (160 for sentinel_polar).
- **Focal**: focal length in pixels of the (shrunk) images (520 at 320 px high for sentinel_polar).
- **Camera height**: the cameras raised above the ring plane, in world units (0 = level; positive = looking down on the object).
- **Initial box** width / depth / height: the size of the box the surface starts as, in voxels at grid 150 (scaled with the
  grid; the MATLAB box is 20 × 60 × 80), and **angle**, which turns the box about the vertical axis (it matters because
  width and depth differ).

- **Box shift** x / y / up: moves the box, with the whole volume, away from the point the cameras look at, in world units.
  In an opened view, the **Move box** tool (key `M`) does it by dragging: the box follows the pointer in that view's image
  plane (a drag to the right in view 36 moves it right in view 36). Strokes only act where the outline passes through
  them, so put the box over the part you want to grow before you draw.

Each dataset has its own values for all of these in `data/datasets.json` (`distance`, `focal`, `cameraHeight`, `box`,
`boxAngle`, `boxOffset`), applied when you pick it. To set them for a dataset, tune in the page, press **Copy settings**
(it copies the JSON entry for the dataset), and paste the values into `datasets.json`. **Reset to defaults** goes back to
the listed values.

Every change redraws the starting surface straight away: as a bold outline on all the views (and the open view) and as the
box in the 3D panel, using a quick coarse-grid preview. The real run is re-initialised once you stop typing for about a
second and a half. Tune until the outline sits just around the object in every view, then Run. Interaction replays
ship with their dataset (`data/<dataset>/strokes/` and its `manifest.json`; zebra has none, so its replay controls are hidden); a strokes file drawn on one dataset is refused on another.
To add a dataset, put `photos/1..N.<ext>` (and `gt/1..N.png` if it has ground truth) in `data/<name>/` and add an entry to `datasets.json`.

## What you see

- **Grid:** the 36 views. *Show* switches between the photo, what the solver sees (the *Input* setting:
  silhouettes plus noise, or the photos), and the ground truth. Yellow outlines are the projected
  reconstruction φ, cyan the estimate φ̂.
- **Click a card** to focus it. Pick **Inside** or **Outside** and draw. Strokes are sent to the
  solver and steer φ̂ from the next iteration (positive = inside, negative = outside, as in
  `erase_from.m` / `interact_to.m`).
- **Sidebar:** the zero level set of φ as a lit surface (shown upright; drag to orbit, scroll to zoom, **Enlarge** for a wider
  panel; optional φ̂ wireframe), a compact ring marking the camera angles, and traces (energy, IoU against
  the ground truth, FP+FN, RMS error, Lyapunov). Outlines on the cards are smoothed sub-pixel contours.

## Saving and replaying strokes

- **Save** downloads `strokes-<time>.json` with every stroke on every view: its type (inside or outside), brush size,
  points, and the iteration it was drawn at, plus the settings of the run (grid, noise, seed, smoothing).
- **Load** puts a saved file's strokes back on the views (and its settings into the controls).
- **Replay** re-initialises with those settings, then re-applies each stroke at the iteration it was originally drawn, so
  you reproduce an interactive session exactly. The counter next to the buttons shows progress. (The MATLAB loader's
  control injection loaded a saved bank once, at iteration 2; this is the timed version of that.)
- **Masks for other tools:** `python scripts/export_masks.py strokes.json masks.npz` writes the per-view control masks
  (36 x 320 x 200, +6 inside, -2 outside), and `... uinput_bank.mat` writes a MATLAB `uinput_bank` struct array with the
  fields of `get_uis.m`, ready for the loaders' injection block.

## Download the 3D model

Under the viewport, **Model** saves the current zero level set as a triangle mesh: pick the shape (the φ reconstruction or the φ̂
estimate) and the format, then **Download**. OBJ (ASCII, with vertex normals), STL (binary) and PLY (binary, with vertex normals)
open in Blender, MeshLab, CAD tools and most 3D viewers. The mesh is the full-resolution surface (what you see in the viewport),
with the triangle winding outward. Units are the world units of the camera setup (camera distance 160), centred on the object
with z up, so a downloaded model lines up with the camera positions; scale it as you like. The file name carries the grid size and
the iteration, e.g. `recon-100grid-120it-phi.obj`. The same files can be written from Python with `recon3d.meshio`
(`Solver.export_mesh()` gives the arrays).

## Interaction replay

`data/sentinel_polar/strokes/` holds the interaction replay recorded for this dataset (`manifest.json` lists it, and it is preselected in the
**Strokes** row): a hand-drawn session on `dataset25` (the sentinel), 290 strokes (95 inside, 195 outside) on all 36 views, 177 of
them at the start and 113 more between iterations 152 and 1099. It was recorded with the **RGB photos** as input, smoothing 0.398
and the loader's re-initialisation (every 5, step 0.5), at grid 300.

- **Play** re-initialises with the settings the replay was recorded with (input, smoothing, re-init schedule; **your grid is kept**,
  since the recorded 300³ is heavy), then plays it from the start: the opening strokes appear at once and the others at the
  iterations they were drawn at. Press nothing else; it starts running. The full timeline is 1099 iterations (about 7 minutes at 100³).
- **Plug in** adds the replay to the run you already have, without touching your settings. Type an iteration in **at** to plug it in
  later (empty = now). Tick **all at once** to apply every stroke immediately instead of at its recorded iteration.
- **Clear all** removes every stroke. **Re-run** (next to Save and Load) re-initialises and re-applies *your own* strokes at the
  iterations you drew them at. Plugged and played strokes are saved and re-run like hand-drawn ones.

To add another replay, save a session from the app (**Save**), drop the file into the dataset's `strokes/` folder and list it in `manifest.json`.

Strokes are in image pixels, so a replay recorded at one grid size plugs into a run at any other. The timing does not carry over
exactly, though: a coarser grid evolves faster per iteration, and not by a fixed ratio (in a test, a 3x coarser run matched the fine
run's state for the first few iterations and then raced ahead), so there is no automatic rescale. Use the recorded timing, or **all at once**.

What it does, under the settings it was recorded with (RGB photos, smoothing 0.398, every 5 / step 0.5), at 64³ over 1100 iterations
(voxels left and IoU against the ground truth; higher IoU is better):

| Iteration | No strokes | Plugged as recorded | All at once |
|---|---|---|---|
| 50 | 111 voxels, 0.208 | 412 voxels, 0.736 | 395 voxels, 0.726 |
| 150 | 9 voxels, 0.013 | 427 voxels, 0.726 | 414 voxels, 0.722 |
| 600 | 6 voxels, 0.009 | 423 voxels, 0.722 | 403 voxels, 0.752 |
| 1100 | 6 voxels, 0.009 | 407 voxels, 0.731 | 396 voxels, 0.726 |

With these settings the shape collapses without the strokes (the loader's re-initialisation schedule shrinks it away), and the strokes
hold it at a steady shape. I have not measured this replay under a gentler schedule. With an earlier recorded session, on clean data with every 10 / step 0.05,
the run did not collapse on its own and the strokes changed little (IoU 0.806 to 0.809).

## Settings

| Setting | Starts at | Notes |
|---|---|---|
| Input | RGB photos | What the solver sees. *RGB photos* runs on the colour images, using the three-channel branch of the MATLAB loaders: the image force and the energy are summed over R, G and B, and the inside/outside means are taken per channel (zero pixels are ignored, as in `get_image_stat.m`). *silhouettes + noise* is what `loader_wt_control_rec_error_trace.m` does (the ground-truth masks with salt & pepper noise). Noise, if set, is applied to each channel independently. The ground-truth masks are always used for IoU and the error metrics. |
| Grid | 150³ | The MATLAB size. 64³ is fast; 200³ and 300³ are slower (about 1 s and 3 s per iteration natively; the browser is a bit slower) and need roughly 0.6 GB and 1 GB. The camera geometry is the same at every size (voxels are rescaled). |
| Noise | 0 | `imnoise(gt, 'salt & pepper', d)` as in the loader, which uses 0.90. At 0 the carving converges visibly within tens of iterations. |
| Seed | 0 | Noise random seed. |
| Smoothing | 0.04 | Weight of the mean-curvature term. The loader uses 0.001, which is very weak. In a clean 64³ run of 60 iterations (every 10 / step 0.05), raising it from 0.001 to 1 made the surface about a quarter smoother (roughness 6.1 to 4.7) and raised IoU from 0.814 to 0.853; at 3 it over-smoothed (IoU 0.75), and much higher values collapse the shape. Applies from the next iteration, no re-initialise needed. |
| Re-init | every 5, step 0.25 | How often the level set is re-initialised (Sussman) and how big each correction is. The loader's schedule (every 5, step 0.5) makes the shape pulse every 5 iterations, because each re-initialisation moves the surface; a smaller step (0.05 to 0.1) and a longer interval (10 to 20) remove the pulsing. Applies from the next iteration. |
| Show | Solver input | What the cards display: the photo, what the solver sees, or the ground truth. |
| all at once | ticked | Next to **Plug in**: apply every stroke of the interaction replay immediately, instead of at the iteration it was recorded at. **Play** always uses the recorded timing and the recorded settings. |

Change a setting, then press **Re-initialise** (strokes you drew are kept and re-sent).

## Not ported

The MATLAB figure windows, per-iteration PNG/`.mat` saving, the commented-out control injection
(`uinput_bank` load) and `diffuse2d` smoothing of strokes (`smask`, unused by this loader).

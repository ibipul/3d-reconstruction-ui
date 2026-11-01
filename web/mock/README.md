# UI mock

Static layout mock (no real computation). It expects `img/1.png` ... `img/36.png`, copied from the MATLAB repo's `dataset25/`, next to `index.html`.

Open it over HTTP, not by double-clicking: `cd web/mock && python3 -m http.server`, then visit http://localhost:8000. As a `file://` page the browser blocks reading the image pixels, so the silhouette overlays are switched off (the rest still works).

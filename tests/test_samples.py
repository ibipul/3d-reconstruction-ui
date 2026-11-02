"""The sample stroke sets shipped in web/app/data/strokes/ are valid for the dataset they are listed for."""
import json
import os

import numpy as np
import pytest

from recon3d import export

STROKES = os.path.join(os.path.dirname(__file__), "..", "web", "app", "data", "strokes")
MANIFEST = os.path.join(STROKES, "manifest.json")

pytestmark = pytest.mark.skipif(not os.path.exists(MANIFEST), reason="no sample strokes shipped")


def _samples():
    return json.load(open(MANIFEST))["samples"]


def test_manifest_matches_the_files():
    samples = _samples()
    assert samples and sum(1 for s in samples if s.get("default")) == 1     # one preselected sample
    for s in samples:
        sess = export.load_session(os.path.join(STROKES, s["file"]))
        assert len(sess["strokes"]) == s["strokes"] > 0
        assert s["title"] and s["description"]


def test_every_stroke_is_well_formed_and_on_the_image():
    for s in _samples():
        sess = export.load_session(os.path.join(STROKES, s["file"]))
        rows, cols = sess["image"]["rows"], sess["image"]["cols"]
        assert (rows, cols) == (320, 200)
        for st in sess["strokes"]:
            assert 0 <= st["view"] < 36 and st["type"] in ("in", "out")
            assert isinstance(st["iter"], int) and st["iter"] >= 0 and st["size"] > 0
            pts = np.asarray(st["pts"], float)
            assert pts.ndim == 2 and pts.shape[1] == 2 and pts.shape[0] >= 1 and np.isfinite(pts).all()
            # a hand-drawn stroke may run a little past the edge (it is clamped when rasterised), not far
            assert pts[:, 0].min() > -15 and pts[:, 0].max() < cols + 15
            assert pts[:, 1].min() > -15 and pts[:, 1].max() < rows + 15


def test_samples_rasterise_to_control_masks():
    for s in _samples():
        umask, per_view = export.masks_from_session(export.load_session(os.path.join(STROKES, s["file"])))
        assert umask.shape == (36, 320, 200) and np.isfinite(umask).all()
        assert umask.max() > 0 and umask.min() < 0          # both inside (+) and outside (-) marks
        n_inside = sum(1 for p in per_view for st in p if st["type"] == "in")
        n_outside = sum(1 for p in per_view for st in p if st["type"] == "out")
        assert n_inside > 0 and n_outside > 0 and n_inside + n_outside == s["strokes"]

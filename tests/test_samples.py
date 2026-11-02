"""The sample stroke sets shipped in web/app/data/strokes/ must match the dataset they claim to be for."""
import json
import os

import numpy as np
import pytest

from recon3d import export
from recon3d.data import gt_mask_from_gray, load_gray

APP = os.path.join(os.path.dirname(__file__), "..", "web", "app", "data")
MANIFEST = os.path.join(APP, "strokes", "manifest.json")

pytestmark = pytest.mark.skipif(not os.path.exists(MANIFEST), reason="sample strokes not generated")


def _manifest():
    return json.load(open(MANIFEST))


def test_manifest_lists_existing_files_with_correct_counts():
    m = _manifest()
    assert len(m["samples"]) >= 3
    for s in m["samples"]:
        path = os.path.join(APP, "strokes", s["file"])
        sess = export.load_session(path)
        assert len(sess["strokes"]) == s["strokes"] > 0
        assert s["title"] and s["description"]


def test_inside_strokes_lie_on_the_object_and_outside_strokes_on_background():
    masks = [gt_mask_from_gray(load_gray(os.path.join(APP, "gt", "%d.png" % (i + 1)))) > 0 for i in range(36)]
    for s in _manifest()["samples"]:
        sess = export.load_session(os.path.join(APP, "strokes", s["file"]))
        for st in sess["strokes"]:
            m = masks[st["view"]]
            xs = np.round([p[0] for p in st["pts"]]).astype(int)
            ys = np.round([p[1] for p in st["pts"]]).astype(int)
            assert (xs >= 0).all() and (xs < m.shape[1]).all() and (ys >= 0).all() and (ys < m.shape[0]).all()
            on_object = m[ys, xs]
            assert on_object.all() if st["type"] == "in" else (~on_object).all(), (s["file"], st["view"], st["type"])


def test_staged_sample_is_staged_and_others_start_immediately():
    by = {s["file"]: export.load_session(os.path.join(APP, "strokes", s["file"])) for s in _manifest()["samples"]}
    staged = by["sentinel-staged.json"]["strokes"]
    assert {st["iter"] for st in staged if st["type"] == "out"} == {0}
    assert {st["iter"] for st in staged if st["type"] == "in"} == {10}
    for name in ("sentinel-outside.json", "sentinel-inside.json", "sentinel-both.json"):
        assert {st["iter"] for st in by[name]["strokes"]} == {0}


def test_samples_rasterise_to_masks_with_the_right_signs():
    sess = export.load_session(os.path.join(APP, "strokes", "sentinel-both.json"))
    umask, per_view = export.masks_from_session(sess)
    # +6 per inside stroke and -2 per outside stroke; crossing strokes add up, as repeated strokes do in MATLAB
    assert umask.max() >= 6.0 and umask.min() <= -2.0
    assert (umask[umask > 0] % 6 == 0).all() and (umask[umask < 0] % 2 == 0).all()
    assert all(len(p) > 0 for p in per_view)           # every view has at least one stroke

"""The interaction replays shipped in web/app/data/<dataset>/strokes/ are valid for the dataset they belong to."""
import json
import os

import numpy as np
import pytest

from recon3d import export

DATA = os.path.join(os.path.dirname(__file__), "..", "web", "app", "data")


def _datasets():
    return {d["id"]: d for d in json.load(open(os.path.join(DATA, "datasets.json")))["datasets"]}


def _cases():
    """(dataset id, strokes folder, sample) for every replay listed by every dataset."""
    out = []
    for ds_id, d in _datasets().items():
        folder = os.path.join(DATA, d["dir"], "strokes")
        manifest = os.path.join(folder, "manifest.json")
        if os.path.exists(manifest):
            out += [(ds_id, folder, s) for s in json.load(open(manifest))["samples"]]
    return out


CASES = _cases()
pytestmark = pytest.mark.skipif(not CASES, reason="no interaction replays shipped")


def _load(case):
    _, folder, s = case
    return export.load_session(os.path.join(folder, s["file"]))


def _ids():
    return ["%s/%s" % (c[0], c[2]["file"]) for c in CASES]


def test_every_dataset_with_replays_has_exactly_one_default():
    for ds_id in {c[0] for c in CASES}:
        samples = [c[2] for c in CASES if c[0] == ds_id]
        assert sum(1 for s in samples if s.get("default")) == 1, ds_id


@pytest.mark.parametrize("case", CASES, ids=_ids())
def test_manifest_matches_the_files(case):
    s, sess = case[2], _load(case)
    assert len(sess["strokes"]) == s["strokes"] > 0
    assert s["title"] and s["description"]
    assert sess.get("dataset", case[0]) == case[0]                   # a session knows which dataset it was drawn on


@pytest.mark.parametrize("case", CASES, ids=_ids())
def test_every_stroke_is_well_formed_and_on_the_image(case):
    d, sess = _datasets()[case[0]], _load(case)
    rows, cols = sess["image"]["rows"], sess["image"]["cols"]
    assert (rows, cols) == (d["height"], d["width"])                 # the page refuses strokes of another image size
    for st in sess["strokes"]:
        assert 0 <= st["view"] < d["views"] and st["type"] in ("in", "out")
        assert isinstance(st["iter"], int) and st["iter"] >= 0 and st["size"] > 0
        pts = np.asarray(st["pts"], float)
        assert pts.ndim == 2 and pts.shape[1] == 2 and pts.shape[0] >= 1 and np.isfinite(pts).all()
        # a hand-drawn stroke may run a little past the edge (it is clamped when rasterised), not far
        assert pts[:, 0].min() > -15 and pts[:, 0].max() < cols + 15
        assert pts[:, 1].min() > -15 and pts[:, 1].max() < rows + 15


@pytest.mark.parametrize("case", CASES, ids=_ids())
def test_samples_rasterise_to_control_masks(case):
    d = _datasets()[case[0]]
    umask, per_view = export.masks_from_session(_load(case))
    assert umask.shape == (d["views"], d["height"], d["width"]) and np.isfinite(umask).all()
    assert umask.max() > 0 and umask.min() < 0          # both inside (+) and outside (-) marks
    n_inside = sum(1 for p in per_view for st in p if st["type"] == "in")
    n_outside = sum(1 for p in per_view for st in p if st["type"] == "out")
    assert n_inside > 0 and n_outside > 0 and n_inside + n_outside == case[2]["strokes"]


@pytest.mark.parametrize("case", CASES, ids=_ids())
def test_recorded_settings_are_ones_the_app_can_apply(case):
    """Play re-applies these to the page's controls; unknown values would be silently ignored."""
    st = _load(case).get("settings", {})
    assert st.get("source", "gt") in ("gt", "photo")
    assert st.get("reinit_every", 5) in (1, 2, 5, 10, 20, 50, 10 ** 9)
    assert st.get("reinit_dt", 0.5) in (0.5, 0.25, 0.1, 0.05)
    assert 0.001 - 1e-12 <= st.get("curvature_weight", 0.001) <= 10 ** 0.5     # the Smoothing slider's range
    assert 0 <= st.get("noise", 0) <= 0.95
    assert st.get("energy", "chan_vese") in ("chan_vese", "bhattacharyya")
    assert st.get("update", "per_view") in ("cfl", "per_view")
    assert 1 <= st.get("force_percentile", 25) <= 100


def test_dataset_run_defaults_are_ones_the_page_can_apply():
    for ds_id, d in _datasets().items():
        run = d.get("run", {})
        assert run.get("energy", "bhattacharyya") in ("chan_vese", "bhattacharyya"), ds_id
        assert run.get("update", "cfl") in ("cfl", "per_view"), ds_id
        assert 1 <= run.get("forcePercentile", 25) <= 100, ds_id
        assert 0.001 - 1e-12 <= run.get("curvatureWeight", 0.04) <= 10 ** 0.5, ds_id
        assert run.get("reinitEvery", 5) in (1, 2, 5, 10, 20, 50, 10 ** 9), ds_id
        assert run.get("reinitDt", 0.25) in (0.5, 0.25, 0.1, 0.05), ds_id

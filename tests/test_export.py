import numpy as np
import pytest

from recon3d import export
from recon3d.strokes import build_umask

SESSION = {
    "format": "recon3d-strokes", "version": 1, "image": {"rows": 80, "cols": 50},
    "settings": {"grid": 40, "noise": 0.0, "seed": 0, "curvature_weight": 0.001},
    "strokes": [
        {"view": 2, "type": "in", "size": 6, "iter": 0, "pts": [[10, 30], [40, 30]]},
        {"view": 2, "type": "out", "size": 6, "iter": 5, "pts": [[10, 60], [40, 60]]},
        {"view": 7, "type": "in", "size": 6, "iter": 9, "pts": [[25, 40]]},
    ],
}


def test_masks_match_the_solver_rasterisation():
    umask, per_view = export.masks_from_session(SESSION)
    assert umask.shape == (36, 80, 50) and [len(p) for p in per_view][:8] == [0, 0, 2, 0, 0, 0, 0, 1]
    assert umask[2, 30, 20] == 6.0 and umask[2, 60, 20] == -2.0     # inside +, outside -
    assert np.allclose(umask[2], build_umask(per_view[2], 80, 50))
    assert not umask[0].any()


def test_npz_and_mat_roundtrip(tmp_path):
    export.save_masks(SESSION, str(tmp_path / "m.npz"))
    z = np.load(tmp_path / "m.npz")
    assert z["umask"].shape == (36, 80, 50) and z["strokes_per_view"].sum() == 3

    export.save_masks(SESSION, str(tmp_path / "b.mat"))
    from scipy.io import loadmat
    bank = loadmat(tmp_path / "b.mat", struct_as_record=False, squeeze_me=True)["uinput_bank"]
    assert len(bank) == 36 and bank[2].umask.shape == (80, 50)
    assert bank[2].utype == "io" and list(bank[2].umask_indices) == [0, 5]
    assert bank[2].umask[30, 20] == 6.0


def test_rejects_other_files():
    with pytest.raises(ValueError):
        export.load_session({"format": "something-else"})

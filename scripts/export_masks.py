"""Convert a strokes file saved by the web app into control masks.

  python scripts/export_masks.py strokes-2026-11-01T20-00-00.json masks.npz   # NumPy: umask (36, 320, 200)
  python scripts/export_masks.py strokes.json uinput_bank.mat                 # MATLAB: uinput_bank struct array
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from recon3d.export import load_session, save_masks

if len(sys.argv) != 3:
    sys.exit(__doc__)
session = load_session(sys.argv[1])
umask = save_masks(session, sys.argv[2])
marked = int((abs(umask).sum(axis=(1, 2)) > 0).sum())
print("%d strokes -> %s (%d of %d views have marks)" % (len(session["strokes"]), sys.argv[2], marked, umask.shape[0]))

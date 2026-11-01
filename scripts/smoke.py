import sys, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src"))
import numpy as np
from recon3d.data import load_trace_dataset
from recon3d.solver import Solver, Config

grid = int(sys.argv[1]) if len(sys.argv) > 1 else 64
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 6
noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
root = "/Users/bipulislam/codes/3d-reconstruction-ui-matlab/dataset_sentinel_gt"
imgs, gts = load_trace_dataset(root, 36, noise)
print("img", imgs[0].shape, imgs[0].dtype, "gt mean", np.mean([g.mean() for g in gts]))
t = time.time()
s = Solver(imgs, gts, Config(grid=grid))
print("init %.2fs" % (time.time() - t), "phi", s.phi.shape, "voxels", s.in_pt.size)
for k in range(iters):
    t = time.time()
    r = s.step()
    P = r["tp"] + r["fn"]
    iou = r["tp"] / max(r["tp"] + r["fp"] + r["fn"], 1)
    print("it %d  %.2fs  E=%.1f rms=%.1f tp=%d fp=%d fn=%d IoU=%.3f voxels=%d" % (
        r["iter"], time.time() - t, r["energy"], r["rms"], r["tp"], r["fp"], r["fn"], iou, (s.phi >= 0).sum()))

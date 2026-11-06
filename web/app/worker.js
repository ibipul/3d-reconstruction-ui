/* Runs the Python solver (recon3d) in Pyodide, off the UI thread.
   Messages in : boot | init | strokes | step
   Messages out: status | booted | init | step | error                              */
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/';
const FILES = ['__init__', 'camera', 'levelset', 'silhouette', 'strokes', 'surface', 'meshio', 'export', 'solver', 'data', 'bridge'];
let pyodide = null;

function status(text) { postMessage({ type: 'status', text }); }

async function boot(srcBase) {
  status('Loading Python runtime…');
  importScripts(PYODIDE + 'pyodide.js');
  pyodide = await loadPyodide({ indexURL: PYODIDE });
  status('Loading NumPy and SciPy…');
  await pyodide.loadPackage(['numpy', 'scipy']);
  status('Loading solver code…');
  pyodide.FS.mkdirTree('/home/pyodide/recon3d');
  for (const f of FILES) {
    const r = await fetch(srcBase + f + '.py', { cache: 'no-cache' });   // revalidate: never run stale solver code
    if (!r.ok) throw new Error('Could not load ' + f + '.py (' + r.status + ')');
    pyodide.FS.writeFile('/home/pyodide/recon3d/' + f + '.py', await r.text());
  }
  pyodide.runPython("import sys\nsys.path.insert(0, '/home/pyodide')\nfrom recon3d import bridge\nsolver = None");
  postMessage({ type: 'booted' });
}

function snapshot(msg) {
  const snap = pyodide.runPython('bridge.snapshot(solver)');
  const parts = snap.toJs(); snap.destroy();
  msg.outlines = parts[0].buffer; msg.mesh = parts[1].buffer; msg.meshHat = parts[2].buffer;
  return [msg.outlines, msg.mesh, msg.meshHat];
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.cmd === 'boot') {
      await boot(m.srcBase);
    } else if (m.cmd === 'init') {
      status('Building the volume…');
      const g = pyodide.globals;
      g.set('_gt', m.gt || undefined); g.set('_focal', m.focal); g.set('_distance', m.distance); g.set('_box', m.box); g.set('_n', m.n); g.set('_rows', m.rows); g.set('_cols', m.cols);
      g.set('_grid', m.grid); g.set('_noise', m.noise); g.set('_seed', m.seed);
      g.set('_photo', m.photos || undefined);   // RGB photos, or undefined to run on the silhouettes
      pyodide.runPython('solver = bridge.init_solver(_gt.to_py() if _gt is not None else None, _n, _rows, _cols, _grid, _noise, _seed, _photo.to_py() if _photo is not None else None, _focal, _distance, _box.to_py())');
      const io = pyodide.runPython('bridge.inputs_bytes(solver)'); const [inp, gts, ch] = io.toJs(); io.destroy();
      const out = { type: 'init', gen: m.gen, channels: ch, inputs: inp.buffer, gts: gts.buffer };
      const tr = snapshot(out);
      postMessage(out, [out.inputs, out.gts, ...tr]);
    } else if (m.cmd === 'strokes') {
      pyodide.globals.set('_v', m.view); pyodide.globals.set('_s', JSON.stringify(m.strokes));
      pyodide.runPython('bridge.set_strokes(solver, _v, _s)');
    } else if (m.cmd === 'export') {
      // the current shape as an OBJ / STL / PLY file; a failure here (e.g. nothing to export) must not stop the run
      try {
        pyodide.globals.set('_fmt', m.fmt); pyodide.globals.set('_hat', !!m.hat);
        const r = pyodide.runPython('bridge.export_model(solver, _fmt, _hat)'); const bytes = r.toJs(); r.destroy();
        postMessage({ type: 'export', fmt: m.fmt, hat: !!m.hat, bytes: bytes.buffer }, [bytes.buffer]);
      } catch (err) {
        postMessage({ type: 'export-error', message: String(err && err.message ? err.message : err).split('\n').slice(-2, -1)[0] || 'export failed' });
      }
    } else if (m.cmd === 'param') {
      pyodide.globals.set('_pn', m.name); pyodide.globals.set('_pv', m.value);
      pyodide.runPython('bridge.set_param(solver, _pn, _pv)');
    } else if (m.cmd === 'step') {
      pyodide.globals.set('_k', m.k);
      const t0 = performance.now();
      const res = JSON.parse(pyodide.runPython('bridge.step(solver, _k)'));
      const out = { type: 'step', gen: m.gen, metrics: res.metrics, collapsed: res.collapsed, ms: performance.now() - t0 };
      const tr = snapshot(out);
      postMessage(out, tr);
    }
  } catch (err) {
    postMessage({ type: 'error', message: String(err && err.message ? err.message : err) });
  }
};

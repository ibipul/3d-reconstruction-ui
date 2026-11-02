"""The web worker copies a fixed list of Python files into the browser. A module missing from that list works in
every test here but makes the live app fail to start, so check the list against the package."""
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_worker_loads_every_module_of_the_package():
    listed = re.search(r"const FILES = \[(.*?)\];", open(os.path.join(ROOT, "web", "app", "worker.js")).read(), re.S).group(1)
    listed = set(re.findall(r"'([^']+)'", listed))
    package = {f[:-3] for f in os.listdir(os.path.join(ROOT, "src", "recon3d")) if f.endswith(".py")}
    assert package - listed == set(), "missing from worker.js FILES: %s" % sorted(package - listed)
    assert listed - package == set(), "listed in worker.js but not in the package: %s" % sorted(listed - package)

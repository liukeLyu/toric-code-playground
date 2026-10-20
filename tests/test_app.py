"""The browser app's JavaScript model (app/toric_code_playground.html) against the Python package.

Runs the JavaScript with Node; skipped if Node is not installed.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import toric

APP = Path(__file__).parent.parent / "app" / "toric_code_playground.html"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")

DRIVER = r"""
const rng = makeRng(%(seed)d), out = [];
for (const L of [3, 4, 5, 6]) {
  const M = makeModel(L);
  for (let t = 0; t < 10; t++) {
    const s = M.fresh(), seq = [];
    for (let g = 0; g < 25; g++) {
      const G = 'XYZ'[Math.floor(rng() * 3)], e = Math.floor(rng() * M.nEdges);
      seq.push([G, e]); M.apply(s, G, e);
    }
    const an = M.anyons(s), d = M.decoded(s);
    out.push({ L, seq, k: s.k, e: an.e, m: an.m, E: M.energy(s).E, w: M.windings(s),
               da: [...d.a], db: [...d.b], dw: M.windings(d) });
  }
}
console.log(JSON.stringify(out));
"""


def run_js(seed):
    html = APP.read_text()
    model = re.search(r"// MODEL-BEGIN.*?// MODEL-END", html, re.S).group(0)
    code = model + "\n" + DRIVER % {"seed": seed}
    return json.loads(subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout)


@pytest.mark.parametrize("seed", [1, 2])
def test_app_model_matches_python(seed):
    for case in run_js(seed):
        s = toric.new_state(case["L"])
        for gate, e in case["seq"]:
            s.apply(gate, e)
        e, m = s.anyons()
        assert [list(p) for p in e] == case["e"] and [list(p) for p in m] == case["m"]
        assert s.k == case["k"] and s.energy() == case["E"]
        w = s.windings()
        js = case["w"]
        assert (w["X1"], w["X2"], w["Z1"], w["Z2"]) == (js["mY"], js["mX"], js["eX"], js["eY"])
        d = toric.decode(s.copy())  # same greedy decoder, same tie-breaking
        assert np.array_equal(d.a, case["da"]) and np.array_equal(d.b, case["db"])

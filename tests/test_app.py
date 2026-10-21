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


FUSE_DRIVER = r"""
const rng = makeRng(%(seed)d), out = [];
for (const L of [3, 4, 6]) {
  const M = makeModel(L);
  for (let t = 0; t < 20; t++) {
    const s = M.fresh();
    for (let g = 0; g < 8; g++) M.apply(s, 'XYZ'[Math.floor(rng() * 3)], Math.floor(rng() * M.nEdges));
    const an = M.anyons(s);
    out.push({ L, e: an.e, m: an.m, fz: M.fuse(an) });
  }
}
// the spin mission: an e at (2, 2) on a 6 x 6 torus, its m walked once through the 4 squares around it
const M = makeModel(6), s = M.fresh(), ops = [['Z', M.h(2, 2)], ['X', M.h(1, 4)], ['X', M.h(1, 3)]];
const spin = [['X', M.h(1, 2)], ['X', M.v(2, 1)], ['X', M.h(2, 2)], ['X', M.v(2, 2)]];
const close = [['X', M.h(1, 3)], ['X', M.h(1, 4)], ['Z', M.h(2, 2)]];
const around = [];
for (const [G, e] of ops) M.apply(s, G, e);
for (const [G, e] of spin) {
  around.push(M.fuse(M.anyons(s)).eps.find(([i, j]) => i === 2 && j === 2)); M.apply(s, G, e);
}
for (const [G, e] of close) M.apply(s, G, e);
console.log(JSON.stringify({ out, around, k: s.k, a: [...s.a], b: [...s.b], star: M.star(2, 2) }));
"""


def run_fuse_js(seed):
    html = APP.read_text()
    model = re.search(r"// MODEL-BEGIN.*?// MODEL-END", html, re.S).group(0)
    code = model + "\n" + FUSE_DRIVER % {"seed": seed}
    return json.loads(subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout)


def test_fusion_view_pairs_neighbours():
    res = run_fuse_js(3)
    for case in res["out"]:
        L, fz = case["L"], case["fz"]
        for i, j, pi, pj in fz["eps"]:  # each ε is an e and an m on one of the 4 squares at its vertex
            assert (pi - i) % L in (0, L - 1) and (pj - j) % L in (0, L - 1)
        assert sorted(fz["e"] + [p[:2] for p in fz["eps"]]) == sorted(case["e"])  # every anyon is drawn exactly once
        assert sorted(fz["m"] + [p[2:] for p in fz["eps"]]) == sorted(case["m"])
        for i, j in fz["e"]:  # no unpaired e is left next to an unpaired m
            assert not any((a - i) % L in (0, L - 1) and (b - j) % L in (0, L - 1) for a, b in fz["m"])


def test_full_turn_of_epsilon_gives_minus_one():
    res = run_fuse_js(3)
    squares = [tuple(p[2:]) for p in res["around"]]  # the m visits all 4 squares while staying bound to the e
    assert squares == [(1, 2), (1, 1), (2, 1), (2, 2)]
    # what is left is X on the star of the e, which is A_v = +1 on the ground state, times i^2 = -1
    assert {e for e, x in enumerate(res["a"]) if x} == set(res["star"]) and not any(res["b"]) and res["k"] == 2


MOVE_DRIVER = r"""
const M = makeModel(6), out = [];
for (const offset of [[0, 0], [-1, 0], [-1, -1], [0, -1]]) {     // the 4 squares at vertex (3, 3) for the m
  const steps = [['+x', M.h(3, 3), [4, 3]], ['-x', M.h(2, 3), [2, 3]],
                 ['+y', M.v(3, 3), [3, 4]], ['-y', M.v(3, 2), [3, 2]]];
  for (const [name, ed, to] of steps) {
    // a lone ε: e pair from (3, 3) to (0, 3), m pair from the chosen square at (3, 3) up a column to row 0
    const s = M.fresh(), [pi, pj] = [3 + offset[0], 3 + offset[1]];
    for (let i = 0; i < 3; i++) M.apply(s, 'Z', M.h(i, 3));
    for (let j = 0; j < pj; j++) M.apply(s, 'X', M.h(pi, j + 1));
    const before = M.fuse(M.anyons(s)), ok = M.moveEps(s, ed), after = M.fuse(M.anyons(s));
    out.push({ name, offset, to, ok, before, after });
  }
}
console.log(JSON.stringify(out));
"""


def test_move_eps_slides_an_epsilon_and_keeps_its_shape():
    html = APP.read_text()
    model = re.search(r"// MODEL-BEGIN.*?// MODEL-END", html, re.S).group(0)
    out = json.loads(subprocess.run(["node", "-e", model + "\n" + MOVE_DRIVER], capture_output=True, text=True,
                                    check=True).stdout)
    for case in out:
        start = [3, 3, (3 + case["offset"][0]) % 6, (3 + case["offset"][1]) % 6]
        assert case["ok"] and start in case["before"]["eps"]
        dx, dy = case["to"][0] - 3, case["to"][1] - 3
        moved = [3 + dx, 3 + dy, (start[2] + dx) % 6, (start[3] + dy) % 6]
        assert moved in case["after"]["eps"], case  # the same ε, one step along, with its m on the same side
        n = lambda f: (len(f["eps"]), len(f["e"]), len(f["m"]))  # noqa: E731
        assert n(case["after"]) == n(case["before"])


SWAP_DRIVER = r"""
const M = makeModel(6), out = {};
// A = (2, 2), O = (3, 2), B = (4, 2), side track C = (3, 3): A -> O -> C, B -> O -> A, C -> O -> B
const legs = [M.h(2, 2), M.v(3, 2), M.h(3, 2), M.h(2, 2), M.v(3, 2), M.h(3, 2)];
for (const kind of ['Z', 'eps']) {
  const s = M.fresh();
  M.apply(s, 'Z', M.h(2, 2)); M.apply(s, 'Z', M.h(3, 2));
  if (kind === 'eps') { M.apply(s, 'X', M.v(3, 2)); M.apply(s, 'X', M.v(4, 2)); }
  const s0 = M.copy(s);
  let last = null;
  for (const ed of legs) last = M.act(s, kind, ed, last);
  out[kind] = { same: s.a.every((x, n) => x === s0.a[n]) && s.b.every((x, n) => x === s0.b[n]), k0: s0.k, k: s.k };
}
console.log(JSON.stringify(out));
"""


def test_swapping_two_epsilons_gives_minus_one_and_two_e_nothing():
    html = APP.read_text()
    model = re.search(r"// MODEL-BEGIN.*?// MODEL-END", html, re.S).group(0)
    out = json.loads(subprocess.run(["node", "-e", model + "\n" + SWAP_DRIVER], capture_output=True, text=True,
                                    check=True).stdout)
    for kind in ("Z", "eps"):
        assert out[kind]["same"]  # the board after the swap is the board before it
    assert out["Z"]["k"] == out["Z"]["k0"]  # e is a boson
    assert (out["eps"]["k"] - out["eps"]["k0"]) % 4 == 2  # ε is a fermion: |ψ⟩ -> -|ψ⟩

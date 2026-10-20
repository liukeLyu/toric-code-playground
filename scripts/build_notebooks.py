"""Write the tutorial notebooks in examples/ from the cell lists below (outputs are filled by executing them)."""

from pathlib import Path

import nbformat as nbf

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = [nbf.v4.new_markdown_cell(s.strip()) if kind == "md" else nbf.v4.new_code_cell(s.strip())
                for kind, s in cells]
    return nb


PLOT_STYLE = '''
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.figsize": (5.5, 3.4)})
'''

N00 = [
    ("md", r"""
# 0. Start here: the toric code in three moves

This notebook is for readers who know what a qubit and the Pauli matrices $X, Y, Z$ are, and have heard words like
*topological order*, *anyons* and *braiding* without them clicking. The fastest way in is to play:
open [`app/toric_code_playground.html`](../app/toric_code_playground.html) in a browser, click an edge, and watch.
Everything the app shows is computed by the same rules as the small Python package used here, `toric`.

**The model.** One qubit sits on every edge of an $L \times L$ square grid whose opposite sides are glued: a torus.

$$H = -\sum_v A_v - \sum_p B_p, \qquad A_v = \prod_{e \ni v} X_e, \qquad B_p = \prod_{e \in p} Z_e .$$

$A_v$ (a *star*) acts on the 4 edges at a vertex, $B_p$ (a *plaquette*) on the 4 edges around a square. They all
commute and each is $\pm 1$, so a ground state has every one of them equal to $+1$.
"""),
    ("code", "import matplotlib.pyplot as plt\nimport numpy as np\n\nimport toric\n" + PLOT_STYLE),
    ("md", r"""
## Move 1: one gate makes two particles

Apply a single $Z$ to one edge of a ground state. It anticommutes with the two stars at the ends of that edge, so
both flip to $-1$. Each flipped star costs energy $+2$ and behaves like a particle, called an **e anyon**.
"""),
    ("code", r"""
s = toric.new_state(6)            # 6 x 6 torus, 72 qubits, in a ground state
t = s.torus
print("ground state:  E =", s.energy())

s.apply("Z", t.h(2, 2))           # Z on the horizontal edge leaving vertex (2, 2)
e, m = s.anyons()
print("after one Z:   E =", s.energy(), " e anyons at", e)
"""),
    ("md", r"""
Anyons always come in pairs out of nothing. Now push one of them: a $Z$ on the next edge flips that star back and
flips the next one. The anyon moves, and **the energy does not change**: only the two ends of the string of $Z$'s
cost energy, never the string itself.
"""),
    ("code", r"""
for step in range(3):
    s.apply("Z", t.h(3 + step, 2))
    print(f"string of {step + 2} Z's:  E = {s.energy()},  e anyons at {s.anyons()[0]}")
"""),
    ("md", r"""
## Move 2: drag a particle around the torus and store a bit

Close a string into a loop that wraps all the way around the torus. No anyons are left and the energy is back to
the ground-state value, yet the state is **different**: the torus has 4 ground states, and loops around it are what
tell them apart. Those 4 states are two qubits, $|q_1 q_2\rangle$, stored in loops that no single edge can see.
"""),
    ("code", r"""
s = toric.new_state(6)
s.apply_string("X", t.x_col(2))   # X across a whole column: an m anyon dragged around the torus
print("anyons:", s.anyons(), " E =", s.energy())
print("logical state, starting from |00>:", np.round(s.logical([1, 0, 0, 0]), 3), "  (that is |10>)")
"""),
    ("md", r"""
## Move 3 (the failure case): heat scrambles the stored qubits

At temperature $T$, random errors create anyon pairs at a rate $\propto e^{-4/T}$. The pairs wander for free and
eventually scramble the two qubits. *Decoding* (joining the anyons back up in pairs) helps only if it is done often.
Below: the number of sweeps (one attempted error per edge each) until the stored state is lost.
"""),
    ("code", r"""
rng = np.random.default_rng(0)
for every in (None, 1):            # never correct, or decode after every sweep
    times = [toric.memory_lifetime(6, 0.6, every, 2000, rng) for _ in range(10)]
    print(f"{'correct every sweep' if every else 'no correction      '}: lost after ~{np.mean(times):.0f} sweeps")
"""),
    ("md", r"""
Without correction the memory is lost quickly, and a bigger torus does **not** help. With correction it survives
much longer, and longer still for bigger tori. Notebook 3 shows why.

## Where to go next

| Step | What you learn | Notebook |
|---|---|---|
| 1 | Anyons in the wavefunction: the ground state as a loop soup, e as a sign twist, m as a loop end, braiding = $-1$ | [`01_anyons`](01_anyons.ipynb) |
| 2 | Two logical qubits: reading them with loops, gates as anyons going around the torus, what braiding cannot do | [`02_logical_qubits`](02_logical_qubits.ipynb) |
| 3 | Heat: thermal anyons, why a lone pair does not simply annihilate, and why protection needs error correction | [`03_thermal_memory`](03_thermal_memory.ipynb) |

## Exercises

1. Make an e pair, then move one e all the way around the torus back to its partner. Is `s.logical([1, 0, 0, 0])`
   changed? Try `s.logical([1, 0, 1, 0] / np.sqrt(2))` instead. (Notebook 2 explains the difference.)
2. Apply $X$ on one edge. Where are the anyons now, and what kind are they (`s.anyons()` returns `(e, m)`)?
3. Rerun the last cell with `L = 4` and `L = 10`. Which column changes, and which does not?
"""),
]

N01 = [
    ("md", r"""
# 1. Anyons in the wavefunction

What *is* an anyon, as a property of the many-qubit wavefunction? This notebook answers it on a $3 \times 3$ torus
(18 qubits), where we can also write the whole state vector down and check every claim exactly
(`toric.ExactToricCode`).
"""),
    ("code", "import matplotlib.pyplot as plt\nimport numpy as np\n\nimport toric\n" + PLOT_STYLE + r"""
L = 3
ex = toric.ExactToricCode(L)
t = ex.torus
print(f"{t.n_edges} qubits, state vectors of length {2**t.n_edges}")
"""),
    ("md", r"""
## The ground state is a soup of closed loops

Write each qubit in the $Z$ basis, $|0\rangle$ or $|1\rangle$. $B_p = +1$ means an even number of $|1\rangle$
edges around every square. Draw each $|1\rangle$ edge as a short segment *across* the edge (on the dual lattice):
then these segments always join into **closed loops**. $A_v = +1$ means that flipping the 4 edges at a vertex,
which adds a tiny loop, does not change the amplitude. So a ground state is an equal superposition of all
closed-loop pictures.
"""),
    ("code", r"""
psi0 = ex.psi0
support = np.flatnonzero(np.abs(psi0) > 1e-12)
print(f"{len(support)} pictures, all with |amplitude|^2 = {abs(psi0[support[0]])**2:.6f} = 1/{len(support)}")
print("energy:", ex.energy(psi0), " (= -2 L^2 =", -2 * L * L, ")")
print("ground-space dimension: the 4 states", "are orthonormal:", np.allclose(ex.basis.conj() @ ex.basis.T, np.eye(4)))
"""),
    ("md", r"""
The other three ground states are the same loop soup with one or two extra loops that wrap around the torus:
there are $2^{L^2 - 1} = 256$ pictures per ground state here, and 4 ground states.

## An m anyon is the end of a loop

$X$ on an edge flips it between $|0\rangle$ and $|1\rangle$ in **every** picture. So every picture now has an open
string, ending at the two squares beside that edge. At those squares the number of $|1\rangle$ edges is odd:
$B_p = -1$. That is an **m anyon**, and you can see it in each single picture.
"""),
    ("code", r"""
s = toric.new_state(L)
s.apply("X", t.h(1, 1))
rng = np.random.default_rng(1)
for _ in range(3):
    base, config, z, amp = s.sample_picture([1, 0, 0, 0], rng)
    counts = [int(config[t.plaquette(i, j)].sum()) for j in range(L) for i in range(L)]
    print("|1> edges around each square:", counts, " -> odd at", [k for k, n in enumerate(counts) if n % 2])
print("m anyons (plaquettes j L + i):", [j * L + i for i, j in s.anyons()[1]])
"""),
    ("md", r"""
## An e anyon is a sign twist between pictures

$Z$ on an edge does not change any picture; it multiplies the amplitude by $-1$ for every picture in which that edge
is $|1\rangle$. Now compare two pictures that differ only by the small loop around one vertex (that is what $A_v$
does). Normally they have the same amplitude. At the two vertices at the ends of the $Z$ they have **opposite
signs**: $A_v = -1$. No single picture shows an e anyon; it lives in the relative signs.
"""),
    ("code", r"""
s = toric.new_state(L)
s.apply("Z", t.h(0, 0))                       # e anyons at vertices (0, 0) and (1, 0)
psi = ex.apply("Z", t.h(0, 0), ex.psi0)
base, config, z, amp = s.sample_picture([1, 0, 0, 0], rng)
for vertex in [(0, 0), (1, 0), (2, 2)]:
    flipped = config.copy()
    flipped[t.star(*vertex)] ^= 1
    ratio = ex.amplitude(flipped, psi) / ex.amplitude(config, psi)
    print(f"vertex {vertex}: amplitude ratio after flipping its star = {ratio.real:+.0f}")
"""),
    ("md", r"""
## The string is invisible

Two strings with the same end points differ by a closed loop. A small closed $Z$ loop is a plaquette $B_p$, which is
$+1$ on the ground state, so it changes nothing. Only where the anyons are matters, plus how the strings wrap the
torus (notebook 2).
"""),
    ("code", r"""
short = ex.apply("Z", t.h(0, 0), ex.psi0)                          # Z on one edge
detour = ex.psi0
for e in [t.v(0, 0), t.h(0, 1), t.v(1, 0)]:                         # the same two end points, the long way round
    detour = ex.apply("Z", e, detour)
print("overlap of the two states:", np.round(np.vdot(short, detour), 12))
"""),
    ("md", r"""
## Braiding: an e taken around an m gives $-1$

$Z$ and $X$ on the same qubit anticommute, $ZX = -XZ$. Make an m pair, then take an e around one m: its closed $Z$
loop crosses the m's $X$ string exactly once. Annihilate everything and you are back in the start state, but with
a minus sign. This $-1$ is the **mutual statistics** of e and m: neither is a boson or a fermion relative to the
other.
"""),
    ("code", r"""
seq = [("X", t.h(1, 1))] + [("Z", e) for e in t.plaquette(1, 1)] + [("X", t.h(1, 1))]
psi = ex.psi0
s = toric.new_state(L)
for gate, e in seq:
    psi = ex.apply(gate, e, psi)
    s.apply(gate, e)
print("exact  <start|end> =", np.round(np.vdot(ex.psi0, psi), 12))
print("toric  <start|end> =", s.logical([1, 0, 0, 0])[0])
"""),
    ("md", r"""
The same holds on any torus: `toric` tracks every state as $i^k X^a Z^b|\text{start}\rangle$ (two bit strings and a
phase), and `tests/test_code.py` checks energies, stabilizers, amplitudes, loops and logical states against these
exact state vectors.

## Exercises

1. Apply $Y$ on one edge. Which stabilizers flip? Check with `ex.stabilizers(psi)`.
2. Take an e around an m **twice**. What is the phase?
3. Take an e around a square that contains *no* m. Show the phase is $+1$, and explain it with the loop soup.
4. Count the pictures in one ground state for $L = 2$ and $L = 3$ and compare with $2^{L^2 - 1}$.
"""),
]

N02 = [
    ("md", r"""
# 2. Two logical qubits and what braiding can do

The 4 ground states of the torus are a register of two qubits, $|q_1 q_2\rangle$. This notebook shows how to read
them, how to apply gates by moving anyons, and why the toric code alone cannot run arbitrary quantum computations.
"""),
    ("code", "import numpy as np\n\nimport toric\n\nex = toric.ExactToricCode(3)\nt = ex.torus"),
    ("md", r"""
## Reading the qubits: loops around the torus

$q_1$ is read by a $Z$ loop along $x$ and $q_2$ by a $Z$ loop along $y$: they give $(-1)^{q_1}$ and $(-1)^{q_2}$.
Moving such a loop by one row multiplies it by the plaquettes in between, all $+1$ in a ground state, so it does not
matter which row you use. No smaller operator can read the qubits: that is why they are protected.
"""),
    ("code", r"""
for z, label in enumerate(["|00>", "|01>", "|10>", "|11>"]):
    psi = ex.basis[z]
    rows = [ex.expectation("Z", t.z_row(r), psi) for r in range(3)]
    cols = [ex.expectation("Z", t.z_col(c), psi) for c in range(3)]
    print(f"{label}: Z loop along x on rows 0,1,2 = {np.round(rows, 6)},  along y on columns = {np.round(cols, 6)}")
"""),
    ("md", r"""
## Gates are anyons dragged around the torus

| Gate | Move | Effect |
|---|---|---|
| $\bar X_1$ | an m around the torus along $y$ ($X$ across a column) | flips $q_1$ |
| $\bar X_2$ | an m around the torus along $x$ | flips $q_2$ |
| $\bar Z_1$ | an e around the torus along $x$ ($Z$ along a row) | $-1$ on the $q_1 = 1$ part |
| $\bar Z_2$ | an e around the torus along $y$ | $-1$ on the $q_2 = 1$ part |

Why is $\bar Z_1$ a *braid*? In the $|q_1 = 1\rangle$ part of the state an m has gone around $y$, so one unit of m
flux threads the loop along $x$. Taking an e around $x$ encircles that flux: $-1$, the e–m braiding phase of
notebook 1. In the $|q_1 = 0\rangle$ part there is no flux and no phase.
"""),
    ("code", r"""
plus0 = np.array([1, 0, 1, 0]) / np.sqrt(2)          # |+0>
s = toric.new_state(6)
s.apply_string("Z", s.torus.z_row(3))                 # an e around x: Zbar_1
print("|+0>  ->", np.round(s.logical(plus0), 3), " = |-0>")
s.apply_string("X", s.torus.x_row(2))                 # an m around x: Xbar_2
print("then Xbar_2 ->", np.round(s.logical(plus0), 3), " = |-1>")
"""),
    ("md", r"""
## What braiding cannot do

Toric-code anyons are **abelian**: braiding only multiplies by $\pm 1$. The gates above are Pauli operators and
nothing more. A Pauli never turns a basis state into a superposition, and never creates entanglement: check it on
all 16 products of the four gates.
"""),
    ("code", r"""
def concurrence(c):
    return 2 * abs(c[0] * c[3] - c[1] * c[2])

paulis = [(a, b, u, w) for a in (0, 1) for b in (0, 1) for u in (0, 1) for w in (0, 1)]
for name, c in [("|00>", [1, 0, 0, 0]), ("|+0>", [1, 0, 1, 0] / np.sqrt(2)), ("Bell", [1, 0, 0, 1] / np.sqrt(2))]:
    C = []
    for a, b, u, w in paulis:
        s = toric.new_state(4)
        T = s.torus
        if a: s.apply_string("X", T.x_col(1))
        if b: s.apply_string("X", T.x_row(1))
        if u: s.apply_string("Z", T.z_row(1))
        if w: s.apply_string("Z", T.z_col(1))
        C.append(concurrence(s.logical(c)))
    print(f"start {name}: concurrence after each of the 16 gates = {sorted({float(round(x, 6)) for x in C})}")
"""),
    ("md", r"""
Entanglement is never created and never destroyed; from a Bell state the gates only move you among the four Bell
states. Computing by braiding alone needs **non-abelian** anyons (for example Fibonacci anyons), where exchanging two
anyons applies a matrix on a degenerate space rather than a phase.

## Exercises

1. Prepare all four basis states from $|00\rangle$ using only $X$ loops. Check them with the $Z$ loops.
2. Start from the Bell state $(|00\rangle + |11\rangle)/\sqrt 2$. Which gate turns it into $(|01\rangle + |10\rangle)/\sqrt 2$?
3. Apply $\bar Z_1$ to $|00\rangle$. Did anything physical change? What about $|10\rangle$?
"""),
]

N03 = [
    ("md", r"""
# 3. Heat, and why protection needs error correction

Here the torus is coupled to a heat bath at temperature $T$. Each sweep, every edge gets on average one attempted
random $X$ or $Z$, accepted with probability $\min(1, e^{-\Delta E/T})$ (Metropolis). Creating a pair costs
$\Delta E = +4$; moving an anyon ($\Delta E = 0$) or annihilating a pair ($\Delta E = -4$) always happens. The state
stays a Pauli string times the start state, so this simulation of Pauli noise is exact.
"""),
    ("code", "import json\nfrom pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nimport toric\n"
     + PLOT_STYLE + "\nRESULTS = Path(\"../results\")\nrng = np.random.default_rng(0)"),
    ("md", r"""
## Equilibrium: heat keeps some anyons around

The second law does not drive the system to zero energy; it minimizes the free energy $F = E - TS$. In the Gibbs
state every stabilizer is independently $-1$ with probability $1/(1 + e^{2/T})$. Live run on a $6 \times 6$ torus:
"""),
    ("code", r"""
s = toric.new_state(6)
fractions = []
for sweep in range(400):
    toric.thermal_sweep(s, 1.0, rng)
    if sweep >= 100:
        fractions.append((s.e_flags.sum() + s.m_flags.sum()) / 72)
print(f"T = 1: fraction of stabilizers at -1 = {np.mean(fractions):.3f}, Gibbs value = {1 / (1 + np.exp(2)):.3f}")
"""),
    ("md", "Full run (`python scripts/make_results.py equilibrium`, under a minute):"),
    ("code", r"""
eq = json.loads((RESULTS / "equilibrium.json").read_text())
T = [p["T"] for p in eq["points"]]
plt.plot(T, [p["gibbs"] for p in eq["points"]], color=BLUE, label="Gibbs: 1 / (1 + e^{2/T})")
plt.plot(T, [p["fraction"] for p in eq["points"]], "o", color=ORANGE, label=f"heat bath, L = {eq['L']}")
plt.xlabel("temperature T")
plt.ylabel("fraction of stabilizers at -1")
plt.legend(frameon=False);
"""),
    ("md", r"""
## A lone pair does not simply annihilate

A fresh pair starts as neighbours, and annihilating is downhill, so many pairs vanish at once. But once the two
anyons are one step apart, **nothing pulls them back**: the energy is $+4$ whatever their separation, because the
string between them costs nothing. Each anyon is then a free random walker. Live: 500 pairs on a $20 \times 20$
torus at $T \to 0$ (no new pairs).
"""),
    ("code", r"""
runs = [toric.single_pair(20, rng) for _ in range(500)]
quick = np.mean([dead and t <= 1 for t, _, dead in runs])
far = np.mean([sep >= 10 for _, sep, _ in runs])
print(f"annihilated within 1 sweep: {quick:.0%}   drifted L/2 = 10 or more apart at some point: {far:.0%}")
"""),
    ("md", "Full run (`python scripts/make_results.py single_pair`, about a minute):"),
    ("code", r"""
sp = json.loads((RESULTS / "single_pair.json").read_text())
print(f"{sp['runs']} pairs on a {sp['L']} x {sp['L']} torus")
for k, f in sp["annihilated_within"].items():
    print(f"  annihilated within {k:>4} sweeps: {f:.1%}")
print(f"  reached L/2 apart: {sp['reached_half_L']:.1%}")
hist = np.array(sp["max_separation_histogram"])
plt.bar(np.arange(len(hist)), hist / hist.sum(), color=BLUE)
plt.xlabel("largest separation reached")
plt.ylabel("fraction of pairs");
"""),
    ("md", r"""
That freedom is what scrambles the memory: a pair that wanders around the torus and meets again has applied a
logical operator. Entropy even favours separation, since there are many more ways to be far apart than close.

## Memory lifetime, without and with correction

*Correct every sweep* measures all stabilizers after each sweep, pairs the closest anyons (greedy decoder) and joins
each pair by a shortest string. A logical error then needs a pair to drift about $L/2$ apart within one sweep.
Live run, 10 runs per point:
"""),
    ("code", r"""
for L in (4, 6, 8):
    out = []
    for every in (None, 1):
        out.append(np.mean([toric.memory_lifetime(L, 0.6, every, 2000, rng) for _ in range(10)]))
    print(f"L = {L}: no correction {out[0]:6.0f} sweeps | correct every sweep {out[1]:6.0f} sweeps")
"""),
    ("md", "Full run (`python scripts/make_results.py memory_lifetimes`, a few minutes on 4 cores):"),
    ("code", r"""
ml = json.loads((RESULTS / "memory_lifetimes.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, T in zip(axes, sorted({p["T"] for p in ml["points"]}), strict=True):
    for every, color, label in [(None, ORANGE, "no correction"), (1, BLUE, "correct every sweep")]:
        pts = [p for p in ml["points"] if p["T"] == T and p["correct_every"] == every]
        ax.semilogy([p["L"] for p in pts], [p["mean"] for p in pts], "o-", color=color, label=label)
        for p in pts:
            if p["capped"]:
                ax.annotate(f"{p['capped']}/{ml['runs']} never failed", (p["L"], p["mean"]), fontsize=7,
                            xytext=(-8, 6), textcoords="offset points", ha="right")
    ax.set_title(f"T = {T}")
    ax.set_xlabel("L")
axes[0].set_ylabel(f"mean sweeps to first logical error\n(runs capped at {ml['max_sweeps']})")
axes[0].legend(frameon=False);
"""),
    ("md", r"""
Without correction a bigger torus does **not** help: the memory time is set by $T$, and it even shrinks a little with
$L$ because there are more places for pairs to appear. With correction it grows rapidly with $L$.

The reason is the **energy barrier**. A logical error only ever costs the $+4$ of one pair, whatever $L$ is, so the
2D toric code is not a *self-correcting* memory. Its topology protects the qubits against small local
perturbations, and against heat only together with active error correction. A memory that protects itself at
$T > 0$ needs a barrier that grows with $L$, as in the 4D toric code, where errors are membranes rather than strings
(Dennis, Kitaev, Landahl, Preskill, [arXiv:quant-ph/0110143](https://arxiv.org/abs/quant-ph/0110143); Alicki,
Horodecki, Horodecki, Horodecki, [arXiv:0811.0033](https://arxiv.org/abs/0811.0033)).

## How often must you correct?

`memory_lifetime(L, T, correct_every=N, ...)` decodes only after every $N$-th sweep. Between corrections the anyons
keep wandering, so the advantage fades fast as $N$ grows. Full run (`python scripts/make_results.py
correction_interval`, a few minutes):
"""),
    ("code", r"""
ci = json.loads((RESULTS / "correction_interval.json").read_text())
for L, color in [(6, ORANGE), (10, BLUE)]:
    pts = [p for p in ci["points"] if p["L"] == L and p["correct_every"]]
    never = next(p for p in ci["points"] if p["L"] == L and p["correct_every"] is None)
    plt.loglog([p["correct_every"] for p in pts], [p["mean"] for p in pts], "o-", color=color, label=f"L = {L}")
    plt.axhline(never["mean"], color=color, ls=":", lw=1)
plt.xlabel("correct every N sweeps")
plt.ylabel("mean sweeps to first logical error")
plt.title(f"T = {ci['T']} (dotted: no correction)")
plt.legend(frameon=False);
"""),
    ("md", r"""
Frequent correction is what makes the bigger torus better: at $N = 1$ the $L = 10$ memory lasts several times longer
than $L = 6$. The advantage shrinks quickly with $N$, and between $N = 10$ and $20$ the curves cross: the bigger
torus is no better, even slightly worse, and both approach no correction at all. Correcting rarely is like not
correcting.

## Exercises

1. Rerun the live memory cell at `T = 0.8`. How does the ratio between the two columns change?
2. For $L = 8$, find the largest $N$ at which correcting still beats $L = 4$ corrected every sweep.
3. In `single_pair`, the energy does not depend on the separation. Change the rule so that each step away from the
   partner costs energy (a string tension). What happens to the largest separation reached?
"""),
]


def main():
    EXAMPLES.mkdir(exist_ok=True)
    for name, cells in [("00_start_here", N00), ("01_anyons", N01), ("02_logical_qubits", N02),
                        ("03_thermal_memory", N03)]:
        nbf.write(notebook(cells), EXAMPLES / f"{name}.ipynb")
        print("wrote", EXAMPLES / f"{name}.ipynb")


if __name__ == "__main__":
    main()

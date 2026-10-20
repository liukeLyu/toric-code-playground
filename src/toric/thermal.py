"""Thermal noise, decoding and memory lifetimes.

Heat bath: pick a random edge and a random X or Z, and accept it with probability min(1, exp(-dE / T)) (Metropolis).
Creating an anyon pair costs dE = +4; moving an anyon (dE = 0) and annihilating a pair (dE = -4) always happen.
One sweep is n_edges attempts, so every edge gets on average one attempted error per sweep.

Decoder: pair up the closest anyons first (greedy matching, separately for e and m) and join each pair by a
shortest string. The state is then back in the ground space; the decoding succeeded if the net logical operator
(noise string plus correction string) is still the identity.
"""

import numpy as np

from .code import PauliState


def delta_energy(state, gate, e):
    flags, sites = (state.e_flags, state.torus.edge_vertices[e]) if gate == "Z" else \
        (state.m_flags, state.torus.edge_plaquettes[e])
    return sum(-2 if flags[s] else 2 for s in sites)


def thermal_sweep(state, T, rng):
    n = state.torus.n_edges
    edges, coins, accept = rng.integers(n, size=n), rng.random(n) < 0.5, rng.random(n)
    for e, z, u in zip(edges, coins, accept, strict=True):
        gate = "Z" if z else "X"
        d = delta_energy(state, gate, e)
        if d <= 0 or u < np.exp(-d / T):
            state.apply(gate, int(e))


def _step(L, a, b):
    d = (b - a) % L
    return d if d <= L / 2 else d - L


def _join(state, gate, p, q):
    """Shortest string from p to q: Z along lattice edges for e anyons, X across edges for m anyons."""
    t, (i, j), (i2, j2) = state.torus, p, q
    dx, dy = _step(t.L, i, i2), _step(t.L, j, j2)
    for _ in range(abs(dx)):
        if gate == "Z":
            state.apply("Z", t.h(i, j) if dx > 0 else t.h(i - 1, j))
        else:
            state.apply("X", t.v(i + 1, j) if dx > 0 else t.v(i, j))
        i += 1 if dx > 0 else -1
    for _ in range(abs(dy)):
        if gate == "Z":
            state.apply("Z", t.v(i, j) if dy > 0 else t.v(i, j - 1))
        else:
            state.apply("X", t.h(i, j + 1) if dy > 0 else t.h(i, j))
        j += 1 if dy > 0 else -1


def greedy_pairs(torus, sites):
    n = len(sites)
    cand = sorted((torus.distance(sites[a], sites[b]), a, b) for a in range(n) for b in range(a + 1, n))
    used, pairs = set(), []
    for _, a, b in cand:
        if a not in used and b not in used:
            used.update((a, b))
            pairs.append((sites[a], sites[b]))
    return pairs


def decode(state):
    """Remove every anyon by joining the closest pairs. Returns the state (modified in place)."""
    e, m = state.anyons()
    for p, q in greedy_pairs(state.torus, e):
        _join(state, "Z", p, q)
    for p, q in greedy_pairs(state.torus, m):
        _join(state, "X", p, q)
    return state


def logical_error(state):
    """Would decoding now apply a non-trivial logical operator?"""
    return any(decode(state.copy()).windings().values())


def memory_lifetime(L, T, correct_every, max_sweeps, rng):
    """Sweeps until the first logical error (max_sweeps if none), starting from a ground state.

    correct_every = N decodes the state after every N-th sweep (None or 0: never). The memory counts as lost at the
    first sweep where decoding would give a logical error, so every setting is judged by the same rule.
    """
    from .code import Torus

    state = PauliState(Torus(L))
    for t in range(1, max_sweeps + 1):
        thermal_sweep(state, T, rng)
        if correct_every and t % correct_every == 0:
            decode(state)
        if logical_error(state):
            return t
    return max_sweeps


def single_pair(L, rng, max_sweeps=5000):
    """Fate of one e pair at T -> 0 (no new pairs): each anyon hops 2 times per sweep on average.

    Returns (sweeps until it annihilates or max_sweeps, largest separation reached, annihilated?).
    """
    from .code import Torus

    torus = Torus(L)
    p, moves, far = [[0, 0], [1, 0]], 0, 1
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while moves < 4 * max_sweeps:
        moves += 1
        a = int(rng.random() < 0.5)
        d = steps[int(rng.integers(4))]
        q = [(p[a][0] + d[0]) % L, (p[a][1] + d[1]) % L]
        if q == p[1 - a]:
            if rng.random() < 0.5:  # the edge between neighbours is reached from either end
                return moves / 4, far, True
        else:
            p[a] = q
            far = max(far, torus.distance(p[0], p[1]))
    return max_sweeps, far, False

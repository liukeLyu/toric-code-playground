"""The toric code on an L x L torus, tracked exactly as a Pauli string acting on a ground state.

Qubits live on edges. Edge h(i, j) = 2 (j L + i) joins vertex (i, j) to (i+1, j); edge v(i, j) = h(i, j) + 1 joins
(i, j) to (i, j+1). Plaquette (i, j) is the square with lower-left corner (i, j).

    H = - sum_v A_v - sum_p B_p,    A_v = prod of X on the 4 edges at v,    B_p = prod of Z around p.

Every state reached by Pauli gates from a ground state |L> has the form

    |psi> = i^k X^a Z^b |L>,       a, b in {0, 1}^(2 L^2),

so it is stored by two bit strings and a phase, for any L. The start state |L> = sum_z c_z |z> lives in the
4-dimensional ground space, with logical basis |z1 z2> = Xbar1^z1 Xbar2^z2 |psi0> and index z = 2 z1 + z2.
The state stores c along with a, b and k.
|psi0> is the equal superposition of all closed loops of |1> edges on the dual lattice; all Z loops are +1 on it.

Logical operators (reference loops):
    Zbar1 = Z on the horizontal edges of row 0      (Z loop along x, reads q1)
    Xbar1 = X on the horizontal edges of column 0   (X loop along y, flips q1)
    Zbar2 = Z on the vertical edges of column 0     (Z loop along y, reads q2)
    Xbar2 = X on the vertical edges of row 0        (X loop along x, flips q2)
"""

from dataclasses import dataclass, field

import numpy as np


class Torus:
    """Geometry of the L x L torus: edges, stars, plaquettes and the reference loops."""

    def __init__(self, L):
        self.L = L
        self.n_edges = 2 * L * L
        self.stars = [self.star(i, j) for j in range(L) for i in range(L)]
        self.plaquettes = [self.plaquette(i, j) for j in range(L) for i in range(L)]
        # for each edge: the two vertices it joins and the two plaquettes beside it (as flat indices j L + i)
        self.edge_vertices = [[self._site(*p) for p in self.ends(e)] for e in range(self.n_edges)]
        self.edge_plaquettes = [[self._site(*p) for p in self.sides(e)] for e in range(self.n_edges)]

    def _site(self, i, j):
        return (j % self.L) * self.L + (i % self.L)

    def h(self, i, j):
        return 2 * self._site(i, j)

    def v(self, i, j):
        return self.h(i, j) + 1

    def star(self, i, j):
        return [self.h(i, j), self.h(i - 1, j), self.v(i, j), self.v(i, j - 1)]

    def plaquette(self, i, j):
        return [self.h(i, j), self.h(i, j + 1), self.v(i, j), self.v(i + 1, j)]

    def edge_info(self, e):
        cell = e >> 1
        return cell % self.L, cell // self.L, e % 2 == 0  # i, j, horizontal

    def ends(self, e):
        i, j, horizontal = self.edge_info(e)
        return [(i, j), ((i + 1) % self.L, j)] if horizontal else [(i, j), (i, (j + 1) % self.L)]

    def sides(self, e):
        i, j, horizontal = self.edge_info(e)
        return [(i, j), (i, (j - 1) % self.L)] if horizontal else [(i, j), ((i - 1) % self.L, j)]

    def z_row(self, r):
        """Z loop along x on row r."""
        return [self.h(i, r) for i in range(self.L)]

    def x_row(self, r):
        """X loop along x, crossing the vertical edges above row r."""
        return [self.v(i, r) for i in range(self.L)]

    def z_col(self, c):
        """Z loop along y on column c."""
        return [self.v(c, j) for j in range(self.L)]

    def x_col(self, c):
        """X loop along y, crossing the horizontal edges right of column c."""
        return [self.h(c, j) for j in range(self.L)]

    def distance(self, p, q):
        """Manhattan distance on the torus between two sites (i, j)."""
        dx, dy = abs(p[0] - q[0]) % self.L, abs(p[1] - q[1]) % self.L
        return min(dx, self.L - dx) + min(dy, self.L - dy)


@dataclass
class PauliState:
    """|psi> = i^k X^a Z^b |start>. Gates are applied on the left; the stabilizer violations are kept up to date.

    |start> = sum_z c_z |z> is a ground state with logical amplitudes c (index z = 2 z1 + z2).
    """

    torus: Torus
    c: np.ndarray = field(default=None)  # logical amplitudes of |start>
    a: np.ndarray = field(default=None)
    b: np.ndarray = field(default=None)
    k: int = 0
    e_flags: np.ndarray = field(default=None)  # A_v = -1 at vertex v (an e anyon)
    m_flags: np.ndarray = field(default=None)  # B_p = -1 at plaquette p (an m anyon)

    def __post_init__(self):
        n, s = self.torus.n_edges, self.torus.L**2
        c = np.asarray([1, 0, 0, 0] if self.c is None else self.c, complex)
        self.c = c / np.linalg.norm(c)
        self.a = np.zeros(n, np.uint8) if self.a is None else self.a
        self.b = np.zeros(n, np.uint8) if self.b is None else self.b
        self.e_flags = np.zeros(s, np.uint8) if self.e_flags is None else self.e_flags
        self.m_flags = np.zeros(s, np.uint8) if self.m_flags is None else self.m_flags

    def copy(self):
        return PauliState(self.torus, self.c, self.a.copy(), self.b.copy(), self.k, self.e_flags.copy(),
                          self.m_flags.copy())

    def __repr__(self):
        phase = ["1", "i", "-1", "-i"][self.k]
        x, z = np.flatnonzero(self.a).tolist(), np.flatnonzero(self.b).tolist()
        c = np.round(self.c, 3)
        c = c.real + 0.0 if not c.imag.any() else c
        return f"PauliState(L={self.torus.L}: {phase} * X on {x} * Z on {z} |start>, start c = {c})"

    def apply(self, gate, e):
        """Left-multiply by X, Y or Z on edge e, keeping the form i^k X^a Z^b."""
        if gate in ("Z", "Y"):
            self.k = (self.k + 2 * int(self.a[e])) % 4  # Z_e X^a = (-1)^{a_e} X^a Z_e
            for vtx in self.torus.edge_vertices[e]:
                self.e_flags[vtx] ^= 1
        if gate == "Y":
            self.k = (self.k + 1) % 4  # Y = i X Z
        if gate in ("X", "Y"):
            self.a[e] ^= 1
            for p in self.torus.edge_plaquettes[e]:
                self.m_flags[p] ^= 1
        if gate in ("Z", "Y"):
            self.b[e] ^= 1
        return self

    def apply_string(self, gate, edges):
        for e in edges:
            self.apply(gate, e)
        return self

    # ---- what the stabilizers see
    def anyons(self):
        L = self.torus.L
        e = [(int(s) % L, int(s) // L) for s in np.flatnonzero(self.e_flags)]
        m = [(int(s) % L, int(s) // L) for s in np.flatnonzero(self.m_flags)]
        return e, m

    @property
    def in_ground_space(self):
        return not (self.e_flags.any() or self.m_flags.any())

    def energy(self):
        """E = -2 L^2 + 2 (number of stabilizers at -1)."""
        return -2 * self.torus.L**2 + 2 * int(self.e_flags.sum() + self.m_flags.sum())

    # ---- logical content
    def windings(self):
        """Net logical Pauli carried by the strings: which reference loops they cross an odd number of times."""
        t = self.torus
        return {
            "X1": int(self.a[t.z_row(0)].sum() % 2),  # an m went around y: flips q1
            "X2": int(self.a[t.z_col(0)].sum() % 2),  # an m went around x: flips q2
            "Z1": int(self.b[t.x_col(0)].sum() % 2),  # an e went around x
            "Z2": int(self.b[t.x_row(0)].sum() % 2),  # an e went around y
        }

    def logical(self):
        """The 4 logical amplitudes of |psi> (valid only without anyons)."""
        if not self.in_ground_space:
            raise ValueError("anyons present: |psi> has no ground-space component")
        w = self.windings()
        z = np.arange(4)
        d = self.c * (-1.0) ** (w["Z1"] * (z >> 1) + w["Z2"] * (z & 1))  # Zbar first
        d = d[z ^ (2 * w["X1"]) ^ w["X2"]]  # then Xbar
        return 1j**self.k * d

    def loop_expectation(self, kind, index):
        """<psi| loop |psi> for kind in 'Zx', 'Xx', 'Zy', 'Xy' on row or column `index`."""
        t, c = self.torus, self.c
        edges = {"Zx": t.z_row, "Xx": t.x_row, "Zy": t.z_col, "Xy": t.x_col}[kind](index)
        crossed = int((self.a if kind[0] == "Z" else self.b)[edges].sum() % 2)  # opposite-type strings it crosses
        p = np.abs(c) ** 2
        logical = {
            "Zx": p[0] + p[1] - p[2] - p[3],
            "Zy": p[0] - p[1] + p[2] - p[3],
            "Xy": 2 * np.real(np.conj(c[0]) * c[2] + np.conj(c[1]) * c[3]),
            "Xx": 2 * np.real(np.conj(c[0]) * c[1] + np.conj(c[2]) * c[3]),
        }[kind]
        return (-1) ** crossed * logical

    # ---- the wavefunction in the Z basis
    def amplitude(self, base, z):
        """<config|psi> for config = base xor a, where base is a closed-loop picture of sector z.

        Every picture has the same size 2^{-(L^2 - 1)/2}; this returns the amplitude in that unit.
        """
        sign = (-1) ** int(self.b[np.asarray(base, bool)].sum() % 2)
        return 1j**self.k * self.c[z] * sign

    def sample_picture(self, rng):
        """A random Z-basis configuration of |psi> and its amplitude (in units of 2^{-(L^2-1)/2})."""
        t = self.torus
        p = np.abs(self.c) ** 2
        z = int(rng.choice(4, p=p / p.sum()))
        base = np.zeros(t.n_edges, np.uint8)
        for s in np.flatnonzero(rng.random(t.L**2) < 0.5):
            base[t.stars[s]] ^= 1
        if z >> 1:
            base[t.x_col(0)] ^= 1
        if z & 1:
            base[t.x_row(0)] ^= 1
        return base, base ^ self.a, z, self.amplitude(base, z)


START_STATES = {
    "00": [1, 0, 0, 0],
    "+0": np.array([1, 0, 1, 0]) / np.sqrt(2),
    "++": np.full(4, 0.5),
    "phi+": np.array([1, 0, 0, 1]) / np.sqrt(2),
}


def new_state(L, start="00"):
    """A ground state of the L x L torus: start is a name in START_STATES or 4 logical amplitudes c."""
    return PauliState(Torus(L), START_STATES[start] if isinstance(start, str) else start)

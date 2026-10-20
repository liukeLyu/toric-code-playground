"""Brute-force state vectors of the toric code for small tori (L <= 3, i.e. up to 18 qubits).

These are the exact references the Pauli-string bookkeeping in `toric.code` is checked against. A basis state is an
integer whose bit (n - 1 - e) is the Z-basis value of edge e, so edge 0 is the most significant bit.
"""

import numpy as np

from .code import Torus


def _mask(edges, n):
    m = 0
    for e in edges:
        m ^= 1 << (n - 1 - e)
    return m


class ExactToricCode:
    """Full 2^(2 L^2) state vectors of the L x L toric code."""

    def __init__(self, L):
        if L > 3:
            raise ValueError("exact state vectors are only practical up to L = 3 (18 qubits)")
        self.torus = t = Torus(L)
        self.n = t.n_edges
        self.configs = np.arange(2**self.n)
        psi = np.zeros(2**self.n, complex)
        psi[0] = 1.0  # all edges |0>: every B_p = +1
        for star in t.stars:  # project onto every A_v = +1
            psi = (psi + self.x(star, psi)) / 2
        self.psi0 = psi / np.linalg.norm(psi)
        # logical basis |z1 z2> = Xbar1^z1 Xbar2^z2 |psi0>, index z = 2 z1 + z2
        self.basis = []
        for z in range(4):
            s = self.psi0
            if z >> 1:
                s = self.x(t.x_col(0), s)
            if z & 1:
                s = self.x(t.x_row(0), s)
            self.basis.append(s)
        self.basis = np.array(self.basis)

    def x(self, edges, psi):
        return psi[self.configs ^ _mask(edges, self.n)]

    def z(self, edges, psi):
        parity = np.bitwise_count(self.configs & _mask(edges, self.n)).astype(int) & 1
        return (1 - 2 * parity) * psi

    def apply(self, gate, e, psi):
        if gate == "X":
            return self.x([e], psi)
        if gate == "Z":
            return self.z([e], psi)
        return 1j * self.x([e], self.z([e], psi))  # Y = i X Z

    def state(self, c):
        return np.asarray(c, complex) @ self.basis

    def energy(self, psi):
        t = self.torus
        stars = sum(np.vdot(psi, self.x(s, psi)).real for s in t.stars)
        plaqs = sum(np.vdot(psi, self.z(p, psi)).real for p in t.plaquettes)
        return -stars - plaqs

    def stabilizers(self, psi):
        """(<A_v> for every vertex, <B_p> for every plaquette)."""
        t = self.torus
        return (np.array([np.vdot(psi, self.x(s, psi)).real for s in t.stars]),
                np.array([np.vdot(psi, self.z(p, psi)).real for p in t.plaquettes]))

    def expectation(self, kind, edges, psi):
        op = self.x if kind == "X" else self.z
        return np.vdot(psi, op(edges, psi)).real

    def amplitude(self, config_bits, psi):
        index = int(sum(int(bit) << (self.n - 1 - e) for e, bit in enumerate(config_bits)))
        return psi[index]

    def logical_amplitudes(self, psi):
        return self.basis.conj() @ psi

"""Pauli-string bookkeeping against brute-force 18-qubit state vectors (L = 3)."""

import numpy as np
import pytest

import toric

L = 3
EX = toric.ExactToricCode(L)


def random_start(rng):
    c = rng.normal(size=4) + 1j * rng.normal(size=4)
    return c / np.linalg.norm(c)


def run(seq, c):
    """Apply a gate sequence both ways: Pauli bookkeeping and exact state vector."""
    s, psi = toric.new_state(L), EX.state(c)
    for gate, e in seq:
        s.apply(gate, e)
        psi = EX.apply(gate, e, psi)
    return s, psi


def closed_sequence(rng):
    """Gates that end in a ground state: loops around the torus, small loops, and a braid."""
    t, seq = EX.torus, []
    for _ in range(6):
        gate, edges = [("Z", t.z_row(1)), ("X", t.x_row(2)), ("Z", t.z_col(2)), ("X", t.x_col(1)),
                       ("Z", t.plaquette(1, 1)), ("X", t.star(2, 0))][rng.integers(6)]
        seq += [(gate, e) for e in edges]
    seq += [("X", t.h(1, 1))] + [("Z", e) for e in t.plaquette(1, 1)] + [("X", t.h(1, 1))]  # e around m: -1
    return seq + [("Y", 4), ("Y", 4)]


def random_sequence(rng, n=15):
    return [("XYZ"[rng.integers(3)], int(rng.integers(EX.n))) for _ in range(n)]


@pytest.mark.parametrize("seed", range(6))
def test_energy_anyons_and_amplitudes_match_exact(seed):
    rng = np.random.default_rng(seed)
    c = random_start(rng)
    s, psi = run(random_sequence(rng), c)
    assert s.energy() == pytest.approx(EX.energy(psi))
    stars, plaqs = EX.stabilizers(psi)
    assert np.array_equal(s.e_flags, (stars < 0).astype(np.uint8))
    assert np.array_equal(s.m_flags, (plaqs < 0).astype(np.uint8))
    for _ in range(5):  # sampled Z-basis pictures carry the exact amplitude
        _, config, _, amp = s.sample_picture(c, rng)
        assert EX.amplitude(config, psi) == pytest.approx(amp * 2 ** (-(L * L - 1) / 2))


@pytest.mark.parametrize("seed", range(6))
def test_loop_expectations_match_exact(seed):
    rng = np.random.default_rng(10 + seed)
    c = random_start(rng)
    s, psi = run(random_sequence(rng) if seed % 2 else closed_sequence(rng), c)
    t = EX.torus
    for kind, edges in [("Zx", t.z_row), ("Xx", t.x_row), ("Zy", t.z_col), ("Xy", t.x_col)]:
        for r in range(L):
            assert s.loop_expectation(kind, r, c) == pytest.approx(EX.expectation(kind[0], edges(r), psi), abs=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_logical_state_and_net_gate_match_exact(seed):
    rng = np.random.default_rng(20 + seed)
    c = random_start(rng)
    s, psi = run(closed_sequence(rng), c)
    assert s.in_ground_space
    assert np.allclose(EX.logical_amplitudes(psi), s.logical(c))
    # the net gate i^k Xbar^w Zbar^u reproduces the state
    w, t, ref = s.windings(), EX.torus, EX.state(c)
    if w["Z1"]:
        ref = EX.z(t.z_row(0), ref)
    if w["Z2"]:
        ref = EX.z(t.z_col(0), ref)
    if w["X1"]:
        ref = EX.x(t.x_col(0), ref)
    if w["X2"]:
        ref = EX.x(t.x_row(0), ref)
    assert np.allclose(psi, 1j**s.k * ref)


def test_braid_gives_minus_one():
    t, s = EX.torus, toric.new_state(L)
    s.apply("X", t.h(1, 1))  # m pair
    s.apply_string("Z", t.plaquette(1, 1))  # an e loop around one m
    s.apply("X", t.h(1, 1))  # annihilate the m pair
    assert s.in_ground_space and np.allclose(s.logical([1, 0, 0, 0]), [-1, 0, 0, 0])


def test_ground_space_is_four_dimensional_and_loops_label_it():
    gram = EX.basis.conj() @ EX.basis.T
    assert np.allclose(gram, np.eye(4))
    for z in range(4):
        assert EX.energy(EX.basis[z]) == pytest.approx(-2 * L * L)
        assert EX.expectation("Z", EX.torus.z_row(0), EX.basis[z]) == pytest.approx((-1) ** (z >> 1))
        assert EX.expectation("Z", EX.torus.z_col(0), EX.basis[z]) == pytest.approx((-1) ** (z & 1))

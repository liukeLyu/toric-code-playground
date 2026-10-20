"""Heat bath, decoder and memory: energy bookkeeping, equilibrium and the effect of correction."""

import numpy as np
import pytest

import toric


def noisy_state(L, n_errors, rng):
    s = toric.new_state(L)
    for _ in range(n_errors):
        s.apply("XZ"[rng.integers(2)], int(rng.integers(s.torus.n_edges)))
    return s


@pytest.mark.parametrize("L", [4, 5, 6])
def test_delta_energy_is_the_energy_change(L):
    rng = np.random.default_rng(L)
    s = noisy_state(L, 20, rng)
    for _ in range(200):
        gate, e = "XZ"[rng.integers(2)], int(rng.integers(s.torus.n_edges))
        before = s.energy()
        d = toric.delta_energy(s, gate, e)
        s.apply(gate, e)
        assert s.energy() - before == d


@pytest.mark.parametrize("L", [4, 5, 6, 8])
def test_decoder_removes_every_anyon(L):
    rng = np.random.default_rng(100 + L)
    for _ in range(50):
        s = toric.decode(noisy_state(L, 25, rng))
        assert s.in_ground_space


def test_decoder_undoes_a_short_error():
    s = toric.new_state(6)
    s.apply_string("Z", [s.torus.h(1, 2), s.torus.h(2, 2)])  # e pair two steps apart
    toric.decode(s)
    assert s.in_ground_space and not any(s.windings().values())


def test_equilibrium_density_matches_gibbs():
    """In the Gibbs state every stabilizer is -1 with probability 1 / (1 + e^{2/T}) (nearly independently)."""
    L, T, rng = 8, 1.0, np.random.default_rng(1)
    s = toric.new_state(L)
    fractions = []
    for sweep in range(1500):
        toric.thermal_sweep(s, T, rng)
        if sweep >= 300:
            fractions.append((s.e_flags.sum() + s.m_flags.sum()) / (2 * L * L))
    assert np.mean(fractions) == pytest.approx(1 / (1 + np.exp(2 / T)), abs=0.004)


def test_correction_extends_the_memory_and_more_often_is_better():
    rng = np.random.default_rng(2)
    life = {n: np.mean([toric.memory_lifetime(6, 0.6, n, 2000, rng) for _ in range(20)]) for n in (None, 5, 1)}
    assert life[1] > 3 * life[5] and life[5] > 1.5 * life[None]


def test_single_pair_often_annihilates_at_once_but_can_wander_far():
    rng = np.random.default_rng(3)
    runs = [toric.single_pair(20, rng) for _ in range(2000)]
    quick = np.mean([dead and t <= 1 for t, _, dead in runs])
    far = np.mean([sep >= 10 for _, sep, _ in runs])
    assert 0.15 < quick < 0.25 and far > 0.4

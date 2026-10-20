"""Full-size runs behind the notebook figures, saved as small JSON summaries in results/.

    python scripts/make_results.py                     # every result, spread over all CPU cores
    python scripts/make_results.py memory_lifetimes

Each notebook runs a seconds-long live version of a result, then loads the full one from results/.
Every task has its own seed, so the files are reproducible. About 5 minutes in total on 4 cores.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import toric

RESULTS = Path(__file__).resolve().parent.parent / "results"


def lifetimes(L, T, correct_every, runs, max_sweeps, seed):
    rng = np.random.default_rng(seed)
    t = [toric.memory_lifetime(L, T, correct_every, max_sweeps, rng) for _ in range(runs)]
    return {"L": L, "T": T, "correct_every": correct_every, "mean": float(np.mean(t)),
            "median": float(np.median(t)), "capped": int(sum(x == max_sweeps for x in t))}


def single_pairs(L, runs, seed):
    rng = np.random.default_rng(seed)
    return [toric.single_pair(L, rng) for _ in range(runs)]


def equilibrium(L, T, sweeps, burn_in, seed):
    rng, s = np.random.default_rng(seed), toric.new_state(L)
    f = []
    for k in range(sweeps):
        toric.thermal_sweep(s, T, rng)
        if k >= burn_in:
            f.append((s.e_flags.sum() + s.m_flags.sum()) / (2 * L * L))
    return {"T": T, "fraction": float(np.mean(f)), "gibbs": float(1 / (1 + np.exp(2 / T)))}


def assemble_pairs(out):
    runs = [r for chunk in out for r in chunk]
    t = np.array([r[0] for r in runs])
    sep = np.array([r[1] for r in runs])
    dead = np.array([r[2] for r in runs])
    return {
        "L": 20, "runs": len(runs),
        "annihilated_within": {str(k): float(np.mean(dead & (t <= k))) for k in [1, 10, 100, 1000]},
        "reached_half_L": float(np.mean(sep >= 10)),
        "max_separation_histogram": np.bincount(sep, minlength=21).tolist(),
    }


SIZES, TEMPS, RUNS, MAX = [4, 6, 8, 10, 12, 14], [0.6, 0.8], 200, 10000
INTERVALS = [1, 2, 3, 5, 10, 20, None]

# Each result: (list of tasks, function assembling the task outputs into a JSON-ready dict).
RESULTS_SPEC = {
    # notebook 03: time to the first logical error, without correction and with correction after every sweep
    "memory_lifetimes": (
        [(lifetimes, L, T, c, RUNS, MAX, 1000 * k + 10 * L + bool(c))
         for k, T in enumerate(TEMPS) for L in SIZES for c in (None, 1)],
        lambda out: {"runs": RUNS, "max_sweeps": MAX, "points": out},
    ),
    # notebook 03: how often to correct: lifetime against the correction interval N, at T = 0.6
    "correction_interval": (
        [(lifetimes, L, 0.6, n, RUNS, MAX, 50000 + 100 * L + (n or 0)) for L in (6, 10) for n in INTERVALS],
        lambda out: {"T": 0.6, "runs": RUNS, "max_sweeps": MAX, "points": out},
    ),
    # notebook 03: the fate of one pair at T -> 0 on a 20 x 20 torus
    "single_pair": (
        [(single_pairs, 20, 5000, seed) for seed in range(4)],
        assemble_pairs,
    ),
    # notebook 03: fraction of stabilizers at -1 in equilibrium, against the Gibbs value
    "equilibrium": (
        [(equilibrium, 10, T, 3000, 500, 7) for T in [0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]],
        lambda out: {"L": 10, "sweeps": 3000, "burn_in": 500, "points": out},
    ),
}


def run(task):
    return task[0](*task[1:])


def main(names):
    names = names or list(RESULTS_SPEC)
    tasks = [(name, k, t) for name in names for k, t in enumerate(RESULTS_SPEC[name][0])]
    with ProcessPoolExecutor() as pool:
        outputs = dict(zip([(n, k) for n, k, _ in tasks], pool.map(run, [t for _, _, t in tasks]), strict=True))
    RESULTS.mkdir(exist_ok=True)
    for name in names:
        out = [outputs[name, k] for k in range(len(RESULTS_SPEC[name][0]))]
        data = {"made_by": f"python scripts/make_results.py {name}", **RESULTS_SPEC[name][1](out)}
        (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=1, default=float) + "\n")
        print("wrote", RESULTS / f"{name}.json")


if __name__ == "__main__":
    main(sys.argv[1:])

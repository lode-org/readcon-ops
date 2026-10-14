"""Time the batch kernels: numpy fallback, compiled float, compiled exact.

python bench/bench_kernels.py [frames] [atoms] [repeats]
Prints one line per case with the median wall time in seconds.
"""

import statistics
import sys
import time

import numpy as np

from readcon_ops import kernels
from readcon_ops.frames import Frame


def _free(r):
    return Frame(r, np.eye(3) * 40.0, ["C"] * len(r), None, (False, False, False))


def median(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    rng = np.random.default_rng(0)
    ref = rng.normal(scale=3.0, size=(n, 3))
    frames = [_free(ref + 0.1 * rng.normal(size=(n, 3))) for _ in range(m)]
    masses = rng.uniform(1.0, 60.0, n)
    v = rng.normal(size=(n, 3))
    cases = {
        "align_all float": lambda: kernels.align_all(_free(ref), frames, masses),
        "align_all exact": lambda: kernels.align_all(_free(ref), frames, masses, exact=True),
        "rmsd_matrix float": lambda: kernels.rmsd_matrix(frames, masses),
        "rmsd_matrix exact": lambda: kernels.rmsd_matrix(frames, masses, exact=True),
        "kinetic_energy float": lambda: kernels.kinetic_energy(masses, v),
        "kinetic_energy exact": lambda: kernels.kinetic_energy(masses, v, exact=True),
    }
    for name, fn in cases.items():
        print(f"{name:24s} frames={m} atoms={n} median_s={median(fn, reps):.6f}")
    saved = kernels._core
    kernels._core, kernels.HAVE_CORE = None, False
    try:
        t = median(lambda: kernels.align_all(_free(ref), frames, masses), reps)
        print(f"{'align_all numpy':24s} frames={m} atoms={n} median_s={t:.4f}")
        t = median(lambda: kernels.rmsd_matrix(frames, masses), max(1, reps // 5))
        print(f"{'rmsd_matrix numpy':24s} frames={m} atoms={n} median_s={t:.4f}")
    finally:
        kernels._core, kernels.HAVE_CORE = saved, True


if __name__ == "__main__":
    main()

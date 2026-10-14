"""mpiexec -n P python bench/bench_parallel.py [frames] [atoms] [repeats]

Median wall time of the MPI all-pairs RMSD matrix, float and exact, and a
sha256 of each matrix (it must not change with P). Set RAYON_NUM_THREADS
so that ranks times threads fits the cores.
"""

import hashlib
import statistics
import sys
import time

import numpy as np
from mpi4py import MPI

from readcon_ops import parallel
from readcon_ops.frames import Frame

comm = MPI.COMM_WORLD
m = int(sys.argv[1]) if len(sys.argv) > 1 else 200
n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rng = np.random.default_rng(0)
ref = rng.normal(scale=3.0, size=(n, 3))
frames = [
    Frame(ref + 0.1 * rng.normal(size=(n, 3)), np.eye(3) * 40, ["C"] * n, None, (False,) * 3)
    for _ in range(m)
]
masses = rng.uniform(1.0, 60.0, n)
for exact in (False, True):
    ts = []
    for _ in range(reps):
        comm.Barrier()
        t = time.perf_counter()
        out = parallel.rmsd_matrix(frames, masses, True, exact, comm)
        comm.Barrier()
        ts.append(time.perf_counter() - t)
    if comm.Get_rank() == 0:
        h = hashlib.sha256(out.tobytes()).hexdigest()[:16]
        tag = "exact" if exact else "float"
        t = statistics.median(ts)
        print(f"rmsd_matrix {tag} ranks={comm.Get_size()} median_s={t:.4f} sha={h}")

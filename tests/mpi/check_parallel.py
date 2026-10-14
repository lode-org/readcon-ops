"""mpiexec -n P python tests/mpi/check_parallel.py

Runs every parallel function on COMM_WORLD and with comm=None, and on rank 0
prints one line per check plus a sha256 over all MPI results. The digest
must be the same for every P. Exits 1 if any check fails.
"""

import hashlib
import sys

import numpy as np
from mpi4py import MPI

from readcon_ops import parallel
from readcon_ops.frames import Frame


def free(r):
    return Frame(r, np.eye(3) * 40.0, ["C"] * len(r), None, (False, False, False))


comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
rng = np.random.default_rng(42)
n_atoms, n_frames = 157, 23
ref = rng.normal(scale=3.0, size=(n_atoms, 3))
frames = [free(ref + 0.2 * rng.normal(size=(n_atoms, 3))) for _ in range(n_frames)]
masses = rng.uniform(1.0, 200.0, n_atoms)
vel = rng.normal(size=(n_atoms, 3)) * 10.0 ** rng.integers(-3, 3, size=(n_atoms, 1))
energies = rng.normal(size=41) * 1e3
mine = parallel.block(n_atoms, size, rank)
emine = parallel.block(len(energies), size, rank)
perm = rng.permutation(n_atoms)
pmine = perm[list(mine)]

checks, digest = [], hashlib.sha256()


def check(name, got, want):
    g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    checks.append((name, g.tobytes() == w.tobytes()))
    digest.update(g.tobytes())


for exact in (False, True):
    tag = "exact" if exact else "float"
    r, mv = parallel.align_all(free(ref), frames, masses, exact, comm)
    r0, mv0 = parallel.align_all(free(ref), frames, masses, exact)
    check(f"align_all {tag}", np.concatenate([r, mv.ravel()]), np.concatenate([r0, mv0.ravel()]))
    check(
        f"rmsd_matrix {tag}",
        parallel.rmsd_matrix(frames, masses, True, exact, comm, tile=5),
        parallel.rmsd_matrix(frames, masses, True, exact),
    )
    check(
        f"band_mean {tag}",
        parallel.band_mean(energies[list(emine)], comm, exact),
        parallel.band_mean(energies, None, exact),
    )
    check(
        f"kinetic_energy {tag}",
        parallel.kinetic_energy(masses[list(mine)], vel[list(mine)], comm, exact),
        parallel.kinetic_energy(masses, vel, None, exact),
    )
    check(
        f"centroid {tag}",
        parallel.centroid(ref[list(mine)], masses[list(mine)], comm, exact),
        parallel.centroid(ref, masses, None, exact),
    )

# Exact values also ignore the atom order across ranks.
check(
    "kinetic_energy exact shuffled",
    parallel.kinetic_energy(masses[pmine], vel[pmine], comm, True),
    parallel.kinetic_energy(masses, vel, None, True),
)
check(
    "centroid exact shuffled",
    parallel.centroid(ref[pmine], masses[pmine], comm, True),
    parallel.centroid(ref, masses, None, True),
)

if rank == 0:
    for name, ok in checks:
        print(f"{name:32s} {'ok' if ok else 'MISMATCH'}")
    print(f"ranks={size} all_ok={all(ok for _, ok in checks)} sha256={digest.hexdigest()}")
sys.exit(0 if all(ok for _, ok in checks) else 1)

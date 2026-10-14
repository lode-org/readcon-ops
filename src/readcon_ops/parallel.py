"""MPI versions of the batch kernels, over a caller's mpi4py communicator.

Every function takes ``comm``; mpi4py is only touched through that object,
so this module imports without MPI. The library never initialises MPI.

Determinism rules:

- Frame work is block-partitioned: rank ``r`` of ``P`` gets a contiguous
  block, the remainder going to low ranks. Results are gathered in rank
  order, which is frame order.
- All-pairs work is cut into square tiles of the upper triangle, dealt
  block-cyclically (tile ``k`` to rank ``k % P``). Each entry is computed
  by one rank with the serial kernel, so the matrix has the same bits for
  every ``P``.
- No float sum uses ``MPI_Allreduce``. The float path gathers the values
  in rank order and sums them once, which equals the serial sum. With
  ``exact=True`` each rank sends a fixed-size exact partial (1,665 bytes
  per quantity) and the merged sum is rounded once, which equals the
  serial exact value for every ``P`` and any input order.
"""

from __future__ import annotations

import numpy as np

from readcon_ops import kernels


def block(n: int, size: int, rank: int) -> range:
    """Contiguous block of ``n`` items owned by ``rank`` of ``size``."""
    base, rem = divmod(n, size)
    start = rank * base + min(rank, rem)
    return range(start, start + base + (1 if rank < rem else 0))


def _rs(comm):
    return (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())


def align_all(reference, frames, masses=None, exact: bool = False, comm=None):
    """:func:`readcon_ops.kernels.align_all` with frames split over ``comm``.

    Every rank passes the full frame list and gets the full result.
    """
    if comm is None:
        return kernels.align_all(reference, frames, masses, exact)
    rank, size = _rs(comm)
    mine = block(len(frames), size, rank)
    if len(mine):
        part = kernels.align_all(reference, [frames[i] for i in mine], masses, exact)
    else:
        part = (np.zeros(0), np.zeros((0, len(reference), 3)))
    got = comm.allgather(part)
    return np.concatenate([g[0] for g in got]), np.concatenate([g[1] for g in got])


def _tiles(m: int, tile: int):
    nt = -(-m // tile) if m else 0
    return [(a, b) for a in range(nt) for b in range(a, nt)]


def _tile_pairs(a: int, b: int, m: int, tile: int) -> np.ndarray:
    rows = range(a * tile, min(m, (a + 1) * tile))
    cols = range(b * tile, min(m, (b + 1) * tile))
    p = [(i, j) for i in rows for j in cols if i < j]
    return np.array(p, dtype=np.int64).reshape(-1, 2)


def rmsd_matrix(frames, masses=None, align: bool = True, exact: bool = False, comm=None, tile=64):
    """:func:`readcon_ops.kernels.rmsd_matrix` with tiles dealt over ``comm``."""
    if comm is None:
        return kernels.rmsd_matrix(frames, masses, align, exact)
    rank, size = _rs(comm)
    stack, fs = kernels._stack(frames)
    if any(f.periodic for f in fs):
        raise ValueError("rmsd_matrix takes free frames")
    m = len(fs)
    kernels._need(exact)
    w = kernels._w(masses, stack.shape[1] if m else 0, exact)
    mine = [t for k, t in enumerate(_tiles(m, tile)) if k % size == rank]
    pairs = [_tile_pairs(a, b, m, tile) for a, b in mine]
    pairs = np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)
    if kernels.HAVE_CORE:
        vals = kernels._core.pair_rmsds(stack, w, np.ascontiguousarray(pairs), align, exact)
    else:
        one = [kernels.rmsd_matrix([fs[i], fs[j]], masses, align)[0, 1] for i, j in pairs]
        vals = np.array(one)
    out = np.zeros((m, m))
    for p, v in comm.allgather((pairs, vals)):
        out[p[:, 0], p[:, 1]] = v
        out[p[:, 1], p[:, 0]] = v
    return out


def _sum(local: np.ndarray, comm, exact: bool, y=None, z=None) -> float:
    """Sum of ``local`` (or of ``local * y [* z]``) over all ranks."""
    x = np.ascontiguousarray(local, dtype=float).ravel()
    if exact:
        kernels._need(True)
        c = kernels._core
        if z is not None:
            part = c.exact_partial3(x, _f(y), _f(z))
        elif y is not None:
            part = c.exact_partial(x, _f(y))
        else:
            part = c.exact_partial(x)
        parts = [part] if comm is None else comm.allgather(part)
        return float(c.exact_round(parts))
    t = x if y is None else x * _f(y) if z is None else x * _f(y) * _f(z)
    if comm is None:
        return float(np.sum(t))
    return float(np.sum(np.concatenate(comm.allgather(t))))


def _f(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=float).ravel()


def band_mean(local_values, comm=None, exact: bool = False) -> float:
    """Mean of values split over ranks (each rank passes its own values)."""
    v = _f(local_values)
    n = len(v) if comm is None else comm.allreduce(len(v))
    return _sum(v, comm, exact) / n


def kinetic_energy(local_masses, local_velocities, comm=None, exact: bool = False) -> float:
    """``sum 0.5 m |v|^2`` over atoms split across ranks."""
    m = _f(local_masses)
    v = np.ascontiguousarray(local_velocities, dtype=float).reshape(-1, 3)
    half = np.repeat(0.5 * m, 3)
    vv = v.ravel()
    if exact:
        return _sum(half, comm, True, vv, vv)
    # Same expression and order as the serial numpy kinetic_energy.
    t = m * np.sum(v * v, axis=1)
    if comm is not None:
        t = np.concatenate(comm.allgather(t))
    return float(0.5 * np.sum(t))


def centroid(local_r, local_masses=None, comm=None, exact: bool = False) -> np.ndarray:
    """Weighted centroid of atoms split across ranks (equal weights if no masses)."""
    r = np.ascontiguousarray(local_r, dtype=float).reshape(-1, 3)
    n_total = len(r) if comm is None else comm.allreduce(len(r))
    if local_masses is None:
        m = np.ones(len(r))
    else:
        m = _f(local_masses)
    if exact:
        tot = _sum(m, comm, True) if local_masses is not None else float(n_total)
        w = m / tot
        return np.array([_sum(w, comm, True, r[:, d]) for d in range(3)])
    if comm is not None:
        r = np.concatenate(comm.allgather(r))
        m = np.concatenate(comm.allgather(m))
    w = kernels._weights(None if local_masses is None else m, len(r))
    return (w[:, None] * r).sum(axis=0)

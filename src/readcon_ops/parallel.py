"""MPI versions of the batch kernels, over a caller's mpi4py communicator.

Every function takes ``comm``; mpi4py is only touched through that object,
so this module imports without MPI. The library never initialises MPI.

Determinism rules:

- Frame work is block-partitioned: rank ``r`` of ``P`` gets a contiguous
  block, the remainder going to low ranks. Results are gathered in rank
  order, which is frame order.
- All-pairs work is split into contiguous row blocks of the upper triangle,
  cut so each rank owns nearly the same number of pairs (row ``i`` owns
  ``j > i``). Within a rank those rows run on rayon threads. Each entry is
  computed by one rank with the serial kernel, then the packed doubles are
  gathered in rank order, so the matrix has the same bits for every ``P``.
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


def _pairs_before(m: int, rows: int) -> int:
    """Upper-triangle pairs owned by rows ``[0, rows)``."""
    return rows * (2 * m - rows - 1) // 2


def row_block(m: int, size: int, rank: int) -> range:
    """Contiguous rows of the upper triangle for ``rank`` of ``size``.

    Row ``i`` owns pairs ``(i, j)`` with ``j > i``. The cut points are the
    rows where the pair count crosses ``rank * total / size``, so the pair
    counts differ by at most one row and the ranges partition ``0 .. m``.
    """
    if m <= 1 or size <= 0:
        return range(0)

    def rows_for(pairs: int) -> int:
        lo, hi = 0, m
        while lo < hi:
            mid = (lo + hi) // 2
            if _pairs_before(m, mid) < pairs:
                lo = mid + 1
            else:
                hi = mid
        return lo

    total = m * (m - 1) // 2
    return range(rows_for(rank * total // size), rows_for((rank + 1) * total // size))


def rmsd_matrix(frames, masses=None, align: bool = True, exact: bool = False, comm=None, tile=64):
    """:func:`readcon_ops.kernels.rmsd_matrix` with rows dealt over ``comm``.

    ``tile`` is accepted and ignored: the split is by balanced row blocks,
    not by a square tile size.
    """
    if tile < 1:
        raise ValueError("tile must be positive")
    if comm is None:
        return kernels.rmsd_matrix(frames, masses, align, exact)
    rank, size = _rs(comm)
    stack, fs = kernels._stack(frames)
    if any(f.periodic for f in fs):
        raise ValueError("rmsd_matrix takes free frames")
    m = len(fs)
    kernels._need(exact)
    w = kernels._w(masses, stack.shape[1] if m else 0, exact)
    if m <= 1:
        return np.zeros((m, m))
    rows = row_block(m, size, rank)
    i0, i1 = rows.start, rows.stop
    if kernels.HAVE_CORE:
        vals = np.asarray(kernels._core.rmsd_upper(stack, w, i0, i1, align, exact), dtype=np.float64)
    else:
        vals = np.array(
            [
                kernels.rmsd_matrix([fs[i], fs[j]], masses, align, exact)[0, 1]
                for i in range(i0, i1)
                for j in range(i + 1, m)
            ],
            dtype=np.float64,
        )
    # An empty numpy buffer is a null pointer; send one dummy double and a count of 0.
    nsend = int(vals.size)
    send = np.ascontiguousarray(vals, dtype=np.float64) if nsend else np.zeros(1, dtype=np.float64)
    counts = comm.allgather(nsend)
    recv = np.empty(sum(counts), dtype=np.float64)
    comm.Allgatherv([send, nsend], (recv, counts))
    spans = comm.allgather((i0, i1))
    out = np.zeros((m, m))
    offset = 0
    for (a, b), count in zip(spans, counts, strict=True):
        block = recv[offset : offset + count]
        k = 0
        for i in range(a, b):
            n = m - i - 1
            if n:
                out[i, i + 1 :] = block[k : k + n]
                out[i + 1 :, i] = block[k : k + n]
                k += n
        offset += count
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

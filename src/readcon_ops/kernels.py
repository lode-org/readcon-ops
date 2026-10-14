"""Batch kernels over many frames.

The compiled ``readcon-ops-core`` extension runs these with rayon threads
when it is installed. Without it, the same functions fall back to numpy.
Both paths give the same values within rounding. The compiled path gives
the same bits for any thread count.

``exact=True`` (compiled path only) makes every reduction exactly rounded:
centroids, the Kabsch covariance, RMSD, kinetic energy and band means are
the correctly rounded exact sum. A compensated pre-pass is kept when its
error bound proves that rounding; otherwise the sum uses a wide integer
accumulator. Those values do not depend on the atom order, the frame split,
the thread count or the MPI rank count. The default is the float path.
"""

from __future__ import annotations

import numpy as np

from readcon_ops.frames import as_frame
from readcon_ops.match import _superimpose, _weights, internal_motion

try:
    import readcon_ops_core as _core
except ImportError:  # pragma: no cover - exercised when the wheel is absent
    _core = None

HAVE_CORE = _core is not None
HAVE_EXACT = bool(HAVE_CORE and _core.HAS_EXACT)


def _need(exact: bool):
    if exact and not HAVE_EXACT:
        raise RuntimeError("exact=True needs readcon-ops-core built with the exact feature")
    return HAVE_CORE


def _c(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=float)


def _w(masses, n: int, exact: bool) -> np.ndarray:
    """Normalised weights. With ``exact`` the total is an exact sum, so the
    weights do not depend on the atom order either."""
    w = _weights(masses, n)
    if exact and masses is not None and n:
        m = _c(masses)
        w = m / _core.exact_round([_core.exact_partial(m)])
    return _c(w)


def _stack(frames) -> tuple[np.ndarray, list]:
    fs = [as_frame(f) for f in frames]
    if not fs:
        return np.zeros((0, 0, 3)), fs
    n = len(fs[0])
    if any(len(f) != n for f in fs):
        raise ValueError("frames have different atom counts")
    return _c(np.stack([f.r for f in fs])), fs


def align_all(reference, frames, masses=None, exact: bool = False):
    """Align every frame onto ``reference``.

    Returns ``(rmsd, moved)`` with shapes ``(m,)`` and ``(m, n, 3)``, in
    frame order. Free frames use the weighted Kabsch superposition of
    :func:`readcon_ops.internal_motion`. Periodic frames, and free frames
    whose covariance is rank 0 or 1, go through ``internal_motion``.
    """
    ref = as_frame(reference)
    stack, fs = _stack(frames)
    _need(exact)
    w = _w(masses, len(ref), exact)
    m = len(fs)
    rmsd = np.zeros(m)
    moved = np.zeros((m, len(ref), 3))
    free = [i for i, f in enumerate(fs) if not (f.periodic or ref.periodic)]
    done = set()
    if free and _need(exact):
        r, _rot, ok, mv = _core.kabsch_batch(_c(ref.r), _c(stack[free]), w, exact)
        for k, i in enumerate(free):
            if ok[k]:
                rmsd[i], moved[i] = r[k], mv[k]
                done.add(i)
    for i, f in enumerate(fs):
        if i in done:
            continue
        if exact and i in free:
            raise RuntimeError(f"exact=True has no rank-deficient Kabsch path; frame {i}")
        a = internal_motion(ref, f, masses)
        rmsd[i], moved[i] = a.rmsd, a.r
    return rmsd, moved


def rmsd_matrix(frames, masses=None, align: bool = True, exact: bool = False) -> np.ndarray:
    """Symmetric ``(m, m)`` RMSD matrix of free frames, Kabsch-aligned per pair."""
    stack, fs = _stack(frames)
    if any(f.periodic for f in fs):
        raise ValueError("rmsd_matrix takes free frames; use internal_motion for periodic ones")
    n = stack.shape[1] if stack.ndim == 3 else 0
    _need(exact)
    w = _w(masses, n, exact)
    if _need(exact):
        return _core.rmsd_matrix(stack, w, align, exact)
    m = len(fs)
    out = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            if align:
                moved, _, _ = _superimpose(stack[i], stack[j], w)
            else:
                moved = stack[j]
            d = moved - stack[i]
            out[i, j] = out[j, i] = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
    return out


def centroid(r, masses=None, exact: bool = False) -> np.ndarray:
    """Weighted centroid (center of mass when ``masses`` is given)."""
    r = _c(r)
    _need(exact)
    w = _w(masses, len(r), exact)
    if HAVE_CORE:
        return np.array(_core.centroid(r, w, exact))
    return (w[:, None] * r).sum(axis=0)


def kinetic_energy(masses, velocities, exact: bool = False) -> float:
    """``sum_i 0.5 m_i |v_i|^2`` in the units of the inputs."""
    m, v = _c(masses), _c(velocities)
    if _need(exact):
        return float(_core.kinetic_energy(m, v, exact))
    return float(0.5 * np.sum(m * np.sum(v * v, axis=1)))


def band_mean(values, exact: bool = False) -> float:
    """Mean of per-image values, for example NEB band energies."""
    x = _c(values).ravel()
    if _need(exact):
        return float(_core.mean(x, exact))
    return float(np.mean(x))


def pair_distances(r) -> np.ndarray:
    """``(n, n)`` atom distance matrix."""
    r = _c(r)
    if HAVE_CORE:
        return _core.pair_distances(r)
    d = r[:, None, :] - r[None, :, :]
    return np.sqrt(np.sum(d * d, axis=2))

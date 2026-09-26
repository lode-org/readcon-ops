"""Bijective match and rigid alignment.

A frame is any object with ``r`` (N, 3), ``box`` (3, 3), ``names``,
``copy``, and ``__len__``. A ``readcon.ConFrame`` adapter can satisfy that.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("readcon_ops")


def per_atom_distance(dr: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Euclidean norm of each row after the minimum-image wrap.

    The wrap is :mod:`minimage`, the same kernel vesin and linkcell use.
    """
    import minimage

    cell = minimage.Cell.from_vesin(np.asarray(box, dtype=float).reshape(3, 3).tolist())
    rows = np.atleast_2d(np.asarray(dr, dtype=float))
    wrapped = np.asarray(cell.wrap_many(rows), dtype=float)
    return np.sqrt(np.sum(wrapped**2.0, axis=1))


def rotational_match(frame_a, frame_b, epsilon_r: float, ira=None, atomic_number=None):
    """True when IRA's Hausdorff distance is inside ``epsilon_r``.

    ``ira`` is the eOn binding ``pyeonclient._core.ira_match`` or a callable
    with that signature: positions, atomic numbers, positions, atomic
    numbers, threshold, returning ``(hausdorff, error)``. ``None`` means
    IRA is not available. This is not a space-group test.
    """
    if ira is None:
        try:
            from pyeonclient import _core

            ira = getattr(_core, "ira_match", None)
        except Exception:
            ira = None
    if ira is None or atomic_number is None:
        return None
    z1 = [int(atomic_number(name)) for name in frame_a.names]
    z2 = [int(atomic_number(name)) for name in frame_b.names]
    hausdorff, err = ira(frame_a.r, z1, frame_b.r, z2, float(epsilon_r))
    if err != 0:
        return None
    return float(hausdorff) < float(epsilon_r)


def identical(frame_a, frame_b, epsilon_r: float) -> bool:
    """True when same-element atoms match within ``epsilon_r``.

    An atom already matched by index stays taken. A second atom cannot
    claim that site.
    """
    if len(frame_a) != len(frame_b):
        return False
    for i in range(3):
        for j in range(3):
            if abs(frame_a.box[i][j] - frame_b.box[i][j]) > 0.0001:
                logger.warning("boxes differ; identical is false")
                return False
    box = frame_a.box
    mismatch = []
    distances = per_atom_distance(frame_a.r - frame_b.r, box)
    for i, distance in enumerate(distances):
        if distance > epsilon_r:
            mismatch.append(i)
        elif frame_a.names[i] != frame_b.names[i]:
            return False
    used = {i for i in range(len(frame_a)) if i not in mismatch}
    for i in mismatch:
        distances = per_atom_distance(frame_a.r - frame_b.r[i], box)
        best = None
        best_d = 1e300
        for j, distance in enumerate(distances):
            if j in used:
                continue
            if (
                distance < epsilon_r
                and distance < best_d
                and frame_a.names[j] == frame_b.names[i]
            ):
                best = j
                best_d = distance
        if best is None:
            return False
        used.add(best)
    return True


def get_rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Name used by existing eOn callers."""
    return rotation_matrix(axis, theta)


def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    ct = np.cos(theta)
    st = np.sin(theta)
    one_minus = 1.0 - ct
    rx, ry, rz = axis
    return np.array(
        [
            [one_minus * rx * rx + ct, one_minus * ry * rx + rz * st, one_minus * rz * rx - ry * st],
            [one_minus * rx * ry - rz * st, one_minus * ry * ry + ct, one_minus * rz * ry + rx * st],
            [one_minus * rx * rz + ry * st, one_minus * ry * rz - rx * st, one_minus * rz * rz + ct],
        ]
    )


def rotate(coordinates: np.ndarray, axis: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    moved = coordinates.copy()
    if abs(angle) == 0.0:
        return moved
    matrix = rotation_matrix(axis, angle)
    center = center.copy()
    moved -= center
    moved = np.dot(moved, matrix)
    moved += center
    return moved


def internal_motion(frame_a, frame_b):
    """Return a copy of ``frame_b`` with rigid motion relative to ``frame_a`` removed."""
    moved = frame_b.copy()
    moved.r += frame_a.r[0] - frame_b.r[0]
    bond_a = (frame_a.r[1] - frame_a.r[0]) / np.linalg.norm(frame_a.r[1] - frame_a.r[0])
    bond_b = (moved.r[1] - moved.r[0]) / np.linalg.norm(moved.r[1] - moved.r[0])
    cross = np.cross(bond_b, bond_a)
    norm = np.linalg.norm(cross)
    if norm > 1e-12:
        theta = np.arccos(np.clip(np.dot(bond_a, bond_b), -1.0, 1.0))
        moved.r = rotate(moved.r, cross / norm, frame_a.r[0], theta)
    return moved

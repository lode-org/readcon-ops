"""Bijective match and rigid alignment.

Correspondence for :func:`internal_motion` is by atom index.
:func:`identical` may rematch atoms of the same name inside ``epsilon_r``.
"""

from __future__ import annotations

import logging

import numpy as np

from readcon_ops.frames import AlignedFrame, as_frame

logger = logging.getLogger("readcon_ops")

# Second singular value at or below this fraction of the first means the
# covariance has no unique rotation about its leading axis.
_AXIS_RANK_TOL = 1e-8
_RANK0_TOL = 1e-12


def _wrap_rows(rows: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Minimum-image form of each row, using minimage."""
    import minimage

    coords = np.atleast_2d(np.asarray(rows, dtype=float))
    if coords.size == 0:
        return np.zeros((0, 3))
    cell = minimage.Cell.from_vesin(np.asarray(box, dtype=float).reshape(3, 3).tolist())
    origin = [0.0, 0.0, 0.0]
    wrapped = [cell.displacement(origin, [float(x) for x in row]) for row in coords]
    return np.asarray(wrapped, dtype=float)


def per_atom_distance(dr: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Euclidean norm of each row after the minimum-image wrap.

    The wrap is :mod:`minimage`, the same kernel vesin and linkcell use.
    ``dr`` is a displacement, or a stack of them, in angstrom.
    """
    wrapped = _wrap_rows(dr, box)
    return np.sqrt(np.sum(wrapped**2.0, axis=1))


def rotational_match(frame_a, frame_b, epsilon_r: float, ira=None, atomic_number=None):
    """True when IRA's Hausdorff distance is inside ``epsilon_r``.

    ``ira`` is the eOn binding ``pyeonclient._core.ira_match`` or a callable
    with that signature: positions, atomic numbers, positions, atomic
    numbers, threshold, returning ``(hausdorff, error)``. ``None`` means
    IRA is not available. This calls neither spglib nor a space-group test.
    Both frames may be pyeonclient objects or ``readcon.ConFrame`` values.
    """
    frame_a = as_frame(frame_a)
    frame_b = as_frame(frame_b)
    if ira is None:
        try:
            from pyeonclient import _core

            ira = getattr(_core, "ira_match", None)
        except ImportError:
            ira = None
    if ira is None or atomic_number is None:
        return None
    z1 = [int(atomic_number(name)) for name in frame_a.names]
    z2 = [int(atomic_number(name)) for name in frame_b.names]
    hausdorff, err = ira(frame_a.r, z1, frame_b.r, z2, float(epsilon_r))
    if err != 0:
        return None
    return float(hausdorff) < float(epsilon_r)


def _symmetry_field(dataset, name):
    if isinstance(dataset, dict):
        return dataset.get(name)
    return getattr(dataset, name, None)


def spacegroup(frame, atomic_number, symprec: float = 1e-5):
    """International symbol, number, and Hall number for a periodic crystal.

    Positions are Cartesian. ``frame.box`` holds the lattice vectors as
    rows. The call is ``spglib.get_symmetry_dataset``. The international
    number is shared by every setting of a group. ``hall_number`` is the
    setting: the hexagonal-axes and rhombohedral-axes forms of one
    rhombohedral group are different Hall numbers. The label comes from
    the atoms and the cell. Lengths and angles alone are not a space
    group. This does not decide whether two clusters match.
    """
    import spglib

    frame = as_frame(frame)
    lattice = np.asarray(frame.box, dtype=float).reshape(3, 3)
    cartesian = np.asarray(frame.r, dtype=float)
    fractional = np.dot(cartesian, np.linalg.inv(lattice))
    numbers = [int(atomic_number(name)) for name in frame.names]
    dataset = spglib.get_symmetry_dataset((lattice, fractional, numbers), symprec=symprec)
    if dataset is None:
        return None
    international = _symmetry_field(dataset, "international")
    number = _symmetry_field(dataset, "number")
    hall_number = _symmetry_field(dataset, "hall_number")
    if number is None or hall_number is None:
        return None
    result = {
        "international": str(international),
        "number": int(number),
        "hall_number": int(hall_number),
    }
    hall = _symmetry_field(dataset, "hall")
    if hall:
        result["hall"] = str(hall)
    choice = _symmetry_field(dataset, "choice")
    if choice:
        result["choice"] = str(choice)
    return result


def identical(frame_a, frame_b, epsilon_r: float) -> bool:
    """True when same-element atoms match within ``epsilon_r``.

    An atom already matched by index stays taken. A second atom cannot
    claim that site. Distances are minimum-image distances in ``frame_a``'s
    cell. Both frames may be pyeonclient objects or ``readcon.ConFrame``
    values.
    """
    frame_a = as_frame(frame_a)
    frame_b = as_frame(frame_b)
    if len(frame_a) != len(frame_b):
        return False
    if np.max(np.abs(frame_a.box - frame_b.box)) > 0.0001:
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
            if distance < epsilon_r and distance < best_d and frame_a.names[j] == frame_b.names[i]:
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
    """Right-multiplier that rotates row coordinates about ``axis``."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    ct = np.cos(theta)
    st = np.sin(theta)
    one_minus = 1.0 - ct
    rx, ry, rz = axis
    return np.array(
        [
            [
                one_minus * rx * rx + ct,
                one_minus * ry * rx + rz * st,
                one_minus * rz * rx - ry * st,
            ],
            [
                one_minus * rx * ry - rz * st,
                one_minus * ry * ry + ct,
                one_minus * rz * ry + rx * st,
            ],
            [
                one_minus * rx * rz + ry * st,
                one_minus * ry * rz - rx * st,
                one_minus * rz * rz + ct,
            ],
        ]
    )


def rotate(
    coordinates: np.ndarray, axis: np.ndarray, center: np.ndarray, angle: float
) -> np.ndarray:
    """Rotate row coordinates about ``axis`` through ``center``."""
    moved = np.array(coordinates, dtype=float, copy=True)
    if abs(angle) == 0.0:
        return moved
    matrix = rotation_matrix(axis, angle)
    center = np.array(center, dtype=float, copy=True)
    moved -= center
    moved = np.dot(moved, matrix)
    moved += center
    return moved


def _weights(masses, count: int) -> np.ndarray:
    if count == 0:
        return np.zeros((0,))
    if masses is None:
        return np.full(count, 1.0 / count)
    weights = np.asarray(masses, dtype=float)
    if weights.shape != (count,):
        raise ValueError("masses must have one value per atom")
    if not np.isfinite(weights).all():
        raise ValueError("masses must be finite")
    if np.any(weights < 0.0):
        raise ValueError("masses must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("masses must sum to a positive value")
    return weights / total


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _unit_perpendicular(axis: np.ndarray) -> np.ndarray:
    helper = np.array([1.0, 0.0, 0.0]) if abs(float(axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = np.cross(axis, helper)
    return perp / np.linalg.norm(perp)


def _column_rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Column rotation ``R`` with ``R @ source = target`` for unit vectors."""
    cross = np.cross(source, target)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    sine = float(np.linalg.norm(cross))
    if sine < 1e-12:
        if cosine > 0.0:
            return np.eye(3)
        return rotation_matrix(_unit_perpendicular(source), np.pi).T
    skew = _skew(cross)
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def _row_rotation(reference: np.ndarray, mobile: np.ndarray, weights: np.ndarray):
    """Right-multiplier aligning centered ``mobile`` rows onto ``reference``.

    The multiplier is ``None`` when the covariance has rank 0. The second
    return is the unit axis whose spin was left undefined, or ``None`` when
    the rotation is fixed by the determinant-sign guard.
    """
    covariance = (reference * weights[:, None]).T @ mobile
    left, singular, vt = np.linalg.svd(covariance)
    if singular[0] <= _RANK0_TOL:
        return None, None
    if singular[1] <= _AXIS_RANK_TOL * singular[0]:
        target = left[:, 0]
        source = vt[0, :]
        target = target / np.linalg.norm(target)
        source = source / np.linalg.norm(source)
        column = _column_rotation_between(source, target)
        return column.T, target
    vt = np.array(vt, dtype=float, copy=True)
    if np.linalg.det(left @ vt) < 0.0:
        vt[-1, :] *= -1.0
    column = left @ vt
    return column.T, None


def _rmsd(moved: np.ndarray, reference: np.ndarray) -> float:
    if moved.shape[0] == 0:
        return 0.0
    delta = moved - reference
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _wrap_periodic(rows: np.ndarray, box: np.ndarray, pbc) -> np.ndarray:
    """Wrap rows into the cell along the periodic axes only.

    All three axes use :func:`_wrap_rows`. A mixed ``pbc`` leaves the open
    axes Cartesian: those directions have no minimum image.
    """
    coords = np.atleast_2d(np.asarray(rows, dtype=float))
    if coords.size == 0:
        return np.zeros((0, 3))
    if all(pbc):
        return _wrap_rows(coords, box)
    lattice = np.asarray(box, dtype=float)
    fractional = coords @ np.linalg.inv(lattice)
    for axis, flag in enumerate(pbc):
        if flag:
            fractional[:, axis] -= np.round(fractional[:, axis])
    return fractional @ lattice


def _mean_min_image(reference, mobile, box, pbc, weights: np.ndarray):
    wrapped = _wrap_periodic(mobile - reference, box, pbc)
    if wrapped.shape[0] == 0:
        return np.zeros(3)
    return (weights[:, None] * wrapped).sum(axis=0)


def internal_motion(frame_a, frame_b, masses=None) -> AlignedFrame:
    """Remove rigid motion of ``frame_b`` relative to ``frame_a``.

    The superposition is the one used for path RMSDs by R. Goswami,
    MethodsX 2026, doi:10.1016/j.mex.2026.103851. Non-periodic frames
    rotate by the Kabsch method (W. Kabsch, Acta Crystallogr. A 32,
    922-923, 1976, doi:10.1107/S0567739476001873). Optional ``masses``
    weight the centroids and the covariance. The covariance SVD uses the
    determinant-sign guard, so the matrix is a proper rotation.

    A collinear pair has no unique spin about its axis. That axial
    rotation is reported on ``undefined_axis`` and is not chosen: the
    returned matrix is the rigid motion that maps one axis onto the other
    and adds no further spin. Rank-0 frames (one atom, or every atom on
    the same point) are translated only.

    Periodic frames, including a frame with no ``pbc`` (CON D12), lose
    only the mean minimum-image translation. A rotation is not a symmetry
    of a cell. Either frame being periodic selects that path. The cell is
    the periodic frame's box, or ``frame_a``'s box when both are periodic.
    An open axis is not wrapped.

    Correspondence is by index. The result is a copy. ``rmsd`` is the
    unweighted root-mean-square distance in angstrom.
    """
    reference = as_frame(frame_a)
    mobile = as_frame(frame_b)
    if len(reference) != len(mobile):
        raise ValueError("frames have different atom counts")
    weights = _weights(masses, len(reference))
    periodic = reference.periodic or mobile.periodic
    if periodic:
        if reference.periodic:
            box = reference.box
            pbc = reference.pbc
            if mobile.periodic and np.max(np.abs(reference.box - mobile.box)) > 0.0001:
                logger.warning("boxes differ; the reference cell is used")
        else:
            box = mobile.box
            pbc = mobile.pbc
        translation = _mean_min_image(reference.r, mobile.r, box, pbc, weights)
        moved = mobile.r - translation
        rotation = None
        undefined_axis = None
    else:
        moved, rotation, undefined_axis = _superimpose(reference.r, mobile.r, weights)
    return AlignedFrame(
        moved,
        mobile.box,
        mobile.names,
        mobile.masses,
        mobile.pbc,
        rotation,
        _rmsd(moved, reference.r),
        undefined_axis,
    )


def _superimpose(reference: np.ndarray, mobile: np.ndarray, weights: np.ndarray):
    count = reference.shape[0]
    if count == 0:
        return mobile.copy(), None, None
    centroid_reference = (weights[:, None] * reference).sum(axis=0)
    centroid_mobile = (weights[:, None] * mobile).sum(axis=0)
    centered_reference = reference - centroid_reference
    centered_mobile = mobile - centroid_mobile
    if count == 1:
        return centered_mobile + centroid_reference, None, None
    rotation, undefined_axis = _row_rotation(centered_reference, centered_mobile, weights)
    if rotation is None:
        return centered_mobile + centroid_reference, None, None
    return centered_mobile @ rotation + centroid_reference, rotation, undefined_axis

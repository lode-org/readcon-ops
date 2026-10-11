"""Frame adapter for the pyeonclient protocol and ``readcon.ConFrame``.

A pyeonclient frame has ``r`` ``(N, 3)``, ``box`` ``(3, 3)``, ``names``,
and ``__len__``. A ``readcon.ConFrame`` has ``cell``, ``angles``, and
``atoms`` (each atom has ``symbol``, ``x``, ``y``, ``z``, and optionally
``mass``). ``xyz`` is used when it is present. ``metadata`` may carry
``pbc`` and ``lattice_vectors``.

Missing ``pbc`` means periodic in all three directions. ``lattice_vectors``
are three row vectors and replace the length-angle cell when present.
Those are CON rules D12 and D4 (a zero-atom frame is legal). The cell
construction follows eOn ``client/ConFileIO.cpp`` and ASE
``cellpar_to_cell``: ``a`` along x, ``b`` in the xy plane, angles in
degrees with alpha = angle(b, c), beta = angle(a, c), gamma = angle(a, b).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Frame:
    """Positions, cell, names, masses, and periodic flags.

    ``r`` is ``(N, 3)`` Cartesian angstrom. ``box`` is ``(3, 3)`` with
    lattice vectors as rows. ``pbc`` is three bools. ``periodic`` is true
    when any flag is true.
    """

    def __init__(self, r, box, names, masses, pbc):
        self.r = np.array(r, dtype=float, copy=True)
        self.box = np.array(box, dtype=float, copy=True).reshape(3, 3)
        self.names = [str(name) for name in names]
        self.masses = None if masses is None else np.array(masses, dtype=float, copy=True)
        self.pbc = (bool(pbc[0]), bool(pbc[1]), bool(pbc[2]))

    def __len__(self) -> int:
        return int(self.r.shape[0])

    @property
    def periodic(self) -> bool:
        return any(self.pbc)

    def copy(self) -> Frame:
        return Frame(self.r, self.box, self.names, self.masses, self.pbc)


class AlignedFrame(Frame):
    """Frame with rigid motion removed, plus the alignment report.

    ``rotation`` is the matrix applied on the right to centered mobile
    rows, or ``None`` when no rotation was applied. ``undefined_axis`` is
    a unit vector when a further rotation about that axis leaves the RMSD
    unchanged and was not chosen. ``rmsd`` is the unweighted
    root-mean-square distance in angstrom.
    """

    def __init__(self, r, box, names, masses, pbc, rotation, rmsd, undefined_axis):
        super().__init__(r, box, names, masses, pbc)
        if rotation is None:
            self.rotation = None
        else:
            self.rotation = np.array(rotation, dtype=float, copy=True)
        self.rmsd = float(rmsd)
        if undefined_axis is None:
            self.undefined_axis = None
        else:
            self.undefined_axis = np.array(undefined_axis, dtype=float, copy=True)

    def copy(self) -> AlignedFrame:
        return AlignedFrame(
            self.r,
            self.box,
            self.names,
            self.masses,
            self.pbc,
            self.rotation,
            self.rmsd,
            self.undefined_axis,
        )


def as_frame(obj: Any) -> Frame:
    """Return a :class:`Frame` for a protocol object or a ``ConFrame``."""
    if isinstance(obj, Frame):
        return obj.copy()
    if _is_protocol(obj):
        return _from_protocol(obj)
    if _is_conframe(obj):
        return _from_conframe(obj)
    raise TypeError("expected a pyeonclient frame or a readcon.ConFrame")


def _is_protocol(obj: Any) -> bool:
    return all(hasattr(obj, name) for name in ("r", "box", "names"))


def _is_conframe(obj: Any) -> bool:
    return all(hasattr(obj, name) for name in ("cell", "angles", "atoms"))


def _positions(array, what: str) -> np.ndarray:
    coords = np.asarray(array, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{what} must have shape (N, 3)")
    if coords.size and not np.isfinite(coords).all():
        raise ValueError(f"{what} must be finite")
    return np.array(coords, dtype=float, copy=True)


def _box(array) -> np.ndarray:
    box = np.asarray(array, dtype=float)
    if box.shape != (3, 3):
        raise ValueError("box must have shape (3, 3)")
    if not np.isfinite(box).all():
        raise ValueError("box must be finite")
    return box


def _names(names, count: int) -> list[str]:
    labels = [str(name) for name in names]
    if len(labels) != count:
        raise ValueError("names must have one entry per atom")
    return labels


def _parse_pbc(value) -> tuple[bool, bool, bool]:
    try:
        flags = list(value)
    except TypeError as exc:
        raise ValueError("pbc must have three bool entries") from exc
    if len(flags) != 3:
        raise ValueError("pbc must have three bool entries")
    parsed: list[bool] = []
    for flag in flags:
        if type(flag) is bool or isinstance(flag, np.bool_):
            parsed.append(bool(flag))
        else:
            raise ValueError("pbc entries must be bool")
    return (parsed[0], parsed[1], parsed[2])


def _pbc_from(obj, metadata) -> tuple[bool, bool, bool]:
    if metadata is not None and "pbc" in metadata and metadata["pbc"] is not None:
        return _parse_pbc(metadata["pbc"])
    attr = getattr(obj, "pbc", None)
    if attr is not None:
        return _parse_pbc(attr)
    return (True, True, True)


def _optional_masses(values, count: int) -> np.ndarray | None:
    if values is None:
        return None
    masses = np.asarray(values, dtype=float)
    if masses.shape != (count,):
        raise ValueError("masses must have one value per atom")
    if masses.size and not np.isfinite(masses).all():
        raise ValueError("masses must be finite")
    return masses


def lengths_angles_to_cell(lengths, angles_deg) -> np.ndarray:
    """Row-vector cell from lengths and alpha, beta, gamma in degrees."""
    lengths = np.asarray(lengths, dtype=float).reshape(3)
    angles = np.asarray(angles_deg, dtype=float).reshape(3)
    if not np.isfinite(lengths).all() or not np.isfinite(angles).all():
        raise ValueError("cell lengths and angles must be finite")
    if np.any(lengths <= 0.0):
        raise ValueError("cell lengths must be positive")
    if bool(np.all(angles == 90.0)):
        return np.diag(lengths)
    alpha, beta, gamma = np.deg2rad(angles)
    sin_gamma = float(np.sin(gamma))
    if abs(sin_gamma) < 1e-12:
        raise ValueError("cell angle gamma is degenerate")
    a_len, b_len, c_len = (float(lengths[0]), float(lengths[1]), float(lengths[2]))
    row_a = np.array([a_len, 0.0, 0.0])
    row_b = np.array([b_len * float(np.cos(gamma)), b_len * sin_gamma, 0.0])
    cx = c_len * float(np.cos(beta))
    cy = c_len * (float(np.cos(alpha)) - float(np.cos(beta)) * float(np.cos(gamma))) / sin_gamma
    cz2 = c_len * c_len - cx * cx - cy * cy
    if cz2 < -1e-8:
        raise ValueError("cell angles are inconsistent")
    row_c = np.array([cx, cy, float(np.sqrt(max(cz2, 0.0)))])
    return np.vstack((row_a, row_b, row_c))


def _lattice_vectors(value) -> np.ndarray:
    rows = np.asarray(value, dtype=float)
    if rows.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3)")
    if not np.isfinite(rows).all():
        raise ValueError("lattice_vectors must be finite")
    return rows


def _from_protocol(obj) -> Frame:
    coords = _positions(obj.r, "r")
    count = coords.shape[0]
    names = _names(obj.names, count)
    masses = _optional_masses(getattr(obj, "masses", None), count)
    return Frame(coords, _box(obj.box), names, masses, _pbc_from(obj, None))


def _from_conframe(obj) -> Frame:
    metadata = getattr(obj, "metadata", None)
    if metadata is None:
        metadata = {}
    atoms = list(obj.atoms)
    count = len(atoms)
    if hasattr(obj, "xyz"):
        coords = _positions(obj.xyz, "xyz")
        if coords.shape[0] != count:
            raise ValueError("xyz and atoms have different lengths")
    elif count == 0:
        coords = np.zeros((0, 3))
    else:
        rows = [[float(atom.x), float(atom.y), float(atom.z)] for atom in atoms]
        coords = _positions(rows, "atoms")
    names: list[str] = []
    mass_values: list[float] = []
    masses_known = True
    for atom in atoms:
        symbol = getattr(atom, "symbol", None)
        if symbol is None:
            symbol = getattr(atom, "name", None)
        if symbol is None:
            raise ValueError("ConFrame atoms need a symbol")
        names.append(str(symbol))
        mass = getattr(atom, "mass", None)
        if mass is None:
            masses_known = False
        else:
            mass_values.append(float(mass))
    masses = _optional_masses(mass_values, count) if masses_known else None
    if "lattice_vectors" in metadata and metadata["lattice_vectors"] is not None:
        box = _lattice_vectors(metadata["lattice_vectors"])
    else:
        box = lengths_angles_to_cell(obj.cell, obj.angles)
    return Frame(coords, box, names, masses, _pbc_from(obj, metadata))

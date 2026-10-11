import numpy as np
import pytest

from readcon_ops import (
    as_frame,
    identical,
    internal_motion,
    rotational_match,
    spacegroup,
)
from readcon_ops.frames import lengths_angles_to_cell


class Atom:
    def __init__(self, symbol, x, y, z, mass=None):
        self.symbol = symbol
        self.x = x
        self.y = y
        self.z = z
        self.mass = mass


class ConFrame:
    def __init__(self, cell, angles, atoms, metadata=None, xyz=None):
        self.cell = cell
        self.angles = angles
        self.atoms = atoms
        self.metadata = {} if metadata is None else metadata
        if xyz is not None:
            self.xyz = np.asarray(xyz, dtype=float)

    def __len__(self):
        return len(self.atoms)


def test_conframe_orthorhombic_cell_and_missing_pbc():
    atoms = [Atom("Cu", 0.0, 0.0, 0.0, mass=63.5), Atom("H", 1.0, 0.0, 0.0, mass=1.0)]
    frame = as_frame(ConFrame((10.0, 12.0, 14.0), (90.0, 90.0, 90.0), atoms))
    assert frame.names == ["Cu", "H"]
    assert np.allclose(frame.box, np.diag([10.0, 12.0, 14.0]))
    assert frame.pbc == (True, True, True)
    assert np.allclose(frame.masses, [63.5, 1.0])
    assert np.allclose(frame.r, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_lattice_vectors_replace_the_length_angle_cell():
    atoms = [Atom("Cu", 0.0, 0.0, 0.0)]
    lattice = [[5.0, 0.0, 0.0], [1.0, 6.0, 0.0], [0.0, 0.0, 7.0]]
    metadata = {"lattice_vectors": lattice, "pbc": [False, False, False]}
    frame = as_frame(ConFrame((10.0, 10.0, 10.0), (90.0, 90.0, 90.0), atoms, metadata))
    assert np.allclose(frame.box, lattice)
    assert frame.pbc == (False, False, False)


def test_skewed_cell_matches_the_eon_row_vectors():
    cell = lengths_angles_to_cell((10.0, 12.0, 14.0), (90.0, 90.0, 60.0))
    assert np.allclose(cell[0], [10.0, 0.0, 0.0])
    assert np.allclose(cell[1], [6.0, 12.0 * np.sin(np.deg2rad(60.0)), 0.0])
    assert np.allclose(cell[2], [0.0, 0.0, 14.0])


def test_conframe_kabsch_and_periodic_translation():
    atoms_a = [Atom("Cu", 0.0, 0.0, 0.0), Atom("Cu", 1.0, 0.0, 0.0), Atom("Cu", 0.2, 0.9, 0.1)]
    reference = ConFrame(
        (20.0, 20.0, 20.0),
        (90.0, 90.0, 90.0),
        atoms_a,
        metadata={"pbc": [False, False, False]},
    )
    rotated = as_frame(reference).r @ np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    atoms_b = [
        Atom("Cu", rotated[0, 0], rotated[0, 1], rotated[0, 2]),
        Atom("Cu", rotated[1, 0], rotated[1, 1], rotated[1, 2]),
        Atom("Cu", rotated[2, 0], rotated[2, 1], rotated[2, 2]),
    ]
    mobile = ConFrame(
        (20.0, 20.0, 20.0),
        (90.0, 90.0, 90.0),
        atoms_b,
        metadata={"pbc": [False, False, False]},
        xyz=rotated,
    )
    aligned = internal_motion(reference, mobile)
    assert aligned.rmsd < 1e-12

    crystal = ConFrame((10.0, 10.0, 10.0), (90.0, 90.0, 90.0), atoms_a)
    shifted = ConFrame(
        (10.0, 10.0, 10.0),
        (90.0, 90.0, 90.0),
        [Atom(atom.symbol, atom.x + 0.2, atom.y, atom.z) for atom in atoms_a],
    )
    periodic = internal_motion(crystal, shifted)
    assert periodic.rotation is None
    assert np.allclose(periodic.r, as_frame(crystal).r)


def test_spacegroup_accepts_a_conframe():
    lattice = 3.6 * np.eye(3)
    fractional = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    cartesian = fractional @ lattice
    atoms = [Atom("Cu", row[0], row[1], row[2]) for row in cartesian]
    frame = ConFrame((3.6, 3.6, 3.6), (90.0, 90.0, 90.0), atoms)
    info = spacegroup(frame, atomic_number=lambda name: 29)
    assert info["number"] == 225
    assert info["international"] == "Fm-3m"
    assert info["hall_number"] > 0


def test_rotational_match_accepts_a_conframe():
    atoms = [Atom("Cu", 0.0, 0.0, 0.0), Atom("Cu", 1.0, 0.0, 0.0)]
    frame_a = ConFrame((10.0, 10.0, 10.0), (90.0, 90.0, 90.0), atoms)
    frame_b = ConFrame((10.0, 10.0, 10.0), (90.0, 90.0, 90.0), list(reversed(atoms)))
    calls = {}

    def ira(r1, z1, r2, z2, thresh):
        calls["n"] = (len(z1), len(z2), thresh)
        calls["z"] = (list(z1), list(z2))
        return 0.01, 0

    assert rotational_match(frame_a, frame_b, 0.1, ira=ira, atomic_number=lambda name: 29) is True
    assert calls["n"] == (2, 2, 0.1)
    assert calls["z"] == ([29, 29], [29, 29])


def test_identical_accepts_a_conframe():
    atoms_a = [Atom("Cu", 0.0, 0.0, 0.0), Atom("Cu", 3.0, 0.0, 0.0)]
    atoms_b = [Atom("Cu", 3.01, 0.0, 0.0), Atom("Cu", 0.01, 0.0, 0.0)]
    frame_a = ConFrame((40.0, 40.0, 40.0), (90.0, 90.0, 90.0), atoms_a)
    frame_b = ConFrame((40.0, 40.0, 40.0), (90.0, 90.0, 90.0), atoms_b)
    assert identical(frame_a, frame_b, 0.1) is True


def test_pbc_entries_must_be_bool():
    atoms = [Atom("Cu", 0.0, 0.0, 0.0)]
    frame = ConFrame((10.0, 10.0, 10.0), (90.0, 90.0, 90.0), atoms, metadata={"pbc": [1, 0, 0]})
    with pytest.raises(ValueError):
        as_frame(frame)


def test_zero_atom_conframe():
    frame = as_frame(ConFrame((8.0, 8.0, 8.0), (90.0, 90.0, 90.0), []))
    assert len(frame) == 0
    assert frame.r.shape == (0, 3)
    assert frame.pbc == (True, True, True)

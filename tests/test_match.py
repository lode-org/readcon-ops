import numpy as np

from readcon_ops import identical, internal_motion
from readcon_ops.match import per_atom_distance


class Frame:
    def __init__(self, positions):
        self.r = np.asarray(positions, dtype=float)
        self.box = np.eye(3) * 40.0
        self.names = ["Cu"] * len(positions)

    def __len__(self):
        return len(self.r)

    def copy(self):
        other = Frame(self.r.copy())
        other.box = self.box.copy()
        other.names = list(self.names)
        return other


def test_identical_rejects_a_reused_site():
    a = Frame([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    b = Frame([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    assert identical(a, b, 0.1) is False


def test_identical_accepts_a_swap():
    a = Frame([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    b = Frame([[3.01, 0.0, 0.0], [0.01, 0.0, 0.0]])
    assert identical(a, b, 0.1) is True


def test_fcc_copper_spacegroup_is_225():
    from readcon_ops import spacegroup

    lattice = 3.6 * np.eye(3)
    fractional = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    )
    frame = Frame(fractional @ lattice)
    frame.box = lattice
    info = spacegroup(frame, atomic_number=lambda name: 29)
    assert info["number"] == 225
    assert info["international"] == "Fm-3m"
    assert info["hall_number"] > 0


def test_primitive_fcc_is_not_called_rhombohedral():
    """A 60-degree primitive cell of fcc is still Fm-3m, number 225.

    The same metric looks rhombohedral. The space group comes from the
    atoms in that cell, not from the angles.
    """
    from readcon_ops import spacegroup

    a = 3.6
    lattice = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    frame = Frame([[0.0, 0.0, 0.0]])
    frame.box = lattice
    info = spacegroup(frame, atomic_number=lambda name: 29)
    assert info["number"] == 225
    assert info["number"] != 166
    assert info["hall_number"] > 0


def test_rotational_match_uses_the_supplied_ira():
    from readcon_ops import rotational_match

    a = Frame([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = Frame([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    calls = {}

    def ira(r1, z1, r2, z2, thresh):
        calls["n"] = (len(z1), len(z2), thresh)
        return 0.05, 0

    assert rotational_match(a, b, 0.1, ira=ira, atomic_number=lambda name: 29) is True
    assert calls["n"] == (2, 2, 0.1)
    assert rotational_match(a, b, 0.1, ira=lambda *args: (2.0, 0), atomic_number=lambda name: 29) is False
    assert rotational_match(a, b, 0.1, ira=lambda *args: (0.0, 1), atomic_number=lambda name: 29) is None
    assert rotational_match(a, b, 0.1) is None


def test_distance_matches_the_minimage_readme():
    dr = np.array([[9.2, 0.0, 0.0]])
    box = np.diag([10.0, 10.0, 10.0])
    distance = per_atom_distance(dr, box)
    assert abs(distance[0] ** 2 - 0.64) < 1e-9


def test_internal_motion_lands_the_first_atom():
    a = Frame([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    b = Frame([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    moved = internal_motion(a, b)
    assert np.allclose(moved.r[0], a.r[0])

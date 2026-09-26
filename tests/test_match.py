import numpy as np

from readcon_ops import identical, internal_motion


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


def test_internal_motion_lands_the_first_atom():
    a = Frame([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    b = Frame([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    moved = internal_motion(a, b)
    assert np.allclose(moved.r[0], a.r[0])

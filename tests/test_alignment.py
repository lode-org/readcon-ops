import numpy as np
import pytest

from readcon_ops import internal_motion, rotate
from readcon_ops.match import rotation_matrix


class Frame:
    def __init__(self, positions, pbc=(True, True, True), box=None):
        self.r = np.asarray(positions, dtype=float)
        self.box = np.eye(3) * 40.0 if box is None else np.asarray(box, dtype=float)
        self.names = ["Cu"] * len(self.r)
        self.pbc = tuple(pbc)

    def __len__(self):
        return len(self.r)


def _free(positions):
    return Frame(positions, pbc=(False, False, False))


def _rmsd(a, b):
    delta = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _chiral(seed):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(8, 3))
    points[0] = np.array([0.0, 0.0, 0.0])
    points[1] = np.array([1.4, 0.2, 0.0])
    points[2] = np.array([0.3, 1.1, 0.4])
    return points


@pytest.mark.parametrize("seed", range(30))
def test_random_rotation_recovers_rmsd(seed):
    rng = np.random.default_rng(seed)
    points = _chiral(seed)
    axis = rng.normal(size=3)
    angle = float(rng.uniform(-np.pi, np.pi))
    center = rng.normal(size=3)
    shift = rng.normal(size=3)
    moved = rotate(points, axis, center, angle) + shift
    aligned = internal_motion(_free(points), _free(moved))
    assert aligned.rmsd < 1e-12
    assert _rmsd(aligned.r, points) < 1e-12
    assert aligned.undefined_axis is None
    assert aligned.rotation is not None
    assert abs(np.linalg.det(aligned.rotation) - 1.0) < 1e-9


@pytest.mark.parametrize("seed", range(8))
def test_mass_weighted_rigid_motion_recovers_rmsd(seed):
    rng = np.random.default_rng(100 + seed)
    points = _chiral(seed)
    moved = rotate(points, rng.normal(size=3), rng.normal(size=3), 0.8) + rng.normal(size=3)
    masses = rng.uniform(0.2, 50.0, size=len(points))
    aligned = internal_motion(_free(points), _free(moved), masses=masses)
    assert aligned.rmsd < 1e-12


def test_masses_change_a_nonrigid_superposition():
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mobile = np.array([[0.2, -0.1, 0.0], [1.3, 0.4, 0.2], [-0.3, 0.7, 0.5]])
    uniform = internal_motion(_free(reference), _free(mobile))
    weighted = internal_motion(_free(reference), _free(mobile), masses=[1.0, 1.0, 80.0])
    assert not np.allclose(uniform.rotation, weighted.rotation)
    weights = np.array([1.0, 1.0, 80.0])
    weights = weights / weights.sum()
    assert np.allclose(
        (weights[:, None] * weighted.r).sum(axis=0), (weights[:, None] * reference).sum(axis=0)
    )


def test_antiparallel_bond_is_aligned():
    reference = _free([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mobile = _free([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    aligned = internal_motion(reference, mobile)
    assert aligned.rmsd < 1e-12
    assert aligned.undefined_axis is not None
    assert abs(abs(aligned.undefined_axis[0]) - 1.0) < 1e-8
    angle = np.arccos(np.clip((np.trace(aligned.rotation) - 1.0) / 2.0, -1.0, 1.0))
    assert abs(angle - np.pi) < 1e-8


def test_antiparallel_line_along_a_tilted_axis():
    base = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    tilted = rotate(base, np.array([0.2, 0.7, 0.4]), np.zeros(3), 0.9)
    centroid = tilted.mean(axis=0)
    flipped = 2.0 * centroid - tilted
    aligned = internal_motion(_free(tilted), _free(flipped + np.array([0.4, -0.2, 0.3])))
    assert aligned.rmsd < 1e-12
    assert aligned.undefined_axis is not None
    direction = tilted[-1] - tilted[0]
    direction = direction / np.linalg.norm(direction)
    assert abs(abs(np.dot(aligned.undefined_axis, direction)) - 1.0) < 1e-8


def test_collinear_noise_does_not_pick_a_spin():
    rng = np.random.default_rng(4)
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    flipped = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-2.5, 0.0, 0.0]])
    flipped = flipped + rng.normal(scale=1e-10, size=flipped.shape)
    aligned = internal_motion(_free(line), _free(flipped))
    assert aligned.undefined_axis is not None
    assert aligned.rmsd < 1e-8
    assert abs(abs(aligned.undefined_axis[0]) - 1.0) < 1e-6


def test_planar_triangle_has_a_defined_rotation():
    reference = _free([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.8, 0.0]])
    mobile = _free(rotate(reference.r, np.array([0.0, 1.0, 1.0]), np.array([0.1, 0.0, 0.2]), 1.1))
    aligned = internal_motion(reference, mobile)
    assert aligned.rmsd < 1e-12
    assert aligned.undefined_axis is None
    assert abs(np.linalg.det(aligned.rotation) - 1.0) < 1e-9


def test_determinant_sign_guard_rejects_a_reflection():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(6, 3))
    mirrored = points.copy()
    mirrored[:, 2] *= -1.0
    aligned = internal_motion(_free(points), _free(mirrored + np.array([1.0, -2.0, 0.5])))
    assert aligned.rotation is not None
    assert abs(np.linalg.det(aligned.rotation) - 1.0) < 1e-9
    assert aligned.undefined_axis is None


def test_coincident_atoms_are_translated_only():
    aligned = internal_motion(
        _free([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), _free([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    )
    assert aligned.rmsd < 1e-12
    assert aligned.rotation is None
    assert aligned.undefined_axis is None


def test_single_atom_and_empty_frames():
    one = internal_motion(_free([[0.0, 0.0, 0.0]]), _free([[1.5, -0.2, 0.4]]))
    assert one.rmsd < 1e-12
    assert one.rotation is None
    empty = internal_motion(_free(np.zeros((0, 3))), _free(np.zeros((0, 3))))
    assert empty.r.shape == (0, 3)
    assert empty.rmsd == 0.0


@pytest.mark.parametrize("seed", range(10))
def test_periodic_frames_do_not_rotate(seed):
    rng = np.random.default_rng(200 + seed)
    points = _chiral(seed)
    axis = rng.normal(size=3)
    angle = float(rng.uniform(0.4, 2.0))
    shift = rng.normal(scale=0.3, size=3)
    rotated = rotate(points, axis, points.mean(axis=0), angle) + shift
    reference = Frame(points, pbc=(True, True, True))
    mobile = Frame(rotated, pbc=(True, True, True))
    aligned = internal_motion(reference, mobile)
    assert aligned.rotation is None
    assert aligned.undefined_axis is None
    assert np.allclose(aligned.r - aligned.r.mean(axis=0), rotated - rotated.mean(axis=0))
    assert aligned.rmsd > 0.1


def test_periodic_mean_uses_the_minimum_image():
    box = np.diag([10.0, 10.0, 10.0])
    reference = Frame([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], pbc=(True, True, False), box=box)
    mobile = Frame([[9.5, 0.0, 0.0], [0.5, 0.0, 0.0]], pbc=(True, True, False), box=box)
    aligned = internal_motion(reference, mobile)
    assert aligned.rotation is None
    wrapped = aligned.r - reference.r
    wrapped[:, 0] -= 10.0 * np.round(wrapped[:, 0] / 10.0)
    assert np.allclose(wrapped.mean(axis=0), 0.0, atol=1e-12)


def test_open_direction_is_not_wrapped():
    box = np.diag([10.0, 10.0, 30.0])
    reference = Frame([[0.0, 0.0, 0.0]], pbc=(True, True, False), box=box)
    mobile = Frame([[0.0, 0.0, 20.0]], pbc=(True, True, False), box=box)
    aligned = internal_motion(reference, mobile)
    assert aligned.rotation is None
    assert np.allclose(aligned.r, [[0.0, 0.0, 0.0]])


def test_missing_pbc_is_periodic_and_does_not_rotate():
    class Bare:
        def __init__(self, positions):
            self.r = np.asarray(positions, dtype=float)
            self.box = np.eye(3) * 30.0
            self.names = ["Cu"] * len(self.r)

        def __len__(self):
            return len(self.r)

    points = _chiral(3)
    rotated = rotate(points, np.array([0.0, 0.0, 1.0]), np.zeros(3), 0.7)
    aligned = internal_motion(Bare(points), Bare(rotated))
    assert aligned.pbc == (True, True, True)
    assert aligned.rotation is None
    assert np.allclose(aligned.r - aligned.r.mean(0), rotated - rotated.mean(0))


def test_masses_must_be_finite_and_match_the_frame():
    reference = _free([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mobile = _free(reference.r + 0.2)
    with pytest.raises(ValueError):
        internal_motion(reference, mobile, masses=[1.0, 1.0])
    with pytest.raises(ValueError):
        internal_motion(reference, mobile, masses=[1.0, -1.0, 1.0])
    with pytest.raises(ValueError):
        internal_motion(reference, mobile, masses=[1.0, np.nan, 1.0])
    with pytest.raises(ValueError):
        internal_motion(reference, _free([[0.0, 0.0, 0.0]]))


def test_nonfinite_coordinates_are_rejected():
    bad = _free([[0.0, 0.0, 0.0], [1.0, np.inf, 0.0]])
    with pytest.raises(ValueError):
        internal_motion(_free([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), bad)


def test_rotation_matrix_matches_a_right_angle():
    matrix = rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
    point = np.array([[1.0, 0.0, 0.0]])
    assert np.allclose(point @ matrix, [[0.0, 1.0, 0.0]])

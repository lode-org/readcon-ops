from fractions import Fraction

import numpy as np
import pytest

from readcon_ops import internal_motion, kernels
from readcon_ops.frames import Frame

core = pytest.importorskip("readcon_ops_core")


def _free(r):
    return Frame(r, np.eye(3) * 40.0, ["C"] * len(r), None, (False, False, False))


def _cloud(seed, n=40):
    return np.random.default_rng(seed).normal(scale=3.0, size=(n, 3))


def _rot(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


@pytest.mark.parametrize("seed", range(10))
def test_align_all_matches_internal_motion(seed):
    ref = _cloud(seed)
    frames = [_free(ref @ _rot(seed + k) + k + 0.01 * _cloud(100 + k)) for k in range(5)]
    masses = np.linspace(1.0, 60.0, len(ref))
    rmsd, moved = kernels.align_all(_free(ref), frames, masses)
    for k, f in enumerate(frames):
        a = internal_motion(_free(ref), f, masses)
        assert rmsd[k] == pytest.approx(a.rmsd, rel=1e-10, abs=1e-12)
        np.testing.assert_allclose(moved[k], a.r, atol=1e-10)


def test_collinear_frame_falls_back_to_python():
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    rmsd, moved = kernels.align_all(_free(line), [_free(line @ _rot(3))])
    a = internal_motion(_free(line), _free(line @ _rot(3)))
    assert rmsd[0] == pytest.approx(a.rmsd, abs=1e-12)


def test_rmsd_matrix_matches_numpy_path():
    frames = [_free(_cloud(7) @ _rot(k) + 0.05 * _cloud(50 + k)) for k in range(6)]
    got = kernels.rmsd_matrix(frames)
    saved, kernels._core = kernels._core, None
    kernels.HAVE_CORE = False
    try:
        want = kernels.rmsd_matrix(frames)
    finally:
        kernels._core, kernels.HAVE_CORE = saved, True
    np.testing.assert_allclose(got, want, atol=1e-10)
    assert np.array_equal(got, got.T)


def test_min_image_and_distances_match_numpy():
    cell = np.diag([10.0, 12.0, 9.0])
    dr = np.random.default_rng(1).uniform(-20, 20, size=(200, 3))
    got = core.min_image(dr, cell, (True, False, True))
    f = dr @ np.linalg.inv(cell)
    f[:, [0, 2]] -= np.round(f[:, [0, 2]])
    np.testing.assert_allclose(got, f @ cell, atol=1e-12)
    r = _cloud(2, 30)
    d = r[:, None] - r[None]
    np.testing.assert_allclose(kernels.pair_distances(r), np.sqrt((d * d).sum(-1)), atol=1e-12)


def _exact_ref(xs):
    return float(sum((Fraction(x) for x in xs), Fraction(0)))


@pytest.mark.parametrize("seed", range(20))
def test_exact_sum_is_correctly_rounded(seed):
    rng = np.random.default_rng(seed)
    xs = rng.normal(size=500) * 10.0 ** rng.integers(-30, 30, size=500)
    xs[::7] = -xs[::5][: len(xs[::7])]
    assert kernels.band_mean(xs, exact=True) == _exact_ref(xs) / len(xs)
    parts = [core.exact_partial(np.ascontiguousarray(c)) for c in np.array_split(xs, 7)]
    assert core.exact_round(parts[::-1]) == _exact_ref(xs)


@pytest.mark.parametrize("seed", range(5))
def test_exact_reductions_ignore_atom_order(seed):
    rng = np.random.default_rng(seed)
    a, b, v = _cloud(seed, 301), _cloud(seed + 50, 301), _cloud(seed + 99, 301)
    m = rng.uniform(1.0, 200.0, size=301)
    p = rng.permutation(301)
    for exact in (True,):  # the float path is not order-invariant
        assert np.array_equal(kernels.centroid(a, m, exact), kernels.centroid(a[p], m[p], exact))
        assert kernels.kinetic_energy(m, v, exact) == kernels.kinetic_energy(m[p], v[p], exact)
        tot = core.exact_round([core.exact_partial(m)])
        w, wp = m / tot, np.ascontiguousarray(m[p] / tot)
        h1 = core.covariance(a, b, w, exact)
        assert np.array_equal(h1, core.covariance(a[p], b[p], wp, exact))
        r1, _ = kernels.align_all(_free(a), [_free(b)], m, exact=True)
        r2, _ = kernels.align_all(_free(a[p]), [_free(b[p])], m[p], exact=True)
        assert r1[0] == r2[0]


def test_exact_matches_fraction_kinetic_energy():
    rng = np.random.default_rng(4)
    m, v = rng.uniform(1, 100, 64), rng.normal(size=(64, 3))
    terms = (Fraction(mi) * Fraction(x) ** 2 / 2 for mi, row in zip(m, v) for x in row)
    want = float(sum(terms))
    assert kernels.kinetic_energy(m, v, exact=True) == want

import numpy as np
import pytest

from readcon_ops import kernels, parallel
from readcon_ops.frames import Frame

pytest.importorskip("readcon_ops_core")


def _free(r):
    return Frame(r, np.eye(3) * 40.0, ["C"] * len(r), None, (False, False, False))


@pytest.mark.parametrize("n,size", [(0, 3), (5, 8), (23, 4), (100, 7)])
def test_block_partition_covers_in_order(n, size):
    got = [i for r in range(size) for i in parallel.block(n, size, r)]
    assert got == list(range(n))


def test_tiles_cover_upper_triangle_once():
    seen = np.concatenate([parallel._tile_pairs(a, b, 23, 5) for a, b in parallel._tiles(23, 5)])
    want = {(i, j) for i in range(23) for j in range(i + 1, 23)}
    assert len(seen) == len(want) and {tuple(p) for p in seen} == want


def test_serial_calls_match_kernels():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(30, 3))
    frames = [_free(ref + 0.1 * rng.normal(size=(30, 3))) for _ in range(6)]
    for exact in (False, True):
        np.testing.assert_array_equal(
            parallel.rmsd_matrix(frames, exact=exact), kernels.rmsd_matrix(frames, exact=exact)
        )
    m, v = rng.uniform(1, 9, 30), rng.normal(size=(30, 3))
    assert parallel.kinetic_energy(m, v, exact=True) == kernels.kinetic_energy(m, v, exact=True)
    np.testing.assert_array_equal(
        parallel.centroid(ref, m, exact=True), kernels.centroid(ref, m, exact=True)
    )

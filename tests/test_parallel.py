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


def test_row_blocks_partition_the_triangle():
    m, size = 23, 4
    seen = []
    counts = []
    for rank in range(size):
        rows = parallel.row_block(m, size, rank)
        n = 0
        for i in rows:
            for j in range(i + 1, m):
                seen.append((i, j))
                n += 1
        counts.append(n)
    want = [(i, j) for i in range(m) for j in range(i + 1, m)]
    assert seen == want
    assert max(counts) - min(counts) <= m
    assert parallel.row_block(0, 3, 1) == range(0)
    assert parallel.row_block(4, 1, 0) == range(3)


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

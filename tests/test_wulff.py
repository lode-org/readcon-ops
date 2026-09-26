import numpy as np

from readcon_ops import wulff_vertices


def test_equal_energies_give_a_cube():
    normals = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    vertices = wulff_vertices(normals, np.ones(6))
    assert len(vertices) == 8
    assert np.allclose(np.abs(vertices), 1.0)

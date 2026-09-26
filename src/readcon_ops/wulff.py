"""Facet distances for a Wulff shape.

The distance of a facet from the center is its surface energy. A point
is inside the shape when it lies behind every facet plane. This module
returns the vertices of that polyhedron. It does not place atoms on the
shape. Atomic decoration belongs to a cluster builder.
"""

from __future__ import annotations

import numpy as np


def wulff_vertices(normals: np.ndarray, energies: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Vertices of the Wulff polyhedron.

    ``normals`` is ``(F, 3)`` and need not be a unit set. ``energies``
    is one positive surface energy per facet. The plane equation is
    ``n · x = energy`` after ``n`` is normalized.
    """
    planes = np.asarray(normals, dtype=float)
    gamma = np.asarray(energies, dtype=float)
    if planes.ndim != 2 or planes.shape[1] != 3:
        raise ValueError("normals must have shape (F, 3)")
    if gamma.shape != (planes.shape[0],):
        raise ValueError("energies must have one value per facet")
    if np.any(gamma <= 0.0):
        raise ValueError("surface energies must be positive")
    lengths = np.linalg.norm(planes, axis=1)
    unit = planes / lengths[:, None]
    found: list[np.ndarray] = []
    count = unit.shape[0]
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                matrix = np.stack((unit[i], unit[j], unit[k]))
                if abs(np.linalg.det(matrix)) < 1e-10:
                    continue
                point = np.linalg.solve(matrix, np.array([gamma[i], gamma[j], gamma[k]]))
                if np.all(unit @ point <= gamma + tol):
                    found.append(point)
    if not found:
        return np.zeros((0, 3))
    stacked = np.vstack(found)
    return np.unique(np.round(stacked, decimals=8), axis=0)

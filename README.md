# readcon-ops

Comparisons and rigid alignment of coordinate frames.

This is the third package in the readcon ecosystem:

| Package | Role |
|---|---|
| [readcon-core](https://github.com/lode-org/readcon-core) / `readcon` | Parse and write a frame. |
| [readcon-db](https://github.com/lode-org/readcon-db) / `readcon_db` | Store a corpus of those frames. |
| readcon-ops / `readcon_ops` | Compare two frames and remove rigid motion. |

A frame is a pyeonclient object (`r`, `box`, `names`) or a `readcon.ConFrame` (`cell`, `angles`, `atoms`). `as_frame` accepts both. The package does not open a file.

```python
from readcon_ops import as_frame, identical, internal_motion, spacegroup, wulff_vertices

same = identical(frame_a, frame_b, epsilon_r=0.1)
aligned = internal_motion(frame_a, frame_b)
group = spacegroup(frame_a, atomic_number=lambda name: 29)
vertices = wulff_vertices(normals, energies)
```

`identical` keeps an index-matched atom taken, so a second atom cannot claim that site. Distances use [minimage](https://github.com/lode-org/minimage) at commit `0545e172d88a887c6d9d8e3977e5c12fc1ba6f38`, the v0.1.1 tag, the same wrap as vesin and linkcell.

`spacegroup` calls spglib and returns the international symbol, the international number, and the Hall number.

`rotational_match` calls Iterative Rotations and Assignments (`pyeonclient._core.ira_match`) when that binding is present and returns `None` when it is not. It does not call spglib.

`internal_motion` superimposes a non-periodic frame with the Kabsch rotation. Optional masses weight the fit. A collinear frame reports the undefined axial rotation and does not choose one. A periodic frame, including a frame with no `pbc`, loses only the mean minimum-image translation. Pass `pbc=(False, False, False)` for a free cluster.

`wulff_vertices` returns the vertices of the polyhedron whose facet distance from the center is the surface energy.

Documentation sources are under `docs/orgmode` (tutorial, how-to, reference, explanation). Licence: BSD-3-Clause.

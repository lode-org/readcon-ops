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
from readcon_ops import (
    as_frame,
    identical,
    internal_motion,
    resolve_named_groups,
    spacegroup,
    wulff_vertices,
)

same = identical(frame_a, frame_b, epsilon_r=0.1)
aligned = internal_motion(frame_a, frame_b)
group = spacegroup(frame_a, atomic_number=lambda name: 29)
vertices = wulff_vertices(normals, energies)
groups = resolve_named_groups(
    reference,
    {"ligand": "resname LIG", "pocket": "resname POCK"},
    disjoint=True,
)
```

`identical` keeps an index-matched atom taken, so a second atom cannot claim that site. Distances use [minimage](https://github.com/lode-org/minimage) at commit `a78009f2af73821ebbd30a736b2b671e5b63ce5e`, minimage `main` on 2026-11-25, the same wrap as vesin and linkcell.

`spacegroup` calls spglib and returns the international symbol, the international number, and the Hall number.

`rotational_match` calls Iterative Rotations and Assignments (`pyeonclient._core.ira_match`) when that binding is present and returns `None` when it is not. It does not call spglib.

`internal_motion` superimposes a non-periodic frame with the Kabsch rotation. Optional masses weight the fit. A collinear frame reports the undefined axial rotation and does not choose one. A periodic frame, including a frame with no `pbc`, loses only the mean minimum-image translation. Pass `pbc=(False, False, False)` for a free cluster.

`resolve_named_groups` evaluates chemfiles selections on a reference frame. `rows` are 0-based indices in frame order and stay the group membership for later frames. `source_ids` is the pre-group index when a type-grouped CON import kept `atom_id`. Residue selectors use a chemfiles topology, or `metadata["chemfiles_residues"]`. `disjoint=True` rejects an empty or overlapping pair.

`wulff_vertices` returns the vertices of the polyhedron whose facet distance from the center is the surface energy.

Documentation sources are under `docs/orgmode` (tutorial, how-to, reference, explanation). Licence: BSD-3-Clause.

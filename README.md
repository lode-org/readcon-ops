# readcon-ops

Comparisons and rigid alignment of coordinate frames.

This is the third package in the readcon ecosystem:

| Package | Role |
|---|---|
| [readcon-core](https://github.com/lode-org/readcon-core) / `readcon` | Parse and write a frame. |
| [readcon-db](https://github.com/lode-org/readcon-db) / `readcon_db` | Store a corpus of those frames. |
| readcon-ops / `readcon_ops` | Compare two frames and remove rigid motion. |

A frame is any object with `r` `(N, 3)`, `box` `(3, 3)`, `names`, `copy`, and `__len__`. The package does not open a file.

```python
from readcon_ops import identical, internal_motion

same = identical(frame_a, frame_b, epsilon_r=0.1)
aligned = internal_motion(frame_a, frame_b)
```

`identical` keeps an index-matched atom taken, so a second atom cannot claim that site. Distances use [minimage](https://github.com/lode-org/minimage) v0.1.1, the same wrap as vesin and linkcell. `internal_motion` translates the first atom of the second frame onto the first atom of the reference, then skips a rotation when the bond is already parallel.

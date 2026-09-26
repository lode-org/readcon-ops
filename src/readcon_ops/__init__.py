"""Operations on coordinate frames.

``readcon`` parses and writes a frame. ``readcon-db`` stores a corpus of
those frames. This package compares two frames and removes rigid motion.
It does not read a file.
"""

from readcon_ops.match import (
    identical,
    internal_motion,
    rotate,
    rotational_match,
    spacegroup,
)

__all__ = [
    "identical",
    "internal_motion",
    "rotate",
    "rotational_match",
    "spacegroup",
]

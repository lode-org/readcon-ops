"""Operations on coordinate frames.

``readcon`` parses and writes a frame. ``readcon-db`` stores a corpus of
those frames. This package compares two frames and removes rigid motion.
It does not read a file. ``spacegroup`` calls spglib.
"""

from readcon_ops.frames import AlignedFrame, Frame, as_frame
from readcon_ops.match import (
    identical,
    internal_motion,
    rotate,
    rotational_match,
    spacegroup,
)
from readcon_ops.wulff import wulff_vertices

__all__ = [
    "AlignedFrame",
    "Frame",
    "as_frame",
    "identical",
    "internal_motion",
    "rotate",
    "rotational_match",
    "spacegroup",
    "wulff_vertices",
]

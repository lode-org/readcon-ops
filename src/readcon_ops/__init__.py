"""Operations on coordinate frames.

``readcon`` parses and writes a frame. ``readcon-db`` stores a corpus of
those frames. This package compares two frames and removes rigid motion.
It does not read a file. ``spacegroup`` calls spglib.
``resolve_named_groups`` evaluates a chemfiles selection on a reference frame.
"""

from readcon_ops.frames import AlignedFrame, Frame, as_frame
from readcon_ops.groups import (
    AtomGroup,
    DuplicateSourceId,
    EmptyGroup,
    GroupResolutionError,
    IncompleteSourceIds,
    MissingSelectionMetadata,
    OverlappingGroups,
    ResolvedGroups,
    SelectionContextError,
    SelectionSyntaxError,
    SourceIndex,
    resolve_named_groups,
)
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
    "AtomGroup",
    "DuplicateSourceId",
    "EmptyGroup",
    "Frame",
    "GroupResolutionError",
    "IncompleteSourceIds",
    "MissingSelectionMetadata",
    "OverlappingGroups",
    "ResolvedGroups",
    "SelectionContextError",
    "SelectionSyntaxError",
    "SourceIndex",
    "as_frame",
    "identical",
    "internal_motion",
    "resolve_named_groups",
    "rotate",
    "rotational_match",
    "spacegroup",
    "wulff_vertices",
]

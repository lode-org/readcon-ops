"""Named atom groups on one reference frame.

Comparisons use frame order: row ``i`` of ``frame.r`` is atom ``i``.
A group is resolved once, on the reference. The returned rows stay the
membership for later frames. They are not selected again.

The selection strings are evaluated by ``chemfiles.Selection``. Rows are
0-based indices in frame order. When the frame still has the identity
from before a type-grouped CON import, that identity is returned beside
each row. The two sequences differ after that import.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import chemfiles

from readcon_ops.frames import Frame, as_frame

_RESIDUE_KEYWORDS = frozenset({"resname", "resid", "resindex"})
_VALUE_KEYWORDS = frozenset({"name", "type"})


class GroupResolutionError(ValueError):
    """A named group could not be resolved."""


class MissingSelectionMetadata(GroupResolutionError):
    """A selector needs residue metadata this frame does not carry."""

    def __init__(self, selection: str, missing: str):
        self.selection = selection
        self.missing = missing
        super().__init__(
            f"selection {selection!r} needs {missing}, and this frame does not have it"
        )


class EmptyGroup(GroupResolutionError):
    """A group required to be non-empty matched no atoms."""

    def __init__(self, name: str, selection: str):
        self.name = name
        self.selection = selection
        super().__init__(f"group {name!r} ({selection}) matched no atoms")


class OverlappingGroups(GroupResolutionError):
    """Two groups that must be disjoint share frame rows."""

    def __init__(self, left: str, right: str, rows: tuple[int, ...]):
        self.left = left
        self.right = right
        self.rows = rows
        super().__init__(f"groups {left!r} and {right!r} share frame rows {list(rows)}")


class SelectionSyntaxError(GroupResolutionError):
    """The selection string was rejected by the evaluator."""

    def __init__(self, selection: str, detail: str):
        self.selection = selection
        self.detail = detail
        super().__init__(f"selection {selection!r} was rejected: {detail}")


class SelectionContextError(GroupResolutionError):
    """The selection is not an atom-context selection."""

    def __init__(self, selection: str, size: int):
        self.selection = selection
        self.size = size
        super().__init__(
            f"selection {selection!r} has context size {size}; atom groups need context size 1"
        )


class IncompleteSourceIds(GroupResolutionError):
    """Some atoms carry a source identity and others do not."""

    def __init__(self):
        super().__init__("source identities are missing on some atoms")


class DuplicateSourceId(GroupResolutionError):
    """The same source identity is stored on two frame rows."""

    def __init__(self, source_id: int, first_row: int, row: int):
        self.source_id = source_id
        self.first_row = first_row
        self.row = row
        super().__init__(f"source identity {source_id} is on rows {first_row} and {row}")


class AtomGroup:
    """One named group on the reference frame.

    ``rows`` is a sorted tuple of 0-based indices into frame order.
    ``source_ids`` is parallel to ``rows`` when identities are available,
    and ``None`` otherwise. After a type-grouped CON import, ``rows[i]``
    is the stored row and ``source_ids[i]`` is the pre-group index.
    """

    def __init__(self, name: str, selection: str, rows: Sequence[int], source_ids):
        self.name = name
        self.selection = selection
        self.rows = tuple(int(row) for row in rows)
        self.source_ids = None if source_ids is None else tuple(int(value) for value in source_ids)


class SourceIndex:
    """Bijection between 0-based frame rows and source identities."""

    def __init__(self, source_of_row: Sequence[int]):
        self.source_of_row = tuple(int(value) for value in source_of_row)
        mapping: dict[int, int] = {}
        for row, source_id in enumerate(self.source_of_row):
            if source_id in mapping:
                raise DuplicateSourceId(source_id, mapping[source_id], row)
            mapping[source_id] = row
        self._row_of_source = mapping

    def row(self, source_id: int) -> int:
        """Frame row for one source identity."""
        try:
            return self._row_of_source[int(source_id)]
        except KeyError as exc:
            raise KeyError(f"source identity {int(source_id)} is not in this frame") from exc

    def source(self, row: int) -> int:
        """Source identity stored on one frame row."""
        return self.source_of_row[int(row)]

    def rows(self, source_ids: Sequence[int]) -> tuple[int, ...]:
        """Frame rows for source identities, in the given order."""
        return tuple(self.row(source_id) for source_id in source_ids)

    def sources(self, rows: Sequence[int]) -> tuple[int, ...]:
        """Source identities for frame rows, in the given order."""
        return tuple(self.source(row) for row in rows)


class ResolvedGroups:
    """Groups resolved on one reference frame.

    ``groups`` follows the insertion order of the selection mapping.
    ``source_index`` is ``None`` when the frame has no source identities.
    """

    def __init__(self, groups: Sequence[AtomGroup], source_index: SourceIndex | None):
        self.groups = tuple(groups)
        self.source_index = source_index
        self._by_name = {group.name: group for group in self.groups}

    def __getitem__(self, name: str) -> AtomGroup:
        return self._by_name[name]

    def __iter__(self):
        return iter(self.groups)

    def __len__(self) -> int:
        return len(self.groups)


def resolve_named_groups(frame, selections: Mapping[str, str], *, disjoint: bool = False):
    """Resolve named selections on a reference frame.

    ``selections`` maps a group name to a chemfiles selection string.
    The result's ``rows`` are 0-based indices into frame order. A later
    frame uses those same integers. This function does not look at that
    later frame.

    A ``chemfiles.Frame`` is evaluated directly, so a residue selector
    sees that frame's topology. Any other frame is projected in its own
    row order. ``metadata["chemfiles_residues"]`` is added to the
    projection when it is present. Those atom indices are frame rows,
    which is the order CON import writes after it groups atoms by type.

    ``source_ids`` on each group, and ``source_index`` on the result,
    are set when every atom has an ``atom_id`` or the frame has a
    ``source_ids`` array. They are ``None`` when no atom has one.
    A mixture raises ``IncompleteSourceIds``. A repeated id raises
    ``DuplicateSourceId``.

    ``disjoint=True`` is the call that requires a disjoint pair: every
    group must match at least one atom, and no two groups may share a
    row. The default returns an empty group and an overlapping pair as
    they matched.

    Residue selectors (``resname``, ``resid``, ``resindex``) raise
    ``MissingSelectionMetadata`` when the frame has no residue table.
    A selection that is not atom context raises ``SelectionContextError``.
    """
    _check_selections(selections)
    if isinstance(frame, chemfiles.Frame):
        evaluated = frame
        source_ids = None
    else:
        adapted = as_frame(frame)
        source_ids = _source_ids(frame, len(adapted))
        residues = _residue_entries(frame)
        metadata = getattr(frame, "metadata", None)
        evaluated = _project(adapted, source_ids, residues, metadata)
    residue_count = len(evaluated.topology.residues)
    groups = []
    for name, selection in selections.items():
        rows = _select_rows(evaluated, selection, residue_count)
        group_sources = None if source_ids is None else tuple(source_ids[row] for row in rows)
        groups.append(AtomGroup(name, selection, rows, group_sources))
    if disjoint:
        _require_disjoint(groups)
    index = None if source_ids is None else SourceIndex(source_ids)
    return ResolvedGroups(groups, index)


def _check_selections(selections: Mapping[str, str]) -> None:
    if isinstance(selections, (str, bytes)) or not isinstance(selections, Mapping):
        raise TypeError("selections must be a mapping of group name to selection")
    for name, selection in selections.items():
        if not isinstance(name, str) or name == "":
            raise ValueError("group names must be non-empty strings")
        if not isinstance(selection, str):
            raise TypeError("selections must be strings")


def _source_ids(obj, count: int) -> tuple[int, ...] | None:
    if isinstance(obj, Frame):
        if obj.source_ids is None:
            return None
        if len(obj.source_ids) != count:
            raise ValueError("source_ids must have one value per atom")
        return tuple(int(value) for value in obj.source_ids)
    atoms = getattr(obj, "atoms", None)
    if atoms is not None:
        atom_list = list(atoms)
        if len(atom_list) == count:
            from_atoms = _ids_from_atoms(atom_list)
            if from_atoms is not None:
                return from_atoms
    for name in ("source_ids", "atom_ids"):
        values = getattr(obj, name, None)
        if values is None or isinstance(values, (str, bytes)) or callable(values):
            continue
        try:
            length = len(values)
        except TypeError:
            continue
        if length != count:
            raise ValueError("source_ids must have one value per atom")
        if any(value is None for value in values):
            raise IncompleteSourceIds()
        return tuple(int(value) for value in values)
    return None


def _ids_from_atoms(atoms) -> tuple[int, ...] | None:
    present = False
    missing = False
    ids: list[int] = []
    for atom in atoms:
        value = getattr(atom, "atom_id", None)
        if value is None:
            missing = True
        else:
            present = True
            ids.append(int(value))
    if present and missing:
        raise IncompleteSourceIds()
    if not present:
        return None
    return tuple(ids)


def _residue_entries(obj) -> list:
    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, dict):
        residues = metadata.get("chemfiles_residues")
        if residues:
            return list(residues)
    return []


def _project(frame: Frame, source_ids, residues, metadata) -> chemfiles.Frame:
    projected = chemfiles.Frame()
    for row, symbol in enumerate(frame.names):
        source_id = None if source_ids is None else source_ids[row]
        name, atomic_type = _chemfiles_labels(metadata, source_id, symbol)
        if atomic_type == name:
            atom = chemfiles.Atom(name)
        else:
            atom = chemfiles.Atom(name, atomic_type)
        x, y, z = frame.r[row]
        projected.add_atom(atom, (float(x), float(y), float(z)))
    _add_residues(projected, residues, len(frame))
    return projected


def _chemfiles_labels(metadata, source_id: int | None, symbol: str) -> tuple[str, str]:
    """Display name and atomic type for one projected atom.

    Import sidecars are indexed by the pre-group identity, matching the
    chemfiles atom the CON row was copied from. The CON symbol is the
    fallback when a sidecar entry is absent.
    """
    name = symbol
    atomic_type = symbol
    if not isinstance(metadata, dict) or source_id is None or source_id < 0:
        return name, atomic_type
    names = metadata.get("chemfiles_atom_names")
    types = metadata.get("chemfiles_atom_types")
    if isinstance(names, (list, tuple)) and source_id < len(names) and names[source_id]:
        name = str(names[source_id])
    if isinstance(types, (list, tuple)) and source_id < len(types) and types[source_id]:
        atomic_type = str(types[source_id])
    return name, atomic_type


def _add_residues(frame: chemfiles.Frame, entries, count: int) -> None:
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("chemfiles_residues entries must be mappings")
        name = entry.get("name")
        if not name:
            raise ValueError("a residue needs a name")
        atoms = entry.get("atoms")
        if atoms is None:
            raise ValueError("a residue needs an atom list")
        residue_id = entry.get("id", None)
        if residue_id is None:
            residue = chemfiles.Residue(str(name))
        else:
            residue = chemfiles.Residue(str(name), int(residue_id))
        for atom in atoms:
            index = int(atom)
            if index < 0 or index >= count:
                raise ValueError("residue atom index is outside the frame")
            residue.atoms.append(index)
        try:
            frame.add_residue(residue)
        except chemfiles.ChemfilesError as exc:
            raise GroupResolutionError(str(exc)) from exc


def _select_rows(frame: chemfiles.Frame, selection: str, residue_count: int) -> tuple[int, ...]:
    try:
        parsed = chemfiles.Selection(selection)
    except chemfiles.ChemfilesError as exc:
        raise SelectionSyntaxError(selection, str(exc)) from exc
    size = int(parsed.size)
    if size != 1:
        raise SelectionContextError(selection, size)
    if _needs_residues(selection) and residue_count == 0:
        raise MissingSelectionMetadata(selection, "residues")
    try:
        matches = parsed.evaluate(frame)
    except chemfiles.ChemfilesError as exc:
        raise SelectionSyntaxError(selection, str(exc)) from exc
    return tuple(sorted({int(match) for match in matches}))


def _needs_residues(selection: str) -> bool:
    tokens = _tokens(selection)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith('"'):
            index += 1
            continue
        head = token.split("(", 1)[0]
        if head in _VALUE_KEYWORDS:
            index += 2
            continue
        if head in _RESIDUE_KEYWORDS:
            return True
        index += 1
    return False


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    size = len(text)
    index = 0
    while index < size:
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char == '"':
            end = text.find('"', index + 1)
            if end < 0:
                tokens.append(text[index:])
                break
            tokens.append(text[index : end + 1])
            index = end + 1
            continue
        if char.isalpha() or char == "_":
            end = index + 1
            while end < size and (text[end].isalnum() or text[end] == "_"):
                end += 1
            if end < size and text[end] == "(":
                close = text.find(")", end)
                if close > end:
                    end = close + 1
            tokens.append(text[index:end])
            index = end
            continue
        index += 1
    return tokens


def _require_disjoint(groups: Sequence[AtomGroup]) -> None:
    for group in groups:
        if not group.rows:
            raise EmptyGroup(group.name, group.selection)
    for left_at, left in enumerate(groups):
        left_rows = set(left.rows)
        for right in groups[left_at + 1 :]:
            shared = tuple(sorted(left_rows.intersection(right.rows)))
            if shared:
                raise OverlappingGroups(left.name, right.name, shared)

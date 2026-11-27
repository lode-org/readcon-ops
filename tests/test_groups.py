"""Named groups on a reference frame.

CON import stores atoms in first-encounter symbol order and keeps the
pre-group index on ``atom_id``. Residue atom lists in
``metadata["chemfiles_residues"]`` use that grouped row space.
"""

import warnings

import chemfiles
import pytest

from readcon_ops import as_frame, internal_motion


class Atom:
    def __init__(self, symbol, x, y, z, atom_id=None, mass=None):
        self.symbol = symbol
        self.x = x
        self.y = y
        self.z = z
        self.atom_id = atom_id
        self.mass = mass


class ConFrame:
    def __init__(self, cell, angles, atoms, metadata=None):
        self.cell = cell
        self.angles = angles
        self.atoms = atoms
        self.metadata = {} if metadata is None else metadata

    def __len__(self):
        return len(self.atoms)


_SPEC = (
    ("C", (0.0, 0.0, 0.0), "ALA"),
    ("H", (1.0, 0.0, 0.0), "ALA"),
    ("N", (2.0, 0.0, 0.0), "GLY"),
    ("H", (3.0, 0.0, 0.0), "ALA"),
    ("C", (4.0, 0.0, 0.0), "GLY"),
)


def _labelled_topology():
    frame = chemfiles.Frame()
    members = {}
    for index, (symbol, xyz, resname) in enumerate(_SPEC):
        frame.add_atom(chemfiles.Atom(symbol), xyz)
        members.setdefault(resname, []).append(index)
    for resname, resid in (("ALA", 1), ("GLY", 2)):
        residue = chemfiles.Residue(resname, resid)
        for index in members[resname]:
            residue.atoms.append(index)
        frame.add_residue(residue)
    return frame


def _type_grouped_import(topology):
    """Group a chemfiles frame the way CON import does.

    Symbol order is first encounter. Within a symbol, the earlier atom
    stays earlier. ``atom_id`` is the chemfiles index. Residue members
    are rewritten into the grouped rows.
    """
    count = len(topology.atoms)
    symbols = [topology.atoms[i].name for i in range(count)]
    positions = [tuple(float(value) for value in row) for row in topology.positions]
    order = []
    buckets = {}
    for index, symbol in enumerate(symbols):
        if symbol not in buckets:
            order.append(symbol)
            buckets[symbol] = []
        buckets[symbol].append(index)
    grouped = [index for symbol in order for index in buckets[symbol]]
    old_to_new = {old: new for new, old in enumerate(grouped)}
    atoms = []
    for old in grouped:
        x, y, z = positions[old]
        atoms.append(Atom(symbols[old], x, y, z, atom_id=old, mass=1.0))
    residues = []
    for residue in topology.topology.residues:
        residues.append(
            {
                "name": residue.name,
                "id": int(residue.id),
                "atoms": [old_to_new[int(index)] for index in residue.atoms],
            }
        )
    return ConFrame(
        (10.0, 10.0, 10.0),
        (90.0, 90.0, 90.0),
        atoms,
        metadata={"pbc": [False, False, False], "chemfiles_residues": residues},
    )


def _resolve():
    from readcon_ops.groups import resolve_named_groups

    return resolve_named_groups


def test_residue_groups_keep_source_ids_after_type_grouped_import():
    resolve_named_groups = _resolve()
    topology = _labelled_topology()
    labelled = resolve_named_groups(
        topology,
        {"ala": "resname ALA", "gly": "resname GLY"},
        disjoint=True,
    )
    assert labelled["ala"].rows == (0, 1, 3)
    assert labelled["gly"].rows == (2, 4)
    assert labelled["ala"].source_ids is None
    assert labelled.source_index is None

    imported_frame = _type_grouped_import(topology)
    assert [atom.symbol for atom in imported_frame.atoms] == ["C", "C", "H", "H", "N"]
    assert [atom.atom_id for atom in imported_frame.atoms] == [0, 4, 1, 3, 2]

    imported = resolve_named_groups(
        imported_frame,
        {"ala": "resname ALA", "gly": "resname GLY"},
        disjoint=True,
    )
    assert imported["ala"].rows == (0, 2, 3)
    assert imported["ala"].source_ids == (0, 1, 3)
    assert imported["gly"].rows == (1, 4)
    assert imported["gly"].source_ids == (4, 2)
    assert imported["ala"].rows != imported["ala"].source_ids
    assert imported["gly"].rows != imported["gly"].source_ids
    assert imported["ala"].source_ids == labelled["ala"].rows
    assert set(imported["gly"].source_ids) == set(labelled["gly"].rows)

    hydrogen = resolve_named_groups(imported_frame, {"h": "name H"})
    assert hydrogen["h"].rows == (2, 3)
    assert hydrogen["h"].source_ids == (1, 3)

    index = imported.source_index
    assert index.rows(imported["ala"].source_ids) == imported["ala"].rows
    assert index.sources(imported["ala"].rows) == imported["ala"].source_ids
    assert index.rows(imported["gly"].source_ids) == imported["gly"].rows
    assert index.sources(imported["gly"].rows) == imported["gly"].source_ids
    assert index.rows(index.source_of_row) == tuple(range(len(imported_frame)))
    assert index.sources(range(len(imported_frame))) == index.source_of_row


def test_disjoint_pair_rejects_empty_and_overlapping_groups():
    resolve_named_groups = _resolve()
    frame = _type_grouped_import(_labelled_topology())
    from readcon_ops.groups import EmptyGroup, OverlappingGroups

    try:
        resolve_named_groups(
            frame,
            {"ala": "resname ALA", "missing": "resname UNK"},
            disjoint=True,
        )
    except EmptyGroup as exc:
        assert exc.name == "missing"
        assert exc.selection == "resname UNK"
    else:
        raise AssertionError("empty group was accepted")

    try:
        resolve_named_groups(
            frame,
            {"ala": "resname ALA", "hydrogen": "name H"},
            disjoint=True,
        )
    except OverlappingGroups as exc:
        assert exc.left == "ala"
        assert exc.right == "hydrogen"
        assert exc.rows == (2, 3)
    else:
        raise AssertionError("overlapping group was accepted")

    permissive = resolve_named_groups(
        frame,
        {"ala": "resname ALA", "hydrogen": "name H", "missing": "resname UNK"},
    )
    assert permissive["missing"].rows == ()
    assert permissive["ala"].rows == (0, 2, 3)
    assert permissive["hydrogen"].rows == (2, 3)


def test_residue_selector_reports_missing_metadata():
    resolve_named_groups = _resolve()
    from readcon_ops.groups import MissingSelectionMetadata

    bare = chemfiles.Frame()
    bare.add_atom(chemfiles.Atom("C"), (0.0, 0.0, 0.0))
    try:
        resolve_named_groups(bare, {"ala": "resname ALA"})
    except MissingSelectionMetadata as exc:
        assert exc.selection == "resname ALA"
        assert exc.missing == "residues"
    else:
        raise AssertionError("missing residues were not reported")

    atoms = [Atom("C", 0.0, 0.0, 0.0, atom_id=0)]
    con = ConFrame((10.0, 10.0, 10.0), (90.0, 90.0, 90.0), atoms)
    try:
        resolve_named_groups(con, {"ala": "resname ALA"})
    except MissingSelectionMetadata as exc:
        assert exc.missing == "residues"
    else:
        raise AssertionError("CON frame without residues was accepted")


def test_source_ids_follow_the_frame_and_stay_optional():
    labelled = _type_grouped_import(_labelled_topology())
    adapted = as_frame(labelled)
    assert list(adapted.source_ids) == [0, 4, 1, 3, 2]

    shifted = ConFrame(
        labelled.cell,
        labelled.angles,
        [
            Atom(atom.symbol, atom.x + 0.2, atom.y, atom.z, atom_id=atom.atom_id)
            for atom in labelled.atoms
        ],
        metadata=labelled.metadata,
    )
    aligned = internal_motion(labelled, shifted)
    assert list(aligned.source_ids) == [0, 4, 1, 3, 2]

    plain = as_frame(
        ConFrame(
            (8.0, 8.0, 8.0),
            (90.0, 90.0, 90.0),
            [Atom("Cu", 0.0, 0.0, 0.0), Atom("H", 1.0, 0.0, 0.0)],
        )
    )
    assert plain.source_ids is None


def test_bad_selectors_and_broken_identities_are_rejected():
    resolve_named_groups = _resolve()
    from readcon_ops.groups import (
        DuplicateSourceId,
        IncompleteSourceIds,
        SelectionContextError,
        SelectionSyntaxError,
    )

    frame = _type_grouped_import(_labelled_topology())
    with pytest.raises(SelectionContextError):
        resolve_named_groups(frame, {"pairs": "bonds: all"})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid character")
        with pytest.raises(SelectionSyntaxError):
            resolve_named_groups(frame, {"bad": "!!!"})

    duplicated = ConFrame(
        (10.0, 10.0, 10.0),
        (90.0, 90.0, 90.0),
        [Atom("C", 0.0, 0.0, 0.0, atom_id=1), Atom("H", 1.0, 0.0, 0.0, atom_id=1)],
    )
    with pytest.raises(DuplicateSourceId):
        resolve_named_groups(duplicated, {"everything": "all"})

    mixed = ConFrame(
        (10.0, 10.0, 10.0),
        (90.0, 90.0, 90.0),
        [Atom("C", 0.0, 0.0, 0.0, atom_id=0), Atom("H", 1.0, 0.0, 0.0)],
    )
    with pytest.raises(IncompleteSourceIds):
        resolve_named_groups(mixed, {"everything": "all"})

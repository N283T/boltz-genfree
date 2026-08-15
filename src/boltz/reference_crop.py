"""Reference-guided receptor block selection for structure-free screening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import gemmi
import numpy as np

from boltz.data import const

if TYPE_CHECKING:
    from pathlib import Path

_NEIGHBORHOOD_SIZE = 10


@dataclass(frozen=True)
class ReferenceCrop:
    """Reference-derived receptor block priority and its provenance."""

    protein_chain: str
    ligand_chain: str
    ligand_residue: int
    ligand_ccd: str
    seed_order: tuple[int, ...]
    blocks: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ReferenceLigandSelector:
    """A deliberately strict single-residue reference ligand selector."""

    chain: str
    residue: int | None = None
    ccd: str | None = None


def protein_atom_counts(sequence: str) -> tuple[int, ...]:
    """Return Boltz heavy-atom counts for canonical protein residue tokens."""
    return tuple(
        len(const.ref_atoms[const.prot_letter_to_token.get(letter, "UNK")])
        for letter in sequence
    )


def _protein_residues(chain: gemmi.Chain) -> list[gemmi.Residue]:
    return [
        residue
        for residue in chain
        if gemmi.find_tabulated_residue(residue.name).is_amino_acid()
    ]


def _protein_sequence(residues: list[gemmi.Residue]) -> str:
    return "".join(
        gemmi.find_tabulated_residue(residue.name).one_letter_code or "X"
        for residue in residues
    )


def _heavy_atom_coords(residues: list[gemmi.Residue]) -> np.ndarray:
    coords = [
        (atom.pos.x, atom.pos.y, atom.pos.z)
        for residue in residues
        if not gemmi.find_tabulated_residue(residue.name).is_water()
        for atom in residue
        if atom.element.atomic_number != 1
    ]
    if not coords:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(coords, dtype=np.float64)


def _residue_center(residue: gemmi.Residue) -> np.ndarray | None:
    ca = residue.find_atom("CA", "*")
    if ca is not None:
        return np.asarray((ca.pos.x, ca.pos.y, ca.pos.z), dtype=np.float64)
    coords = _heavy_atom_coords([residue])
    return np.mean(coords, axis=0) if coords.size else None


def _choose_protein_chain(
    model: gemmi.Model,
    sequence: str,
    requested: str | None,
) -> tuple[gemmi.Chain, list[gemmi.Residue]]:
    candidates = [
        (chain, residues) for chain in model if (residues := _protein_residues(chain))
    ]
    if requested is not None:
        candidates = [item for item in candidates if item[0].name == requested]
        if not candidates:
            msg = f"Reference protein chain {requested!r} was not found."
            raise ValueError(msg)
    exact = [item for item in candidates if _protein_sequence(item[1]) == sequence]
    if len(exact) != 1:
        available = (
            ", ".join(
                f"{chain.name}({len(residues)} aa)" for chain, residues in candidates
            )
            or "none"
        )
        qualifier = (
            f"chain {requested!r}" if requested is not None else "exactly one chain"
        )
        msg = (
            f"Reference must contain {qualifier} whose observed protein sequence "
            f"exactly matches target.sequence; candidates: {available}."
        )
        raise ValueError(msg)
    return exact[0]


def _ligand_candidates(
    model: gemmi.Model,
) -> list[tuple[str, gemmi.Residue]]:
    candidates: list[tuple[str, gemmi.Residue]] = []
    for chain in model:
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if (
                not info.is_water()
                and not info.is_amino_acid()
                and not info.is_nucleic_acid()
                and _heavy_atom_coords([residue]).size
            ):
                candidates.append((chain.name, residue))
    return candidates


def _choose_ligand_residue(
    model: gemmi.Model,
    selector: ReferenceLigandSelector,
) -> gemmi.Residue:
    if (selector.residue is None) != (selector.ccd is None):
        msg = "Reference ligand residue and CCD must be specified together."
        raise ValueError(msg)
    chain_candidates = [
        residue
        for chain_name, residue in _ligand_candidates(model)
        if chain_name == selector.chain
    ]
    if selector.residue is None:
        candidates = chain_candidates
    else:
        candidates = [
            residue
            for residue in chain_candidates
            if residue.seqid.num == selector.residue
            and residue.name.upper() == selector.ccd
        ]
    if not candidates:
        if selector.residue is None:
            msg = (
                "Reference ligand chain matched no non-polymer residue: "
                f"chain={selector.chain!r}."
            )
        else:
            msg = (
                "Reference ligand selector matched no residue: "
                f"chain={selector.chain!r}, residue={selector.residue}, "
                f"ccd={selector.ccd!r}."
            )
        raise ValueError(msg)
    if len(candidates) != 1:
        if selector.residue is None:
            msg = (
                "Reference ligand chain matched multiple residues; specify "
                "reference_ligand with chain, residue, and CCD."
            )
        else:
            msg = (
                "Reference ligand selector matched multiple residues; preprocess the "
                "reference so chain, residue, and CCD identify exactly one residue."
            )
        raise ValueError(msg)
    selected = candidates[0]
    if selected.seqid.icode.strip():
        msg = (
            "Reference ligand uses an insertion code, which is unsupported. "
            "Preprocess the reference to a unique residue without an insertion code."
        )
        raise ValueError(msg)
    if selected.entity_type == gemmi.EntityType.Branched:
        msg = (
            "Multi-residue or branched reference ligands are unsupported; provide "
            "a preprocessed single-residue ligand."
        )
        raise ValueError(msg)
    if any(atom.has_altloc() for atom in selected):
        msg = (
            "Reference ligand contains alternate locations, which are unsupported. "
            "Preprocess the reference to one ligand conformer."
        )
        raise ValueError(msg)
    return selected


def _block_around(seed: int, sequence_length: int) -> tuple[int, ...]:
    if sequence_length <= _NEIGHBORHOOD_SIZE:
        return tuple(range(sequence_length))
    start = seed - (_NEIGHBORHOOD_SIZE // 2)
    start = min(max(0, start), sequence_length - _NEIGHBORHOOD_SIZE)
    return tuple(range(start, start + _NEIGHBORHOOD_SIZE))


def load_reference_crop(
    path: Path,
    sequence: str,
    *,
    protein_chain: str | None = None,
    ligand: ReferenceLigandSelector,
) -> ReferenceCrop:
    """Rank receptor seeds by distance to a ligand and form sequence blocks."""
    try:
        structure = gemmi.read_structure(str(path))
    except (RuntimeError, ValueError) as error:
        msg = f"Could not parse crop reference {path}: {error}"
        raise ValueError(msg) from error
    if len(structure) == 0:
        msg = f"Crop reference contains no coordinate model: {path}"
        raise ValueError(msg)
    model = structure[0]
    selected_protein, residues = _choose_protein_chain(
        model,
        sequence,
        protein_chain,
    )
    selected_ligand = _choose_ligand_residue(
        model,
        ligand,
    )
    ligand_coords = _heavy_atom_coords([selected_ligand])
    centers = [_residue_center(residue) for residue in residues]
    resolved = [index for index, center in enumerate(centers) if center is not None]
    if not resolved:
        msg = "Reference protein contains no resolved heavy atoms."
        raise ValueError(msg)
    distances = [
        float(np.linalg.norm(ligand_coords - centers[index], axis=1).min())
        for index in resolved
    ]
    seed_order = tuple(
        index
        for _, index in sorted(
            zip(distances, resolved, strict=True),
            key=lambda item: (item[0], item[1]),
        )
    )
    blocks = tuple(_block_around(seed, len(sequence)) for seed in seed_order)
    return ReferenceCrop(
        protein_chain=selected_protein.name,
        ligand_chain=ligand.chain,
        ligand_residue=selected_ligand.seqid.num,
        ligand_ccd=selected_ligand.name.upper(),
        seed_order=seed_order,
        blocks=blocks,
    )


def select_receptor_tokens(
    crop: ReferenceCrop,
    sequence: str,
    *,
    ligand_tokens: int,
    max_tokens: int = 256,
    max_atoms: int = 2048,
    max_protein_tokens: int = 200,
) -> tuple[int, ...]:
    """Apply the affinity crop budget without ever truncating the ligand."""
    atom_counts = protein_atom_counts(sequence)
    selected: set[int] = set()
    selected_atoms = 0
    for block in crop.blocks:
        new_indices = set(block) - selected
        new_atoms = sum(atom_counts[index] for index in new_indices)
        if (
            len(selected) + len(new_indices) + ligand_tokens > max_tokens
            or selected_atoms + new_atoms + ligand_tokens > max_atoms
            or len(selected) + len(new_indices) > max_protein_tokens
        ):
            break
        selected.update(new_indices)
        selected_atoms += new_atoms
    if not selected:
        msg = (
            "No reference-derived receptor block fits after reserving the complete "
            f"{ligand_tokens}-token ligand."
        )
        raise ValueError(msg)
    return tuple(sorted(selected))

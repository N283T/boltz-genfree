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
    seed_order: tuple[int, ...]
    blocks: tuple[tuple[int, ...], ...]


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


def _choose_ligand_residues(
    model: gemmi.Model,
    requested: str | None,
) -> tuple[str, list[gemmi.Residue]]:
    candidates: list[tuple[str, list[gemmi.Residue]]] = []
    for chain in model:
        residues = [
            residue
            for residue in chain
            if not gemmi.find_tabulated_residue(residue.name).is_water()
            and not gemmi.find_tabulated_residue(residue.name).is_amino_acid()
            and not gemmi.find_tabulated_residue(residue.name).is_nucleic_acid()
        ]
        if _heavy_atom_coords(residues).size:
            candidates.append((chain.name, residues))
    if requested is not None:
        candidates = [item for item in candidates if item[0] == requested]
        if not candidates:
            msg = f"Reference ligand chain {requested!r} was not found."
            raise ValueError(msg)
    if len(candidates) != 1:
        names = ", ".join(name for name, _ in candidates) or "none"
        msg = (
            "Reference must contain exactly one non-water ligand chain after "
            f"selection; candidates: {names}. Set target.crop.reference_ligand."
        )
        raise ValueError(msg)
    return candidates[0]


def _block_around(seed: int, sequence_length: int) -> tuple[int, ...]:
    if sequence_length <= _NEIGHBORHOOD_SIZE:
        return tuple(range(sequence_length))
    lower_bound = max(0, seed - _NEIGHBORHOOD_SIZE)
    upper_bound = min(sequence_length - 1, seed + _NEIGHBORHOOD_SIZE)
    radius = 0
    block = (seed,)
    while len(block) < _NEIGHBORHOOD_SIZE:
        radius += 1
        lower = max(lower_bound, seed - radius)
        upper = min(upper_bound, seed + radius)
        block = tuple(range(lower, upper + 1))
    return block


def load_reference_crop(
    path: Path,
    sequence: str,
    *,
    protein_chain: str | None = None,
    ligand_chain: str | None = None,
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
    selected_ligand, ligand_residues = _choose_ligand_residues(
        model,
        ligand_chain,
    )
    ligand_coords = _heavy_atom_coords(ligand_residues)
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
        ligand_chain=selected_ligand,
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

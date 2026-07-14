from typing import Literal

import torch
from torch import Tensor

COORDINATE_DIMENSIONS = 3

AffinityDistogramMode = Literal[
    "control",
    "collapsed_coordinates",
    "constant_middle",
    "constant_far",
    "zero_conditioning",
]

AFFINITY_DISTOGRAM_MODES: tuple[AffinityDistogramMode, ...] = (
    "control",
    "collapsed_coordinates",
    "constant_middle",
    "constant_far",
    "zero_conditioning",
)


def validate_affinity_distogram_mode(mode: str) -> AffinityDistogramMode:
    """Validate and narrow an affinity distogram intervention name."""
    if mode not in AFFINITY_DISTOGRAM_MODES:
        choices = ", ".join(AFFINITY_DISTOGRAM_MODES)
        msg = f"Unknown affinity distogram mode {mode!r}; expected one of: {choices}"
        raise ValueError(msg)
    return mode  # type: ignore[return-value]


def protein_ligand_pair_mask(
    receptor_mask: Tensor,
    ligand_mask: Tensor,
) -> Tensor:
    """Return a symmetric mask containing only protein-ligand token pairs."""
    receptor_mask = receptor_mask.to(torch.bool)
    ligand_mask = ligand_mask.to(torch.bool)
    return (ligand_mask[:, :, None] & receptor_mask[:, None, :]) | (
        receptor_mask[:, :, None] & ligand_mask[:, None, :]
    )


def build_affinity_distogram_bins(
    representative_coordinates: Tensor,
    boundaries: Tensor,
    receptor_mask: Tensor,
    ligand_mask: Tensor,
    mode: str = "control",
) -> tuple[Tensor, Tensor]:
    """Build effective affinity distogram bins and the intervened pair mask.

    Constant interventions replace only protein-ligand pairs. Protein-protein
    and ligand-ligand geometry remains coordinate-derived. Collapsed coordinates
    affect all pairs containing a ligand token. Zero conditioning retains the
    real bins for auditing; the caller is responsible for skipping the
    conditioner term for every valid affinity token pair.
    """
    mode = validate_affinity_distogram_mode(mode)
    if representative_coordinates.ndim != COORDINATE_DIMENSIONS:
        msg = "Representative coordinates must have shape (batch, tokens, 3)"
        raise ValueError(msg)
    expected_mask_shape = representative_coordinates.shape[:2]
    if receptor_mask.shape != expected_mask_shape:
        msg = f"Receptor mask shape {receptor_mask.shape} != {expected_mask_shape}"
        raise ValueError(msg)
    if ligand_mask.shape != expected_mask_shape:
        msg = f"Ligand mask shape {ligand_mask.shape} != {expected_mask_shape}"
        raise ValueError(msg)

    coordinates = representative_coordinates
    if mode == "collapsed_coordinates":
        coordinates = torch.where(
            ligand_mask.to(torch.bool).unsqueeze(-1),
            torch.zeros_like(coordinates),
            coordinates,
        )

    distances = torch.cdist(coordinates, coordinates)
    bins = (distances.unsqueeze(-1) > boundaries).sum(dim=-1).long()
    protein_ligand_mask = protein_ligand_pair_mask(receptor_mask, ligand_mask)
    valid_mask = receptor_mask.to(torch.bool) | ligand_mask.to(torch.bool)
    valid_pair_mask = valid_mask[:, :, None] & valid_mask[:, None, :]
    ligand_pair_mask = (
        ligand_mask.to(torch.bool)[:, :, None] | ligand_mask.to(torch.bool)[:, None, :]
    ) & valid_pair_mask

    if mode == "control":
        intervened_pair_mask = torch.zeros_like(valid_pair_mask)
    elif mode == "collapsed_coordinates":
        intervened_pair_mask = ligand_pair_mask
    elif mode == "zero_conditioning":
        intervened_pair_mask = valid_pair_mask
    else:
        intervened_pair_mask = protein_ligand_mask

    if mode == "constant_middle":
        num_bins = boundaries.numel() + 1
        bins = torch.where(protein_ligand_mask, num_bins // 2, bins)
    elif mode == "constant_far":
        far_bin = boundaries.numel()
        bins = torch.where(protein_ligand_mask, far_bin, bins)

    return bins, intervened_pair_mask


def uses_affinity_conditioning(mode: str) -> bool:
    """Return whether the PairwiseConditioning contribution should be added."""
    return validate_affinity_distogram_mode(mode) != "zero_conditioning"

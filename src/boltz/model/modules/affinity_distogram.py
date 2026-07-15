from typing import Literal

import torch
from torch import Tensor

COORDINATE_DIMENSIONS = 3

AffinityDistogramMode = Literal[
    "control",
    "collapsed_coordinates",
    "constant_middle",
    "constant_far",
    "uniform_random",
    "normal_random",
    "zero_distogram_features",
    "uniform_mean_embedding",
    "zero_conditioner_pair_features",
    "distogram_only_affinity",
    "zero_conditioning",
]
RandomPairScope = Literal["all", "protein_ligand"]

AFFINITY_DISTOGRAM_MODES: tuple[AffinityDistogramMode, ...] = (
    "control",
    "collapsed_coordinates",
    "constant_middle",
    "constant_far",
    "uniform_random",
    "normal_random",
    "zero_distogram_features",
    "uniform_mean_embedding",
    "zero_conditioner_pair_features",
    "distogram_only_affinity",
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
    random_seed: int = 0,
    normal_mean: float = 12.0,
    normal_std: float = 4.0,
    random_pair_scope: RandomPairScope = "all",
) -> tuple[Tensor, Tensor]:
    """Build effective affinity distogram bins and the intervened pair mask.

    Constant interventions replace only protein-ligand pairs. Protein-protein
    and ligand-ligand geometry remains coordinate-derived. Collapsed coordinates
    affect all pairs containing a ligand token. Random interventions replace all
    valid pairs with a symmetric, structure-independent pseudo-distogram.
    Feature-level null interventions and zero conditioning retain the real bins
    for auditing. The caller replaces the embedded features for feature-level
    interventions or skips the conditioner term for zero conditioning.
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

    if random_pair_scope not in {"all", "protein_ligand"}:
        msg = f"Unknown random pair scope: {random_pair_scope!r}"
        raise ValueError(msg)

    valid_mask = receptor_mask.to(torch.bool) | ligand_mask.to(torch.bool)
    valid_pair_mask = valid_mask[:, :, None] & valid_mask[:, None, :]
    random_bins = None
    if mode in {"uniform_random", "normal_random"}:
        if normal_std <= 0:
            raise ValueError("normal_std must be positive.")
        generator = torch.Generator(device=representative_coordinates.device)
        generator.manual_seed(random_seed)
        shape = valid_pair_mask.shape
        num_bins = boundaries.numel() + 1
        if mode == "uniform_random":
            sampled = torch.randint(
                0,
                num_bins,
                shape,
                generator=generator,
                device=representative_coordinates.device,
            )
        else:
            sampled_distances = torch.randn(
                shape,
                generator=generator,
                device=representative_coordinates.device,
                dtype=boundaries.dtype,
            )
            sampled_distances = (
                sampled_distances * normal_std + normal_mean
            ).clamp_min(0)
            sampled = (sampled_distances.unsqueeze(-1) > boundaries).sum(
                dim=-1
            ).long()
        upper = torch.triu(sampled, diagonal=1)
        random_bins = upper + upper.transpose(1, 2)
        random_bins = torch.where(
            valid_pair_mask,
            random_bins,
            torch.zeros_like(random_bins),
        )
        if random_pair_scope == "all":
            return random_bins, valid_pair_mask

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
    ligand_pair_mask = (
        ligand_mask.to(torch.bool)[:, :, None] | ligand_mask.to(torch.bool)[:, None, :]
    ) & valid_pair_mask

    if random_bins is not None:
        bins = torch.where(protein_ligand_mask, random_bins, bins)
        intervened_pair_mask = protein_ligand_mask
    elif mode == "control":
        intervened_pair_mask = torch.zeros_like(valid_pair_mask)
    elif mode == "collapsed_coordinates":
        intervened_pair_mask = ligand_pair_mask
    elif mode in {
        "zero_distogram_features",
        "uniform_mean_embedding",
        "zero_conditioner_pair_features",
        "distogram_only_affinity",
        "zero_conditioning",
    }:
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


def intervene_affinity_distogram_features(
    distogram: Tensor,
    embedding_weight: Tensor,
    mode: str,
) -> Tensor:
    """Apply a structure-independent intervention after bin embedding."""
    mode = validate_affinity_distogram_mode(mode)
    if mode == "zero_distogram_features":
        return torch.zeros_like(distogram)
    if mode == "uniform_mean_embedding":
        mean_embedding = embedding_weight.mean(dim=0)
        return mean_embedding.view(*([1] * (distogram.ndim - 1)), -1).expand_as(
            distogram
        )
    return distogram


def intervene_affinity_pair_features(z: Tensor, mode: str) -> tuple[Tensor, Tensor]:
    """Return residual and conditioner pair features for a mechanism ablation."""
    mode = validate_affinity_distogram_mode(mode)
    if mode == "distogram_only_affinity":
        zeros = torch.zeros_like(z)
        return zeros, zeros
    if mode == "zero_conditioner_pair_features":
        return z, torch.zeros_like(z)
    return z, z


def uses_affinity_conditioning(mode: str) -> bool:
    """Return whether the PairwiseConditioning contribution should be added."""
    return validate_affinity_distogram_mode(mode) != "zero_conditioning"

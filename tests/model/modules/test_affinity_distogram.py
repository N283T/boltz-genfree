# ruff: noqa: ANN001, ANN201, D103

import pytest
import torch

from boltz.model.modules.affinity_distogram import (
    AFFINITY_DISTOGRAM_MODES,
    build_affinity_distogram_bins,
    protein_ligand_pair_mask,
    uses_affinity_conditioning,
)


@pytest.fixture
def geometry():
    coordinates = torch.tensor(
        [
            [
                [3.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ]
    )
    receptor_mask = torch.tensor([[True, True, False, False]])
    ligand_mask = torch.tensor([[False, False, True, True]])
    boundaries = torch.linspace(2, 22, 63)
    return coordinates, boundaries, receptor_mask, ligand_mask


def test_control_matches_legacy_distogram_exactly(geometry):
    coordinates, boundaries, receptor_mask, ligand_mask = geometry
    legacy_distances = torch.cdist(coordinates, coordinates)
    legacy_bins = (legacy_distances.unsqueeze(-1) > boundaries).sum(dim=-1).long()

    bins, _ = build_affinity_distogram_bins(
        coordinates,
        boundaries,
        receptor_mask,
        ligand_mask,
        mode="control",
    )

    assert torch.equal(bins, legacy_bins)


def test_collapsed_coordinates_matches_explicit_origin_replacement(geometry):
    coordinates, boundaries, receptor_mask, ligand_mask = geometry
    collapsed = coordinates.clone()
    collapsed[ligand_mask] = 0
    expected = (
        (torch.cdist(collapsed, collapsed).unsqueeze(-1) > boundaries)
        .sum(dim=-1)
        .long()
    )

    bins, pair_mask = build_affinity_distogram_bins(
        coordinates,
        boundaries,
        receptor_mask,
        ligand_mask,
        mode="collapsed_coordinates",
    )

    assert torch.equal(bins, expected)
    assert bins[0, 2, 3].item() == 0
    expected_pair_mask = ligand_mask[:, :, None] | ligand_mask[:, None, :]
    assert torch.equal(pair_mask, expected_pair_mask)


@pytest.mark.parametrize(
    ("mode", "expected_bin"),
    [("constant_middle", 32), ("constant_far", 63)],
)
def test_constant_modes_replace_only_protein_ligand_pairs(
    geometry,
    mode,
    expected_bin,
):
    coordinates, boundaries, receptor_mask, ligand_mask = geometry
    control, control_pair_mask = build_affinity_distogram_bins(
        coordinates,
        boundaries,
        receptor_mask,
        ligand_mask,
        mode="control",
    )
    bins, returned_pair_mask = build_affinity_distogram_bins(
        coordinates,
        boundaries,
        receptor_mask,
        ligand_mask,
        mode=mode,
    )

    pair_mask = protein_ligand_pair_mask(receptor_mask, ligand_mask)
    assert not control_pair_mask.any()
    assert torch.equal(returned_pair_mask, pair_mask)
    assert torch.all(bins[pair_mask] == expected_bin)
    assert torch.equal(bins[~pair_mask], control[~pair_mask])
    assert torch.equal(bins, bins.transpose(1, 2))


def test_protein_ligand_pair_mask_excludes_ligand_ligand_pairs(geometry):
    _, _, receptor_mask, ligand_mask = geometry
    mask = protein_ligand_pair_mask(receptor_mask, ligand_mask)

    assert mask[0, 0, 2]
    assert mask[0, 2, 0]
    assert not mask[0, 2, 3]
    assert not mask[0, 0, 1]


def test_zero_conditioning_retains_real_bins_for_audit(geometry):
    coordinates, boundaries, receptor_mask, ligand_mask = geometry
    control, _ = build_affinity_distogram_bins(
        coordinates,
        boundaries,
        receptor_mask,
        ligand_mask,
        mode="control",
    )
    zero_conditioning, pair_mask = build_affinity_distogram_bins(
        coordinates,
        boundaries,
        receptor_mask,
        ligand_mask,
        mode="zero_conditioning",
    )

    assert torch.equal(zero_conditioning, control)
    assert pair_mask.all()
    assert not uses_affinity_conditioning("zero_conditioning")


def test_unknown_mode_is_rejected(geometry):
    coordinates, boundaries, receptor_mask, ligand_mask = geometry
    with pytest.raises(ValueError, match="Unknown affinity distogram mode"):
        build_affinity_distogram_bins(
            coordinates,
            boundaries,
            receptor_mask,
            ligand_mask,
            mode="mystery",
        )


def test_all_declared_modes_are_accepted(geometry):
    coordinates, boundaries, receptor_mask, ligand_mask = geometry
    for mode in AFFINITY_DISTOGRAM_MODES:
        build_affinity_distogram_bins(
            coordinates,
            boundaries,
            receptor_mask,
            ligand_mask,
            mode=mode,
        )

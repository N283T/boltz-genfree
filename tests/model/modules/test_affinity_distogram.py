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


@pytest.mark.parametrize("mode", ["uniform_random", "normal_random"])
def test_random_modes_are_symmetric_reproducible_and_structure_independent(
    geometry,
    mode,
):
    coordinates, boundaries, receptor_mask, ligand_mask = geometry
    first, pair_mask = build_affinity_distogram_bins(
        coordinates,
        boundaries,
        receptor_mask,
        ligand_mask,
        mode=mode,
        random_seed=17,
    )
    repeated, _ = build_affinity_distogram_bins(
        coordinates * 1000,
        boundaries,
        receptor_mask,
        ligand_mask,
        mode=mode,
        random_seed=17,
    )
    changed_seed, _ = build_affinity_distogram_bins(
        coordinates,
        boundaries,
        receptor_mask,
        ligand_mask,
        mode=mode,
        random_seed=18,
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed_seed)
    assert torch.equal(first, first.transpose(1, 2))
    assert torch.all(first.diagonal(dim1=1, dim2=2) == 0)
    assert pair_mask.all()


def test_normal_random_rejects_non_positive_standard_deviation(geometry):
    coordinates, boundaries, receptor_mask, ligand_mask = geometry
    with pytest.raises(ValueError, match="normal_std must be positive"):
        build_affinity_distogram_bins(
            coordinates,
            boundaries,
            receptor_mask,
            ligand_mask,
            mode="normal_random",
            normal_std=0,
        )


@pytest.mark.parametrize("mode", ["uniform_random", "normal_random"])
def test_random_modes_do_not_compute_coordinate_distances(geometry, monkeypatch, mode):
    coordinates, boundaries, receptor_mask, ligand_mask = geometry

    def fail_cdist(*args, **kwargs):
        raise AssertionError("random modes must not inspect coordinates")

    monkeypatch.setattr(torch, "cdist", fail_cdist)
    build_affinity_distogram_bins(
        coordinates,
        boundaries,
        receptor_mask,
        ligand_mask,
        mode=mode,
    )


@pytest.mark.parametrize("mode", ["uniform_random", "normal_random"])
def test_protein_ligand_random_scope_preserves_other_pairs(geometry, mode):
    coordinates, boundaries, receptor_mask, ligand_mask = geometry
    control, _ = build_affinity_distogram_bins(
        coordinates,
        boundaries,
        receptor_mask,
        ligand_mask,
        mode="control",
    )
    bins, intervened = build_affinity_distogram_bins(
        coordinates,
        boundaries,
        receptor_mask,
        ligand_mask,
        mode=mode,
        random_seed=17,
        random_pair_scope="protein_ligand",
    )
    protein_ligand = protein_ligand_pair_mask(receptor_mask, ligand_mask)

    assert torch.equal(intervened, protein_ligand)
    assert torch.equal(bins[~protein_ligand], control[~protein_ligand])
    assert torch.equal(bins, bins.transpose(1, 2))


def test_unknown_random_pair_scope_is_rejected(geometry):
    coordinates, boundaries, receptor_mask, ligand_mask = geometry
    with pytest.raises(ValueError, match="Unknown random pair scope"):
        build_affinity_distogram_bins(
            coordinates,
            boundaries,
            receptor_mask,
            ligand_mask,
            mode="uniform_random",
            random_pair_scope="mystery",
        )


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

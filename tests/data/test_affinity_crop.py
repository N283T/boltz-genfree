import numpy as np
import pytest

from boltz.data.crop.affinity import AffinityCropper
from boltz.data.types import TokenBondV2, Tokenized, TokenV2


def _tokenized() -> Tokenized:
    tokens = np.zeros(4, dtype=TokenV2)
    tokens["token_idx"] = np.arange(4)
    tokens["affinity_mask"][3] = True
    bonds = np.array([(0, 1, 1), (1, 3, 1), (2, 3, 1)], dtype=TokenBondV2)
    return Tokenized(tokens=tokens, bonds=bonds, structure=None, msa={})  # type: ignore[arg-type]


def _sequence_tokenized() -> Tokenized:
    tokens = np.zeros(7, dtype=TokenV2)
    tokens["token_idx"] = np.arange(7)
    tokens["atom_num"] = [5, 5, 5, 5, 1, 1, 1]
    tokens["mol_type"][:4] = 0
    tokens["mol_type"][4:] = 3
    tokens["affinity_mask"][4:] = True
    # Coordinates and resolution must not affect the sequence crop.
    tokens["center_coords"] = np.arange(21).reshape(7, 3)
    tokens["resolved_mask"] = [False, True, False, True, False, True, False]
    bonds = np.zeros(0, dtype=TokenBondV2)
    return Tokenized(tokens=tokens, bonds=bonds, structure=None, msa={})  # type: ignore[arg-type]


def test_explicit_affinity_crop_preserves_token_order_and_internal_bonds():
    cropped = AffinityCropper.crop_by_token_indices(
        _tokenized(),
        np.array([3, 1]),
    )

    assert cropped.tokens["token_idx"].tolist() == [1, 3]
    assert cropped.bonds[["token_1", "token_2"]].tolist() == [(1, 3)]


def test_explicit_affinity_crop_requires_binder():
    with pytest.raises(ValueError, match="does not contain the binder"):
        AffinityCropper.crop_by_token_indices(_tokenized(), np.array([0, 1]))


def test_explicit_affinity_crop_rejects_unknown_token_id():
    with pytest.raises(ValueError, match="missing token IDs"):
        AffinityCropper.crop_by_token_indices(_tokenized(), np.array([3, 9]))


def test_receptor_crop_unions_complete_binder_and_checks_budget():
    cropped = AffinityCropper(max_tokens_protein=2).crop_by_receptor_token_indices(
        _sequence_tokenized(),
        np.array([1, 2]),
        max_tokens=5,
        max_atoms=20,
    )

    assert cropped.tokens["token_idx"].tolist() == [1, 2, 4, 5, 6]

    with pytest.raises(ValueError, match="exceeds 4 total tokens"):
        AffinityCropper(max_tokens_protein=2).crop_by_receptor_token_indices(
            _sequence_tokenized(),
            np.array([1, 2]),
            max_tokens=4,
        )


def test_sequence_prefix_crop_keeps_all_binder_and_prefix_receptor_tokens():
    cropper = AffinityCropper(max_tokens_protein=2)

    cropped = cropper.crop_sequence_prefix(
        _sequence_tokenized(),
        max_tokens=5,
        max_atoms=20,
    )

    assert cropped.tokens["token_idx"].tolist() == [0, 1, 4, 5, 6]


def test_sequence_prefix_crop_does_not_depend_on_coordinates_or_resolution():
    cropper = AffinityCropper(max_tokens_protein=2)
    first = _sequence_tokenized()
    second = _sequence_tokenized()
    second.tokens["center_coords"] *= -100
    second.tokens["resolved_mask"] = ~second.tokens["resolved_mask"]

    first_crop = cropper.crop_sequence_prefix(first, max_tokens=5, max_atoms=20)
    second_crop = cropper.crop_sequence_prefix(second, max_tokens=5, max_atoms=20)

    assert (
        first_crop.tokens["token_idx"].tolist()
        == second_crop.tokens["token_idx"].tolist()
    )


def test_full_screen_crop_keeps_every_token_when_input_fits():
    cropped = AffinityCropper(max_tokens_protein=4).crop_full_screen(
        _sequence_tokenized(),
        max_tokens=7,
        max_atoms=23,
    )

    assert cropped.tokens["token_idx"].tolist() == list(range(7))


def test_full_screen_crop_fails_instead_of_truncating_receptor():
    with pytest.raises(ValueError, match="kept 5/7 tokens"):
        AffinityCropper(max_tokens_protein=2).crop_full_screen(
            _sequence_tokenized(),
            max_tokens=5,
            max_atoms=20,
        )


def test_sequence_window_crop_uses_zero_based_protein_offset():
    cropper = AffinityCropper(max_tokens_protein=2)

    cropped = cropper.crop_sequence_window(
        _sequence_tokenized(),
        protein_start=1,
        max_tokens=5,
        max_atoms=20,
    )

    assert cropped.tokens["token_idx"].tolist() == [1, 2, 4, 5, 6]


def test_sequence_window_crop_rejects_start_outside_receptor():
    with pytest.raises(ValueError, match="outside the 4-token receptor"):
        AffinityCropper().crop_sequence_window(
            _sequence_tokenized(),
            protein_start=4,
            max_tokens=5,
        )

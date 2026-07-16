import numpy as np
import pytest

from boltz.data.crop.affinity import AffinityCropper
from boltz.data.types import TokenBondV2, TokenV2, Tokenized


def _tokenized() -> Tokenized:
    tokens = np.zeros(4, dtype=TokenV2)
    tokens["token_idx"] = np.arange(4)
    tokens["affinity_mask"][3] = True
    bonds = np.array([(0, 1, 1), (1, 3, 1), (2, 3, 1)], dtype=TokenBondV2)
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

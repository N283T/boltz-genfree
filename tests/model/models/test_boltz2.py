from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from boltz.model.models import boltz2


class _DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def _minimal_boltz2(
    validate_structure,
    affinity_distogram_modes=None,
    skip_run_structure=False,
):
    return boltz2.Boltz2(
        atom_s=2,
        atom_z=2,
        token_s=2,
        token_z=2,
        num_bins=4,
        training_args={},
        validation_args={},
        embedder_args={},
        msa_args={},
        pairformer_args={},
        score_model_args={
            "atom_encoder_depth": 1,
            "atom_encoder_heads": 1,
            "token_transformer_depth": 1,
            "token_transformer_heads": 1,
            "atom_decoder_depth": 1,
            "atom_decoder_heads": 1,
            "conditioning_transition_layers": 1,
        },
        diffusion_process_args={},
        diffusion_loss_args={},
        confidence_prediction=False,
        validate_structure=validate_structure,
        validators=[],
        affinity_distogram_modes=affinity_distogram_modes,
        skip_run_structure=skip_run_structure,
    )


def test_boltz2_stores_validate_structure(monkeypatch):
    for module_name in [
        "InputEmbedder",
        "RelativePositionEncoder",
        "ContactConditioning",
        "MSAModule",
        "PairformerModule",
        "DiffusionConditioning",
        "AtomDiffusion",
        "DistogramModule",
    ]:
        monkeypatch.setattr(boltz2, module_name, _DummyModule)

    for validate_structure in [False, True]:
        model = _minimal_boltz2(validate_structure)

        assert model.validate_structure is validate_structure

    model = _minimal_boltz2(False, ["constant_far"])
    assert model.affinity_distogram_modes == ("constant_far",)


def test_select_affinity_coordinates_can_reuse_prepared_geometry(monkeypatch):
    for module_name in [
        "InputEmbedder",
        "RelativePositionEncoder",
        "ContactConditioning",
        "MSAModule",
        "PairformerModule",
        "DiffusionConditioning",
        "AtomDiffusion",
        "DistogramModule",
    ]:
        monkeypatch.setattr(boltz2, module_name, _DummyModule)

    model = _minimal_boltz2(False, skip_run_structure=True)
    prepared = torch.randn(2, 1, 7, 3)

    selected = model._select_affinity_coordinates({}, {"coords": prepared})

    assert selected is prepared


def test_select_affinity_coordinates_uses_highest_iptm_pose(monkeypatch):
    for module_name in [
        "InputEmbedder",
        "RelativePositionEncoder",
        "ContactConditioning",
        "MSAModule",
        "PairformerModule",
        "DiffusionConditioning",
        "AtomDiffusion",
        "DistogramModule",
    ]:
        monkeypatch.setattr(boltz2, module_name, _DummyModule)

    model = _minimal_boltz2(False)
    poses = torch.randn(3, 7, 3)
    selected = model._select_affinity_coordinates(
        {"iptm": torch.tensor([0.2, 0.9, 0.4]), "sample_atom_coords": poses},
        {},
    )

    torch.testing.assert_close(selected, poses[1][None, None])


def test_load_resume_embeddings_supports_nested_layout_and_crop(tmp_path):
    record_dir = tmp_path / "target"
    record_dir.mkdir()
    s = np.arange(12, dtype=np.float32).reshape(4, 3)
    z = np.arange(32, dtype=np.float32).reshape(4, 4, 2)
    np.savez_compressed(record_dir / "embeddings_target.npz", s=s[None], z=z[None])

    loaded = boltz2.Boltz2._load_resume_embeddings(
        [SimpleNamespace(id="target")],
        tmp_path,
        token_indices=torch.tensor([[3, 1]]),
        token_masks=torch.tensor([[True, True]]),
    )

    loaded_s, loaded_z = loaded[0]
    torch.testing.assert_close(loaded_s, torch.from_numpy(s[[3, 1]]))
    torch.testing.assert_close(
        loaded_z,
        torch.from_numpy(z[[3, 1]][:, [3, 1]]),
    )


def test_load_resume_embeddings_rejects_missing_arrays(tmp_path):
    np.savez_compressed(tmp_path / "embeddings_target.npz", s=np.zeros((2, 3)))

    with pytest.raises(ValueError, match="must contain s and z"):
        boltz2.Boltz2._load_resume_embeddings(
            [SimpleNamespace(id="target")],
            tmp_path,
        )


def test_load_resume_embeddings_aligns_cropped_cache_by_token_id(tmp_path):
    record_dir = tmp_path / "target"
    record_dir.mkdir()
    token_idx = np.array([2, 7, 11])
    s = np.arange(9, dtype=np.float32).reshape(3, 3)
    z = np.arange(18, dtype=np.float32).reshape(3, 3, 2)
    np.savez_compressed(
        record_dir / "affinity_trunk_embeddings_target.npz",
        s=s,
        z=z,
        token_idx=token_idx,
    )

    loaded = boltz2.Boltz2._load_resume_embeddings(
        [SimpleNamespace(id="target")],
        tmp_path,
        token_indices=torch.tensor([[7, 11]]),
        token_masks=torch.tensor([[True, True]]),
    )

    loaded_s, loaded_z = loaded[0]
    torch.testing.assert_close(loaded_s, torch.from_numpy(s[[1, 2]]))
    torch.testing.assert_close(
        loaded_z,
        torch.from_numpy(z[[1, 2]][:, [1, 2]]),
    )


def test_load_resume_embeddings_rejects_missing_saved_token_id(tmp_path):
    np.savez_compressed(
        tmp_path / "affinity_trunk_embeddings_target.npz",
        s=np.zeros((2, 3), dtype=np.float32),
        z=np.zeros((2, 2, 4), dtype=np.float32),
        token_idx=np.array([3, 5]),
    )

    with pytest.raises(ValueError, match="do not contain requested token IDs"):
        boltz2.Boltz2._load_resume_embeddings(
            [SimpleNamespace(id="target")],
            tmp_path,
            token_indices=torch.tensor([[3, 8]]),
            token_masks=torch.tensor([[True, True]]),
        )


def test_load_affinity_context_supports_nested_layout(tmp_path):
    record_dir = tmp_path / "target"
    record_dir.mkdir()
    arrays = {
        "s_inputs": np.zeros((1, 2, 4), dtype=np.float32),
        "z": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "coords": np.zeros((1, 1, 5, 3), dtype=np.float32),
        "token_pad_mask": np.ones((1, 2), dtype=np.float32),
        "mol_type": np.zeros((1, 2), dtype=np.float32),
        "affinity_token_mask": np.array([[0, 1]], dtype=np.float32),
        "token_to_rep_atom": np.zeros((1, 2, 5), dtype=np.float32),
        "affinity_mw": np.array([123.4], dtype=np.float32),
    }
    np.savez_compressed(
        record_dir / "affinity_head_context_target.npz",
        **arrays,
    )

    loaded = boltz2.Boltz2._load_affinity_context(
        [SimpleNamespace(id="target")],
        tmp_path,
    )

    assert set(loaded) == set(arrays)
    for key, value in arrays.items():
        torch.testing.assert_close(loaded[key], torch.from_numpy(value))

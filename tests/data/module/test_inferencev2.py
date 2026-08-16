from pathlib import Path

import pytest

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest


def test_predict_dataloader_uses_configured_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("boltz.data.module.inferencev2.load_canonicals", lambda _: {})

    data_module = Boltz2InferenceDataModule(
        manifest=Manifest(records=[]),
        target_dir=tmp_path / "structures",
        msa_dir=tmp_path / "msa",
        mol_dir=tmp_path / "mols",
        num_workers=0,
        batch_size=3,
    )

    assert data_module.predict_dataloader().batch_size == 3


def test_predict_dataloader_defaults_to_batch_size_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("boltz.data.module.inferencev2.load_canonicals", lambda _: {})

    data_module = Boltz2InferenceDataModule(
        manifest=Manifest(records=[]),
        target_dir=tmp_path / "structures",
        msa_dir=tmp_path / "msa",
        mol_dir=tmp_path / "mols",
        num_workers=0,
    )

    assert data_module.predict_dataloader().batch_size == 1


def test_prepared_geometry_affinity_rejects_batch_size_above_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("boltz.data.module.inferencev2.load_canonicals", lambda _: {})

    with pytest.raises(
        ValueError,
        match="Batched affinity inference requires affinity_structure_free",
    ):
        Boltz2InferenceDataModule(
            manifest=Manifest(records=[]),
            target_dir=tmp_path / "structures",
            msa_dir=tmp_path / "msa",
            mol_dir=tmp_path / "mols",
            num_workers=0,
            batch_size=2,
            affinity=True,
        )


def test_structure_free_affinity_accepts_batch_size_above_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("boltz.data.module.inferencev2.load_canonicals", lambda _: {})

    data_module = Boltz2InferenceDataModule(
        manifest=Manifest(records=[]),
        target_dir=tmp_path / "structures",
        msa_dir=tmp_path / "msa",
        mol_dir=tmp_path / "mols",
        num_workers=0,
        batch_size=2,
        affinity=True,
        affinity_structure_free=True,
    )

    assert data_module.predict_dataloader().batch_size == 2


def test_structure_free_single_sequence_is_forwarded_to_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("boltz.data.module.inferencev2.load_canonicals", lambda _: {})

    data_module = Boltz2InferenceDataModule(
        manifest=Manifest(records=[]),
        target_dir=tmp_path / "structures",
        msa_dir=tmp_path / "msa",
        mol_dir=tmp_path / "mols",
        num_workers=0,
        affinity=True,
        affinity_structure_free=True,
        affinity_structure_free_single_sequence=True,
    )

    dataset = data_module.predict_dataloader().dataset
    assert dataset.affinity_structure_free_single_sequence is True


def test_structure_free_random_seed_is_forwarded_to_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("boltz.data.module.inferencev2.load_canonicals", lambda _: {})

    data_module = Boltz2InferenceDataModule(
        manifest=Manifest(records=[]),
        target_dir=tmp_path / "structures",
        msa_dir=tmp_path / "msa",
        mol_dir=tmp_path / "mols",
        num_workers=0,
        affinity=True,
        affinity_structure_free=True,
        affinity_random_seed=17,
    )

    dataset = data_module.predict_dataloader().dataset
    assert dataset.affinity_random_seed == 17

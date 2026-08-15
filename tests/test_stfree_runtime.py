"""Tests for the single-route structure-free runtime contract."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from boltz.data import const
from boltz.data.types import (
    AffinityInfo,
    ChainInfo,
    Manifest,
    Record,
    StructureInfo,
    get_affinity_output_id,
)
from boltz.main import download_stfree
from boltz.stfree_runtime import (
    StFreeRuntimeOptions,
    _run_provenance,
    _validate_one_to_one_manifest,
    _validate_resume_state,
    stfree_model_kwargs,
)


def _record(*, extra_ligand: bool = False) -> Record:
    chains = [
        ChainInfo(
            chain_id=0,
            chain_name="A",
            mol_type=const.chain_type_ids["PROTEIN"],
            cluster_id="protein",
            msa_id=-1,
            num_residues=40,
        ),
        ChainInfo(
            chain_id=1,
            chain_name="B",
            mol_type=const.chain_type_ids["NONPOLYMER"],
            cluster_id="ligand",
            msa_id=-1,
            num_residues=1,
        ),
    ]
    if extra_ligand:
        chains.append(
            ChainInfo(
                chain_id=2,
                chain_name="C",
                mol_type=const.chain_type_ids["NONPOLYMER"],
                cluster_id="other-ligand",
                msa_id=-1,
                num_residues=1,
            )
        )
    return Record(
        id="record",
        structure=StructureInfo(num_chains=len(chains)),
        chains=chains,
        interfaces=[],
        affinity=AffinityInfo(chain_id=1, chain_name="B", mw=46.0),
    )


def test_runtime_accepts_one_protein_one_ligand() -> None:
    _validate_one_to_one_manifest(Manifest([_record()]))


def test_runtime_rejects_additional_complex_chains() -> None:
    with pytest.raises(click.ClickException, match="one unconstrained protein"):
        _validate_one_to_one_manifest(Manifest([_record(extra_ligand=True)]))


def test_model_route_never_runs_structure_sampling(tmp_path: Path) -> None:
    kwargs = stfree_model_kwargs(StFreeRuntimeOptions(cache=tmp_path))

    assert kwargs["affinity_distogram_modes"] == ["zero_distogram_features"]
    assert kwargs["affinity_coordinate_free"] is True
    assert kwargs["skip_run_structure"] is True
    assert kwargs["skip_confidence_forward"] is True
    assert kwargs["skip_pdistogram_forward"] is True


def test_download_stfree_omits_structure_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "mols").mkdir()
    downloads: list[Path] = []

    def fake_download(url: str, destination: str) -> None:  # noqa: ARG001
        path = Path(destination)
        downloads.append(path)
        path.write_bytes(b"checkpoint")

    monkeypatch.setattr("boltz.main._download", fake_download)

    download_stfree(tmp_path)

    assert [path.name for path in downloads] == ["boltz2_aff.ckpt"]
    assert not (tmp_path / "boltz2_conf.ckpt").exists()


def test_download_stfree_skips_default_checkpoint_for_custom_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "mols").mkdir()
    downloads: list[str] = []
    monkeypatch.setattr(
        "boltz.main._download",
        lambda url, destination: downloads.append(destination),
    )

    download_stfree(tmp_path, include_default_affinity=False)

    assert downloads == []


def _write_existing_score(
    run_dir: Path,
    manifest: Manifest,
    provenance: dict[str, object],
) -> Path:
    record = manifest.records[0]
    output_id = get_affinity_output_id(record)
    prediction_dir = run_dir / "predictions"
    score_dir = prediction_dir / record.id
    score_dir.mkdir(parents=True)
    (prediction_dir / "affinity_run_metadata.json").write_text(
        json.dumps({"experiment": {"provenance": provenance}}),
        encoding="utf-8",
    )
    score_path = (
        score_dir / f"affinity_{output_id}__distogram_zero_distogram_features.json"
    )
    score_path.write_text(
        json.dumps(
            {
                "affinity_pred_value": 1.0,
                "affinity_probability_binary": 0.5,
                "distogram_intervention": "zero_distogram_features",
                "experiment": {"run_metadata_file": "../affinity_run_metadata.json"},
            }
        ),
        encoding="utf-8",
    )
    return score_path


def test_resume_rejects_changed_provenance(tmp_path: Path) -> None:
    manifest = Manifest([_record()])
    _write_existing_score(tmp_path, manifest, {"checkpoint": "old"})

    with pytest.raises(click.ClickException, match="different checkpoint"):
        _validate_resume_state(
            manifest=manifest,
            run_dir=tmp_path,
            provenance={"checkpoint": "new"},
            override=False,
        )


def test_resume_rejects_malformed_score_and_override_removes_old_output(
    tmp_path: Path,
) -> None:
    manifest = Manifest([_record()])
    provenance = {"checkpoint": "same"}
    score_path = _write_existing_score(tmp_path, manifest, provenance)
    score_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(click.ClickException, match="score is invalid"):
        _validate_resume_state(
            manifest=manifest,
            run_dir=tmp_path,
            provenance=provenance,
            override=False,
        )
    _validate_resume_state(
        manifest=manifest,
        run_dir=tmp_path,
        provenance=provenance,
        override=True,
    )
    assert not score_path.exists()


def test_run_provenance_captures_checkpoint_and_method(tmp_path: Path) -> None:
    provenance = _run_provenance(
        StFreeRuntimeOptions(cache=tmp_path),
        runtime_version="test-version",
        checkpoint_sha256="abc123",
    )

    assert provenance["affinity_checkpoint_sha256"] == "abc123"
    assert provenance["method"]["structure_generation_active"] is False
    assert provenance["method"]["ligand_reference_conformer"] == {
        "mode": "skip",
        "reference_positions": "zero",
        "etkdg_seed": None,
    }
    assert len(provenance["runtime_code_sha256"]) == 64


def test_run_provenance_records_seeded_etkdg(tmp_path: Path) -> None:
    provenance = _run_provenance(
        StFreeRuntimeOptions(
            cache=tmp_path,
            ligand_reference_conformer="etkdg",
            etkdg_seed=17,
        ),
        runtime_version="test-version",
        checkpoint_sha256="abc123",
    )

    assert provenance["method"]["ligand_reference_conformer"] == {
        "mode": "etkdg",
        "reference_positions": "etkdg_v3_uff",
        "etkdg_seed": 17,
    }

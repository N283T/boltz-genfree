"""Tests for the single-route structure-free runtime contract."""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from boltz.data import const
from boltz.data.types import AffinityInfo, ChainInfo, Manifest, Record, StructureInfo
from boltz.main import download_stfree
from boltz.stfree_runtime import (
    StFreeRuntimeOptions,
    _validate_one_to_one_manifest,
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

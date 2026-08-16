"""Tests for the screen-first configuration and library workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from collections.abc import Callable
from pathlib import Path

import click
import gemmi
import pytest
import yaml
from click.testing import CliRunner
from rdkit import Chem

import boltz.stfree_runtime as stfree_runtime
from boltz import reference_crop as reference_crop_module
from boltz.main import cli
from boltz.screen import (
    ScreenProgress,
    build_screen_options,
    collect_screen_scores,
    ensure_auto_msa,
    invalidate_screen_aggregates,
    prepare_screen,
)
from boltz.stfree_runtime import stfree_model_kwargs

SEQUENCE = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEI"


def _write_config(
    tmp_path: Path,
    ligands: dict[str, object],
    *,
    options: dict[str, object] | None = None,
    chunk_size: int = 1000,
    sequence: str = SEQUENCE,
) -> Path:
    config = {
        "version": 1,
        "target": {"sequence": sequence, "msa": "empty"},
        "ligands": ligands,
        "output_dir": "./output",
        "chunk_size": chunk_size,
        "options": options or {},
    }
    path = tmp_path / "screen.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _write_ccd_cache(tmp_path: Path, molecules: dict[str, str]) -> Path:
    cache = tmp_path / "cache"
    mol_dir = cache / "mols"
    mol_dir.mkdir(parents=True)
    for code, smiles in molecules.items():
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None
        with (mol_dir / f"{code}.pkl").open("wb") as handle:
            pickle.dump(molecule, handle)
    return cache


def _write_reference(
    tmp_path: Path,
    length: int,
    ligand_position: int = 5,
    ligand_chain: str = "L",
    ligand_residue: int = 1,
    ligand_ccd: str = "LIG",
    ligand_insertion_code: str = "",
    ligand_altloc: str = "",
) -> Path:
    lines = []
    serial = 1
    for residue_index in range(1, length + 1):
        x = float(residue_index)
        lines.append(
            f"ATOM  {serial:5d}  CA  ALA A{residue_index:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
        )
        serial += 1
    lines.append("TER")
    lines.append(
        f"HETATM{serial:5d} {'C1':^4}{ligand_altloc:1}{ligand_ccd:>3} "
        f"{ligand_chain:1}{ligand_residue:4d}{ligand_insertion_code:1}   "
        f"{float(ligand_position):8.3f}{1.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
    )
    lines.append("END")
    path = tmp_path / "reference.pdb"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_mmcif_reference_with_missing_residues(
    tmp_path: Path,
    *,
    length: int,
    observed_positions: tuple[int, ...],
    ligand_position: int,
) -> Path:
    structure = gemmi.Structure()
    structure.name = "missing-residue-reference"
    protein_entity = gemmi.Entity("1")
    protein_entity.entity_type = gemmi.EntityType.Polymer
    protein_entity.polymer_type = gemmi.PolymerType.PeptideL
    protein_entity.full_sequence = ["ALA"] * length
    structure.entities.append(protein_entity)
    ligand_entity = gemmi.Entity("2")
    ligand_entity.entity_type = gemmi.EntityType.NonPolymer
    structure.entities.append(ligand_entity)

    model = gemmi.Model("1")
    protein_chain = gemmi.Chain("A")
    for position in observed_positions:
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId(position, " ")
        residue.label_seq = position
        residue.entity_id = "1"
        residue.entity_type = gemmi.EntityType.Polymer
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(float(position), 0.0, 0.0)
        residue.add_atom(atom)
        protein_chain.add_residue(residue)
    model.add_chain(protein_chain)

    ligand_chain = gemmi.Chain("L")
    ligand = gemmi.Residue()
    ligand.name = "LIG"
    ligand.seqid = gemmi.SeqId(1, " ")
    ligand.entity_id = "2"
    ligand.entity_type = gemmi.EntityType.NonPolymer
    atom = gemmi.Atom()
    atom.name = "C1"
    atom.element = gemmi.Element("C")
    atom.pos = gemmi.Position(float(ligand_position), 1.0, 0.0)
    ligand.add_atom(atom)
    ligand_chain.add_residue(ligand)
    model.add_chain(ligand_chain)
    structure.add_model(model)

    path = tmp_path / "missing-residue-reference.cif"
    structure.make_mmcif_document().write_file(str(path))
    return path


def test_prepare_inline_smiles_chunks_and_is_idempotent(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "smiles": [
                {"id": "compound one", "smiles": "CCO"},
                {"id": "compound/two", "smiles": "CCN"},
            ]
        },
        chunk_size=1,
    )

    prepared = prepare_screen(config_path)
    first_plan = prepared.plan_path.read_text(encoding="utf-8")
    repeated = prepare_screen(config_path)

    assert prepared == repeated
    assert repeated.plan_path.read_text(encoding="utf-8") == first_plan
    assert [path.name for path in prepared.chunk_dirs] == [
        "chunk_00000",
        "chunk_00001",
    ]
    generated = sorted(prepared.output_dir.glob(".boltz-stfree/inputs/chunk_*/*.yaml"))
    assert [path.stem for path in generated] == [
        "000001_compound_one",
        "000002_compound_two",
    ]
    first = yaml.safe_load(generated[0].read_text(encoding="utf-8"))
    assert first["sequences"][0]["protein"]["msa"] == "empty"
    assert first["sequences"][1]["ligand"] == {"id": "B", "smiles": "CCO"}
    assert first["properties"] == [{"affinity": {"binder": "B"}}]
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert [record["ligand_tokens"] for record in manifest["records"]] == [3, 3]


def test_prepare_csv_library_resolves_config_relative_path(tmp_path: Path) -> None:
    library_path = tmp_path / "library.csv"
    library_path.write_text("compound,canonical\na,CCO\nb,CCN\n", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        {
            "csv": "./library.csv",
            "id_column": "compound",
            "smiles_column": "canonical",
        },
    )

    prepared = prepare_screen(config_path)
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))

    assert [record["ligand_id"] for record in manifest["records"]] == ["a", "b"]
    assert manifest["ligand_source"]["path"] == str(library_path.resolve())
    assert len(manifest["ligand_source"]["sha256"]) == 64


def test_prepare_sdf_library_uses_molecule_names(tmp_path: Path) -> None:
    sdf_path = tmp_path / "library.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    for name, smiles in (("ethanol", "CCO"), ("ethylamine", "CCN")):
        molecule = Chem.MolFromSmiles(smiles)
        molecule.SetProp("_Name", name)
        writer.write(molecule)
    writer.close()
    config_path = _write_config(tmp_path, {"sdf": "./library.sdf"})

    prepared = prepare_screen(config_path)
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))

    assert [record["ligand_id"] for record in manifest["records"]] == [
        "ethanol",
        "ethylamine",
    ]
    assert all(record["ligand_kind"] == "smiles" for record in manifest["records"])


def test_prepare_ccd_library(tmp_path: Path) -> None:
    _write_ccd_cache(tmp_path, {"ATP": "CCO", "ADP": "CCN"})
    config_path = _write_config(
        tmp_path,
        {"ccd": ["ATP", {"id": "adp-reference", "ccd": "ADP"}]},
        options={"cache": "./cache"},
    )

    prepared = prepare_screen(config_path)
    generated = sorted(prepared.output_dir.glob(".boltz-stfree/inputs/chunk_*/*.yaml"))
    payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in generated]

    assert payloads[0]["sequences"][1]["ligand"]["ccd"] == "ATP"
    assert payloads[1]["sequences"][1]["ligand"]["ccd"] == "ADP"


def test_prepare_ccd_library_can_populate_missing_cache(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"ccd": ["ATP"]},
        options={"cache": "./cache"},
    )
    calls: list[Path] = []

    def ensure_cache(cache: Path) -> None:
        calls.append(cache)
        _write_ccd_cache(tmp_path, {"ATP": "CCO"})

    prepared = prepare_screen(config_path, ensure_ccd_cache=ensure_cache)

    assert calls == [tmp_path / "cache"]
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["records"][0]["ligand_tokens"] == 3


def test_auto_msa_enables_server_by_default(tmp_path: Path) -> None:
    config = {
        "version": 1,
        "target": {"sequence": SEQUENCE},
        "ligands": {"smiles": ["CCO"]},
        "output_dir": "./output",
    }
    path = tmp_path / "auto.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    prepared = prepare_screen(path)

    assert "use_msa_server" not in prepared.options
    assert prepared.auto_msa_path == (tmp_path / "output/.boltz-stfree/msa/target.csv")
    generated = next(prepared.output_dir.glob(".boltz-stfree/inputs/chunk_*/*.yaml"))
    payload = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert payload["sequences"][0]["protein"]["msa"] == str(prepared.auto_msa_path)


def test_auto_msa_is_generated_once_and_verified(tmp_path: Path) -> None:
    config = {
        "version": 1,
        "target": {"sequence": SEQUENCE},
        "ligands": {"smiles": ["CCO"]},
        "output_dir": "./output",
    }
    path = tmp_path / "auto.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    prepared = prepare_screen(path)
    calls: list[dict[str, object]] = []

    def fake_compute_msa(**kwargs: object) -> None:
        calls.append(kwargs)
        msa_dir = Path(str(kwargs["msa_dir"]))
        msa_dir.mkdir(parents=True, exist_ok=True)
        (msa_dir / "target.csv").write_text(
            "key,sequence\n-1," + SEQUENCE + "\n",
            encoding="utf-8",
        )

    options = {
        "msa_server_url": "https://msa.example.test",
        "msa_pairing_strategy": "greedy",
    }
    screen_options = build_screen_options(options, config_dir=tmp_path)
    ensure_auto_msa(
        prepared,
        screen_options=screen_options,
        compute_msa_fn=fake_compute_msa,
    )
    ensure_auto_msa(
        prepared,
        screen_options=screen_options,
        compute_msa_fn=fake_compute_msa,
    )

    assert len(calls) == 1
    assert calls[0]["data"] == {"target": SEQUENCE}
    assert prepared.auto_msa_path is not None
    metadata_path = prepared.auto_msa_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["server_url"] == "https://msa.example.test"
    changed_options = build_screen_options(
        {**options, "msa_pairing_strategy": "complete"},
        config_dir=tmp_path,
    )
    with pytest.raises(click.ClickException, match="source settings"):
        ensure_auto_msa(
            prepared,
            screen_options=changed_options,
            compute_msa_fn=fake_compute_msa,
        )


def test_custom_msa_is_hashed_in_manifest(tmp_path: Path) -> None:
    msa_path = tmp_path / "target.a3m"
    msa_path.write_text(">query\nAAAA\n", encoding="utf-8")
    config = {
        "version": 1,
        "target": {"sequence": SEQUENCE, "msa": "./target.a3m"},
        "ligands": {"smiles": ["CCO"]},
        "output_dir": "./output",
    }
    path = tmp_path / "custom-msa.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    prepared = prepare_screen(path)
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))

    assert manifest["target"]["msa"]["mode"] == "file"
    assert manifest["target"]["msa"]["path"] == str(msa_path.resolve())
    assert len(manifest["target"]["msa"]["sha256"]) == 64


def test_full_screen_rejects_target_above_200_tokens(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"smiles": ["CCO"]},
        sequence="A" * 201,
    )

    with pytest.raises(click.UsageError, match="target_tokens=201 exceeds"):
        prepare_screen(config_path)


def test_reference_crop_accepts_long_target_and_records_provenance(
    tmp_path: Path,
) -> None:
    reference = _write_reference(tmp_path, 230)
    config_path = _write_config(
        tmp_path,
        {"smiles": ["CCO", "C" * 60]},
        sequence="A" * 230,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["target"]["crop"] = {
        "reference": reference.name,
        "reference_protein": "A",
        "reference_ligand": {"chain": "L", "residue": 1, "ccd": "LIG"},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    prepared = prepare_screen(config_path)
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    crops = [record["crop"] for record in manifest["records"]]

    assert prepared.receptor_crop_indices_dir is not None
    assert manifest["target"]["crop"]["budget"] == "per_ligand"
    assert manifest["target"]["crop"]["reference_sha256"]
    assert crops[0]["receptor_tokens"] > crops[1]["receptor_tokens"]
    crop_document = json.loads(Path(crops[0]["path"]).read_text(encoding="utf-8"))
    assert crop_document["selection"] == "reference_contiguous_blocks"
    assert crop_document["ligand_tokens_reserved"] == 3
    assert (
        crop_document["reference_sha256"]
        == manifest["target"]["crop"]["reference_sha256"]
    )
    for record in manifest["records"]:
        prediction_dir = (
            prepared.runs_dir
            / f"boltz_results_{record['chunk']}"
            / "predictions"
            / record["record_id"]
        )
        prediction_dir.mkdir(parents=True, exist_ok=True)
        score_path = (
            prediction_dir / "affinity_B__distogram_zero_distogram_features.json"
        )
        score_path.write_text(
            json.dumps(
                {
                    "affinity_pred_value": 1.0,
                    "affinity_probability_binary": 0.5,
                }
            ),
            encoding="utf-8",
        )
    with collect_screen_scores(prepared).open(encoding="utf-8", newline="") as handle:
        score_rows = list(csv.DictReader(handle))
    assert {row["crop_budget"] for row in score_rows} == {"per_ligand"}
    assert [int(row["receptor_tokens"]) for row in score_rows] == [
        crop["receptor_tokens"] for crop in crops
    ]


def test_reference_crop_records_adjacent_bootstrap_provenance(tmp_path: Path) -> None:
    reference = _write_reference(tmp_path, 230)
    reference_sha256 = hashlib.sha256(reference.read_bytes()).hexdigest()
    result_path = tmp_path / "bootstrap_result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "complete",
                "reference": str(reference.resolve()),
                "reference_sha256": reference_sha256,
                "checkpoint": "/models/boltz2_conf.ckpt",
                "checkpoint_sha256": "a" * 64,
                "selected_confidence": 0.8,
                "target_chain": "A",
                "anchor_chain": "L",
            }
        ),
        encoding="utf-8",
    )
    config_path = _write_config(
        tmp_path,
        {"smiles": ["CCO"]},
        sequence="A" * 230,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["target"]["crop"] = {
        "reference": reference.name,
        "reference_protein": "A",
        "reference_ligand": "L",
        "budget": "library_fixed",
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    prepared = prepare_screen(config_path)
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))

    source = manifest["target"]["crop"]["reference_source"]
    assert source["type"] == "predicted_holo"
    assert source["result"] == str(result_path)
    assert source["checkpoint_sha256"] == "a" * 64
    assert source["selected_confidence"] == 0.8


@pytest.mark.parametrize("seed", [0, 1, 5, 50, 94, 98, 99])
def test_reference_crop_blocks_are_exactly_ten_contiguous_residues(seed: int) -> None:
    block = reference_crop_module._block_around(seed, 100)

    assert len(block) == 10
    assert seed in block
    assert block == tuple(range(block[0], block[0] + 10))


def test_reference_sequence_normalizes_modified_residue_codes() -> None:
    residue = gemmi.Residue()
    residue.name = "MSE"

    assert reference_crop_module._protein_sequence([residue]) == "M"


def test_reference_crop_maps_observed_residues_to_declared_sequence(
    tmp_path: Path,
) -> None:
    reference = _write_mmcif_reference_with_missing_residues(
        tmp_path,
        length=30,
        observed_positions=tuple(range(11, 31)),
        ligand_position=21,
    )

    crop = reference_crop_module.load_reference_crop(
        reference,
        "A" * 30,
        protein_chain="A",
        ligand=reference_crop_module.ReferenceLigandSelector(
            chain="L",
            residue=1,
            ccd="LIG",
        ),
    )

    assert crop.seed_order[0] == 20
    assert crop.blocks[0] == tuple(range(15, 25))


@pytest.mark.parametrize(
    ("length", "observed_positions", "ligand_position", "expected_seed"),
    [
        pytest.param(
            33,
            (*range(1, 11), *range(14, 34)),
            14,
            10,
            id="internal-unobserved-insertion",
        ),
        pytest.param(
            37,
            tuple(range(8, 38)),
            8,
            0,
            id="terminal-extension",
        ),
    ],
)
def test_reference_crop_remaps_exact_observed_sequence_to_target_indices(
    tmp_path: Path,
    length: int,
    observed_positions: tuple[int, ...],
    ligand_position: int,
    expected_seed: int,
) -> None:
    reference = _write_mmcif_reference_with_missing_residues(
        tmp_path,
        length=length,
        observed_positions=observed_positions,
        ligand_position=ligand_position,
    )
    target_sequence = "A" * len(observed_positions)

    crop = reference_crop_module.load_reference_crop(
        reference,
        target_sequence,
        protein_chain="A",
        ligand=reference_crop_module.ReferenceLigandSelector(
            chain="L",
            residue=1,
            ccd="LIG",
        ),
    )

    assert crop.seed_order[0] == expected_seed
    assert all(0 <= index < len(target_sequence) for index in crop.seed_order)
    assert all(
        0 <= index < len(target_sequence)
        for block in crop.blocks
        for index in block
    )


def test_reference_crop_rejects_partial_observed_sequence_match(
    tmp_path: Path,
) -> None:
    reference = _write_mmcif_reference_with_missing_residues(
        tmp_path,
        length=33,
        observed_positions=(*range(1, 11), *range(14, 34)),
        ligand_position=14,
    )

    with pytest.raises(ValueError, match="exactly matches target.sequence"):
        reference_crop_module.load_reference_crop(
            reference,
            "A" * 29,
            protein_chain="A",
            ligand=reference_crop_module.ReferenceLigandSelector(
                chain="L",
                residue=1,
                ccd="LIG",
            ),
        )


def test_sequence_pocket_crop_uses_one_based_residue_identity_guards() -> None:
    crop = reference_crop_module.load_sequence_pocket_crop(
        "ACDEFGHIKLMNPQRSTVWY",
        (
            reference_crop_module.SequencePocketResidue(residue=6, name="g"),
            reference_crop_module.SequencePocketResidue(residue=15, name="R"),
        ),
    )

    assert crop.seed_order == (5, 14)
    assert crop.blocks == (tuple(range(10)), tuple(range(9, 19)))
    assert [item.name for item in crop.pocket] == ["G", "R"]


def test_sequence_pocket_crop_rejects_target_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="target.sequence contains G"):
        reference_crop_module.load_sequence_pocket_crop(
            "AAAAAGAAAA",
            (reference_crop_module.SequencePocketResidue(residue=6, name="A"),),
        )


def test_sequence_pocket_crop_applies_budget_per_ligand(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"smiles": ["CCO", "C" * 60]},
        sequence="A" * 230,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["target"]["crop"] = {
        "pocket": [
            {"residue": residue, "name": "A"}
            for residue in range(6, 227, 11)
        ],
        "budget": "per_ligand",
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    prepared = prepare_screen(config_path)
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    crop_documents = [
        json.loads(Path(record["crop"]["path"]).read_text(encoding="utf-8"))
        for record in manifest["records"]
    ]

    assert manifest["target"]["crop"]["selection"] == "sequence_pocket_blocks"
    assert crop_documents[0]["selection"] == "sequence_pocket_blocks"
    assert "reference" not in crop_documents[0]
    assert (
        crop_documents[0]["ligand_tokens_reserved"]
        < crop_documents[1]["ligand_tokens_reserved"]
    )
    assert (
        len(crop_documents[0]["receptor_token_idx"])
        > len(crop_documents[1]["receptor_token_idx"])
    )
    assert all(
        0 <= index < 230
        for document in crop_documents
        for index in document["receptor_token_idx"]
    )


def test_reference_crop_library_fixed_uses_one_receptor_selection(
    tmp_path: Path,
) -> None:
    reference = _write_reference(tmp_path, 230)
    config_path = _write_config(
        tmp_path,
        {"smiles": ["CCO", "C" * 60]},
        sequence="A" * 230,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["target"]["crop"] = {
        "reference": reference.name,
        "reference_protein": "A",
        "reference_ligand": {"chain": "L", "residue": 1, "ccd": "LIG"},
        "budget": "library_fixed",
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    prepared = prepare_screen(config_path)
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    crop_documents = [
        json.loads(Path(record["crop"]["path"]).read_text(encoding="utf-8"))
        for record in manifest["records"]
    ]

    assert (
        crop_documents[0]["receptor_token_idx"]
        == crop_documents[1]["receptor_token_idx"]
    )
    assert {item["ligand_tokens_reserved"] for item in crop_documents} == {60}


def test_reference_crop_reuses_selection_for_equal_ligand_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _write_reference(tmp_path, 230)
    config_path = _write_config(
        tmp_path,
        {"smiles": ["CCO", "CCN", "CCCC"]},
        sequence="A" * 230,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["target"]["crop"] = {
        "reference": reference.name,
        "reference_protein": "A",
        "reference_ligand": "L",
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    calls: list[int] = []
    original = reference_crop_module.select_receptor_tokens

    def tracked_selection(*args, ligand_tokens: int, **kwargs):
        calls.append(ligand_tokens)
        return original(*args, ligand_tokens=ligand_tokens, **kwargs)

    monkeypatch.setattr("boltz.screen.select_receptor_tokens", tracked_selection)

    prepare_screen(config_path)

    assert calls == [3, 4]


def test_reference_ligand_may_share_protein_author_chain(tmp_path: Path) -> None:
    reference = _write_reference(tmp_path, 30, ligand_chain="A")
    config_path = _write_config(
        tmp_path,
        {"smiles": ["CCO"]},
        sequence="A" * 30,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["target"]["crop"] = {
        "reference": reference.name,
        "reference_protein": "A",
        "reference_ligand": "A",
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    prepared = prepare_screen(config_path)
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))

    assert manifest["target"]["crop"]["protein_chain"] == "A"
    assert manifest["target"]["crop"]["ligand"] == {
        "chain": "A",
        "residue": 1,
        "ccd": "LIG",
    }


def test_reference_ligand_selector_requires_chain_residue_and_ccd(
    tmp_path: Path,
) -> None:
    reference = _write_reference(tmp_path, 30)
    config_path = _write_config(tmp_path, {"smiles": ["CCO"]}, sequence="A" * 30)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["target"]["crop"] = {
        "reference": reference.name,
        "reference_ligand": {"chain": "L", "residue": 1},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(click.UsageError, match="missing: ccd"):
        prepare_screen(config_path)


def test_reference_ligand_selector_rejects_zero_and_multiple_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _write_reference(tmp_path, 30)
    config_path = _write_config(tmp_path, {"smiles": ["CCO"]}, sequence="A" * 30)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["target"]["crop"] = {
        "reference": reference.name,
        "reference_ligand": "L",
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    original_candidates = reference_crop_module._ligand_candidates

    def duplicate_candidates(model: object) -> list[tuple[str, object]]:
        candidates = original_candidates(model)  # type: ignore[arg-type]
        return [*candidates, *candidates]  # type: ignore[return-value]

    monkeypatch.setattr(
        reference_crop_module,
        "_ligand_candidates",
        duplicate_candidates,
    )

    with pytest.raises(click.UsageError, match="chain matched multiple residues"):
        prepare_screen(config_path)

    monkeypatch.setattr(
        reference_crop_module,
        "_ligand_candidates",
        original_candidates,
    )
    config["target"]["crop"]["reference_ligand"] = "Z"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(click.UsageError, match="matched no non-polymer residue"):
        prepare_screen(config_path)


@pytest.mark.parametrize(
    ("reference_kwargs", "message"),
    [
        ({"ligand_insertion_code": "A"}, "insertion code"),
        ({"ligand_altloc": "A"}, "alternate locations"),
    ],
)
def test_reference_ligand_selector_rejects_unprocessed_identifiers(
    tmp_path: Path,
    reference_kwargs: dict[str, object],
    message: str,
) -> None:
    reference = _write_reference(tmp_path, 30, **reference_kwargs)
    config_path = _write_config(tmp_path, {"smiles": ["CCO"]}, sequence="A" * 30)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["target"]["crop"] = {
        "reference": reference.name,
        "reference_ligand": {"chain": "L", "residue": 1, "ccd": "LIG"},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(click.UsageError, match=message):
        prepare_screen(config_path)


def test_full_screen_rejects_ligand_above_128_tokens(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"smiles": [{"id": "too-large", "smiles": "C" * 129}]},
    )

    with pytest.raises(click.UsageError, match="too-large: ligand_tokens=129"):
        prepare_screen(config_path)


def test_full_screen_rejects_target_ligand_total_above_256(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"smiles": [{"id": "does-not-fit", "smiles": "C" * 57}]},
        sequence="A" * 200,
    )

    with pytest.raises(click.UsageError, match=r"200 \+ ligand_tokens=57 = 257"):
        prepare_screen(config_path)


def test_full_screen_warns_above_training_ligand_limit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(
        tmp_path,
        {"smiles": [{"id": "large-but-valid", "smiles": "C" * 57}]},
    )

    prepare_screen(config_path)

    assert "large-but-valid=57" in capsys.readouterr().err


def test_screen_rejects_explicit_crop_options(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"smiles": ["CCO"]},
        options={
            "affinity_crop_indices_dir": "./crops",
            "affinity_structure_free_crop_label": "known-pocket-v1",
        },
    )

    with pytest.raises(click.UsageError, match="full-screen-only"):
        prepare_screen(config_path)


def test_screen_options_are_yaml_driven_and_limited(tmp_path: Path) -> None:
    options = build_screen_options(
        {
            "devices": 1,
            "num_workers": 4,
            "cache": "./cache",
            "affinity_recycling_steps": 3,
        },
        config_dir=tmp_path,
    )

    assert options.runtime.devices == 1
    assert options.runtime.num_workers == 4
    assert options.runtime.cache == (tmp_path / "cache").resolve()
    assert options.runtime.affinity_recycling_steps == 3
    assert options.runtime.ligand_reference_conformer == "skip"
    assert options.runtime.etkdg_seed == 42
    model_options = stfree_model_kwargs(options.runtime)
    assert model_options["affinity_coordinate_free"] is True
    assert model_options["affinity_distogram_modes"] == ["zero_distogram_features"]
    assert model_options["skip_run_structure"] is True
    assert model_options["skip_pdistogram_forward"] is True


def test_screen_options_allow_seeded_etkdg(tmp_path: Path) -> None:
    options = build_screen_options(
        {
            "ligand_reference_conformer": "etkdg",
            "etkdg_seed": 17,
        },
        config_dir=tmp_path,
    )

    assert options.runtime.ligand_reference_conformer == "etkdg"
    assert options.runtime.etkdg_seed == 17


def test_screen_options_reject_unknown_ligand_conformer(tmp_path: Path) -> None:
    with pytest.raises(click.UsageError, match="must be skip or etkdg"):
        build_screen_options(
            {"ligand_reference_conformer": "rdkit_2d"},
            config_dir=tmp_path,
        )


def test_screen_options_reject_structure_route_options(tmp_path: Path) -> None:
    with pytest.raises(click.UsageError, match="Unknown Boltz-StFree options"):
        build_screen_options(
            {"affinity_distogram_modes": ["control"]},
            config_dir=tmp_path,
        )


def test_screen_options_reject_unsupported_accelerator_and_devices(
    tmp_path: Path,
) -> None:
    with pytest.raises(click.UsageError, match="gpu, cpu, or mps"):
        build_screen_options(
            {"accelerator": "tpu"},
            config_dir=tmp_path,
        )
    with pytest.raises(click.UsageError, match="must be 1"):
        build_screen_options(
            {"accelerator": "cpu", "devices": [0, 1]},
            config_dir=tmp_path,
        )


def test_runtime_options_do_not_change_prepared_input_identity(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"smiles": ["CCO"]},
        options={"override": False},
    )
    first = prepare_screen(config_path)
    first_manifest = first.manifest_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["options"]["override"] = True
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    second = prepare_screen(config_path)

    assert second.screen_options.runtime.override is True
    assert second.manifest_path.read_text(encoding="utf-8") == first_manifest


def test_preprocessing_options_are_part_of_prepared_input_identity(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        {"smiles": ["CCO"]},
        options={"max_msa_seqs": 1024},
    )
    prepare_screen(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["options"]["max_msa_seqs"] = 2048
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(click.UsageError, match="state does not match"):
        prepare_screen(config_path)


def test_cache_and_auto_msa_source_are_part_of_input_identity(tmp_path: Path) -> None:
    config = {
        "version": 1,
        "target": {"sequence": SEQUENCE, "msa": "auto"},
        "ligands": {"smiles": ["CCO"]},
        "output_dir": "./output",
        "options": {
            "cache": "./cache-one",
            "msa_server_url": "https://msa-one.example.test",
        },
    }
    config_path = tmp_path / "screen.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    prepare_screen(config_path)

    config["options"]["cache"] = "./cache-two"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(click.UsageError, match="state does not match"):
        prepare_screen(config_path)


def test_override_invalidation_removes_published_aggregates(tmp_path: Path) -> None:
    prepared = prepare_screen(
        _write_config(tmp_path, {"smiles": ["CCO"]}),
    )
    scores = prepared.output_dir / "scores.csv"
    summary = prepared.output_dir / "screen_summary.json"
    scores.write_text("old scores\n", encoding="utf-8")
    summary.write_text('{"completed": 1}\n', encoding="utf-8")

    invalidate_screen_aggregates(prepared)

    assert not scores.exists()
    assert not summary.exists()


def test_screen_cli_prepare_only_never_runs_inference(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"smiles": ["CCO", "CCN"]})

    result = CliRunner().invoke(cli, ["screen", str(config_path), "--prepare-only"])

    assert result.exit_code == 0, result.output
    assert "Prepared 1 chunk(s)" in result.output
    assert "Screen plan:" in result.output
    assert "Library: 2 records" in result.output
    assert "Manifest:" in result.output
    plan = json.loads((tmp_path / "output/screen_plan.json").read_text())
    assert plan["library"] == {
        "source": {"type": "smiles"},
        "records": 2,
        "ligand_tokens_min": 3,
        "ligand_tokens_max": 3,
        "above_training_token_limit": 0,
    }
    assert plan["execution"]["chunks"] == [{"name": "chunk_00000", "records": 2}]
    status = json.loads((tmp_path / "output/screen_status.json").read_text())
    assert status["state"] == "prepared"
    assert status["expected"] == 2
    assert status["completed"] == 0


def test_bootstrap_reference_prepare_only_stages_one_holo_input(
    tmp_path: Path,
) -> None:
    config = {
        "version": 1,
        "target": {"sequence": "A" * 30, "msa": "empty"},
        "anchor": {"id": "B", "smiles": "CCO"},
        "output_dir": "./reference-output",
        "options": {"diffusion_samples": 1, "seed": 7},
    }
    config_path = tmp_path / "bootstrap.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        ["bootstrap-reference", str(config_path), "--prepare-only"],
    )

    assert result.exit_code == 0, result.output
    assert "no model was loaded and no GPU work was started" in result.output
    output = tmp_path / "reference-output"
    staged = yaml.safe_load(
        (output / ".boltz-stfree/input/anchor.yaml").read_text(encoding="utf-8")
    )
    assert staged["sequences"] == [
        {
            "protein": {
                "id": "A",
                "sequence": "A" * 30,
                "msa": "empty",
            }
        },
        {"ligand": {"id": "B", "smiles": "CCO"}},
    ]
    status = json.loads(
        (output / "bootstrap_status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "prepared"
    assert not (output / "reference.pdb").exists()


def test_screen_cli_prepare_only_validates_stfree_options(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"smiles": ["CCO"]},
        options={"not_a_stfree_option": True},
    )

    result = CliRunner().invoke(cli, ["screen", str(config_path), "--prepare-only"])

    assert result.exit_code != 0
    assert "Unknown Boltz-StFree options" in result.output
    assert not (tmp_path / "output").exists()


def test_screen_progress_updates_are_idempotent(tmp_path: Path) -> None:
    prepared = prepare_screen(
        _write_config(
            tmp_path,
            {"smiles": ["CCO", "CCN"]},
            chunk_size=1,
        )
    )
    progress = ScreenProgress(prepared)
    manifest = json.loads(prepared.manifest_path.read_text())
    record_id = manifest["records"][0]["record_id"]

    progress.start_chunk("chunk_00000")
    progress.record_batch([record_id], False)
    progress.record_batch([record_id], False)
    progress.record_batch([record_id], True)
    failed = json.loads(prepared.status_path.read_text())
    progress.record_batch([record_id], False)
    recovered = json.loads(prepared.status_path.read_text())

    assert failed["completed"] == 0
    assert failed["failed"] == 1
    assert failed["chunks"][0]["failed"] == 1
    assert recovered["completed"] == 1
    assert recovered["failed"] == 0
    assert recovered["chunks"][0]["completed"] == 1


def test_screen_cli_runs_chunks_with_stfree_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(
        tmp_path,
        {"smiles": ["CCO", "CCN"]},
        options={"num_workers": 3},
        chunk_size=1,
    )
    calls: list[dict[str, object]] = []

    def fake_run_stfree_chunk(
        *,
        data: Path,
        out_dir: Path,
        options: object,
        receptor_crop_indices_dir: Path | None,
        progress_callback: Callable[[list[str], bool], None],
    ) -> None:
        calls.append(
            {
                "data": data,
                "out_dir": out_dir,
                "options": options,
                "receptor_crop_indices_dir": receptor_crop_indices_dir,
            }
        )
        data_dir = data
        prediction_root = out_dir / f"boltz_results_{data_dir.stem}" / "predictions"
        for input_path in data_dir.glob("*.yaml"):
            record_dir = prediction_root / input_path.stem
            record_dir.mkdir(parents=True)
            score_path = (
                record_dir / "affinity_B__distogram_zero_distogram_features.json"
            )
            score_path.write_text(
                json.dumps(
                    {
                        "affinity_pred_value": 1.0,
                        "affinity_probability_binary": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            progress_callback([input_path.stem], False)

    monkeypatch.setattr(stfree_runtime, "run_stfree_chunk", fake_run_stfree_chunk)

    result = CliRunner().invoke(cli, ["screen", str(config_path)])

    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert all(call["options"].num_workers == 3 for call in calls)
    assert (tmp_path / "output/scores.csv").is_file()
    assert "Screen complete" in result.output
    status = json.loads((tmp_path / "output/screen_status.json").read_text())
    assert status["state"] == "complete"
    assert status["completed"] == 2
    assert status["pending"] == 0
    assert all(chunk["state"] == "complete" for chunk in status["chunks"])


def test_screen_cli_publishes_failure_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, {"smiles": ["CCO"]})

    def fail_run(**kwargs: object) -> None:  # noqa: ARG001
        raise RuntimeError("GPU stopped")

    monkeypatch.setattr(stfree_runtime, "run_stfree_chunk", fail_run)

    result = CliRunner().invoke(cli, ["screen", str(config_path)])

    assert result.exit_code == 1
    status = json.loads((tmp_path / "output/screen_status.json").read_text())
    assert status["state"] == "failed"
    assert status["completed"] == 0
    assert status["message"] == "RuntimeError: GPU stopped"


def test_cli_exposes_product_commands_without_upstream_predict() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "screen" in result.output
    assert "bootstrap-reference" in result.output
    assert "predict" not in result.output


def test_collect_screen_scores_preserves_library_identity(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"smiles": [{"id": "first", "smiles": "CCO"}, "CCN"]},
        chunk_size=1,
    )
    prepared = prepare_screen(config_path)
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    for index, record in enumerate(manifest["records"], start=1):
        prediction_dir = (
            prepared.runs_dir
            / f"boltz_results_{record['chunk']}"
            / "predictions"
            / record["record_id"]
        )
        prediction_dir.mkdir(parents=True)
        payload = {
            "affinity_pred_value": float(index),
            "affinity_probability_binary": index / 10,
        }
        score_path = (
            prediction_dir / "affinity_B__distogram_zero_distogram_features.json"
        )
        score_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    scores_path = collect_screen_scores(prepared)

    with scores_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["ligand_id"] for row in rows] == ["first", "ligand_000002"]
    assert [row["affinity_pred_value"] for row in rows] == ["1.0", "2.0"]

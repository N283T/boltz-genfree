"""CPU adapter integrity, module isolation, route/target and resume tests."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import click
import pytest
import torch
from click.testing import CliRunner

from boltz.affinity_adapter import (
    ROUTE_KEYS,
    apply_adapter,
    import_research_heads,
    load_adapter,
    validate_contract,
    validate_metadata,
)
from boltz.main import cli
from boltz.screen import build_screen_options, prepare_screen
from boltz.stfree_runtime import (
    StFreeRuntimeOptions,
    _patch_torch_checkpoint_loading,
    _run_provenance,
    _validate_resume_state,
)
from test_screen import SEQUENCE, _write_config
from test_stfree_runtime import _record, _write_existing_score
from boltz.data.types import Manifest


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.affinity_module1 = torch.nn.Linear(3, 2)
        self.affinity_module2 = torch.nn.Linear(3, 2)
        self.trunk = torch.nn.Linear(3, 3)


@pytest.fixture
def example(tmp_path):
    options = StFreeRuntimeOptions(
        cache=tmp_path,
        seed=17,
        target_sequence_sha256=hashlib.sha256(SEQUENCE.encode()).hexdigest(),
    )
    metadata = dict(
        base_checkpoint_sha256="a" * 64,
        target=dict(name="CPU fixture", sequence_sha256=options.target_sequence_sha256),
        feature_route={key: getattr(options, key) for key in ROUTE_KEYS},
        receptor_context=dict(mode="complete_input"),
        training=dict(
            seed=0,
            label_split_sha256="b" * 64,
            source_plan_sha256="c" * 64,
            frozen_upstream=True,
            input_context="independent_genfree",
        ),
        distogram_mode="zero_distogram_features",
        structure_generation_active=False,
    )
    model = TinyModel()
    state = {
        f"members.{i}.{key}": value.detach().clone() + 0.25
        for i, module in enumerate((model.affinity_module1, model.affinity_module2))
        for key, value in module.state_dict().items()
    }
    checkpoint = tmp_path / "final_heads.pt"
    torch.save(
        dict(state_dict=state, base_checkpoint_sha256="a" * 64, plan_sha256="c" * 64),
        checkpoint,
    )
    adapter = import_research_heads(checkpoint, metadata, tmp_path / "adapter")
    return model, options, metadata, checkpoint, adapter


def test_roundtrip_changes_only_affinity_modules_on_cpu(example):
    model, options, metadata, checkpoint, adapter = example
    before = {key: value.clone() for key, value in model.state_dict().items()}
    original_checkpoint = checkpoint.read_bytes()
    validate_contract(adapter, options, base_checkpoint_sha256="a" * 64)
    apply_adapter(model, load_adapter(adapter.path))
    for key, value in model.state_dict().items():
        assert torch.equal(
            value,
            before[key] + 0.25 if key.startswith("affinity_module") else before[key],
        )
        assert value.device.type == "cpu"
    assert checkpoint.read_bytes() == original_checkpoint
    assert load_adapter(adapter.path).identity() == adapter.identity()


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 18),
        ("affinity_recycling_steps", 1),
        ("affinity_conditioner_mode", "folded"),
        ("ligand_reference_conformer", "rdkit_2d"),
        ("affinity_mw_correction", True),
        ("num_subsampled_msa", 512),
        ("max_msa_seqs", 100),
        ("subsample_msa", False),
        ("etkdg_seed", 99),
    ],
)
def test_rejects_changed_feature_route(example, field, value):
    _, options, _, _, adapter = example
    with pytest.raises(ValueError, match="feature route mismatch"):
        validate_contract(adapter, replace(options, **{field: value}))


def test_rejects_wrong_base_target_reference_and_unverified_target(example):
    _, options, _, _, adapter = example
    with pytest.raises(ValueError, match="base checkpoint"):
        validate_contract(adapter, options, base_checkpoint_sha256="d" * 64)
    for digest in ("d" * 64, None):
        with pytest.raises(ValueError, match="target sequence"):
            validate_contract(adapter, replace(options, target_sequence_sha256=digest))
    with pytest.raises(ValueError, match="receptor reference"):
        validate_contract(adapter, replace(options, receptor_reference_sha256="e" * 64))


@pytest.mark.parametrize("failure", ["name", "shape", "dtype"])
def test_model_validation_is_complete_before_any_tensor_mutation(example, failure):
    model, _, _, _, adapter = example
    before = {key: value.clone() for key, value in model.state_dict().items()}
    key = "affinity_module2.weight"
    if failure == "name":
        adapter.state.pop(key)
    if failure == "shape":
        adapter.state[key] = torch.zeros((9, 3))
    if failure == "dtype":
        adapter.state[key] = adapter.state[key].double()
    with pytest.raises(ValueError):
        apply_adapter(model, adapter)
    assert all(
        torch.equal(before[key], value) for key, value in model.state_dict().items()
    )


def test_rejects_hash_corruption_before_unpickling(example, monkeypatch):
    *_, adapter = example
    (adapter.path / "weights.pt").write_bytes(b"not a checkpoint")
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("must check hash first")
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_adapter(adapter.path)


def test_rejects_inventory_mismatch_unknown_schema_and_generated_context(example):
    *_, adapter = example
    path = adapter.path / "manifest.json"
    original = json.loads(path.read_text())
    for update, match in [
        ({"schema_version": 2}, "schema"),
        ({"structure_generation_active": True}, "no candidate"),
        ({"tensors": {}}, "inventory"),
    ]:
        path.write_text(json.dumps(dict(original, **update)))
        with pytest.raises(ValueError, match=match):
            load_adapter(adapter.path)
    path.write_text(json.dumps(original))


def test_nonaffinity_nonfinite_and_plan_mismatch_rejected_without_output(
    example, tmp_path
):
    _, _, metadata, checkpoint, _ = example
    saved = torch.load(checkpoint, weights_only=True)
    destination = tmp_path / "bad-adapter"
    for mode in ("trunk", "nan", "plan"):
        bad = copy.deepcopy(saved)
        if mode == "trunk":
            bad["state_dict"]["trunk.weight"] = torch.zeros((3, 3))
        if mode == "nan":
            bad["state_dict"]["members.0.weight"][0, 0] = float("nan")
        if mode == "plan":
            bad["plan_sha256"] = "d" * 64
        path = tmp_path / "bad.pt"
        torch.save(bad, path)
        with pytest.raises(ValueError):
            import_research_heads(path, metadata, destination)
        assert not destination.exists()


def test_import_never_overwrites_existing_adapter(example):
    _, _, metadata, checkpoint, adapter = example
    before = {p.name: p.read_bytes() for p in adapter.path.iterdir()}
    with pytest.raises(FileExistsError):
        import_research_heads(checkpoint, metadata, adapter.path)
    assert before == {p.name: p.read_bytes() for p in adapter.path.iterdir()}


def test_cli_import_and_inspect_are_model_free(example, tmp_path, monkeypatch):
    _, _, metadata, checkpoint, adapter = example
    import boltz.main as main

    monkeypatch.setattr(
        main, "download_stfree", lambda *a, **k: pytest.fail("no model download")
    )
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps(metadata))
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "adapter",
            "import-heads",
            "--checkpoint",
            str(checkpoint),
            "--metadata",
            str(meta),
            "--output",
            str(tmp_path / "cli-adapter"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["inference_performed"] is False
    inspected = runner.invoke(cli, ["adapter", "inspect", str(adapter.path)])
    assert inspected.exit_code == 0, inspected.output
    assert json.loads(inspected.output)["tensor_count"] == 4


def test_screen_prepare_binds_target_and_adapter_option(example, tmp_path):
    _, options, _, _, adapter = example
    path = _write_config(
        tmp_path,
        {"smiles": ["CCO"]},
        options={"seed": 17, "affinity_adapter": "adapter"},
    )
    prepared = prepare_screen(path)
    assert prepared.screen_options.runtime.affinity_adapter == adapter.path
    assert (
        prepared.screen_options.runtime.target_sequence_sha256
        == options.target_sequence_sha256
    )
    wrong = _write_config(
        tmp_path,
        {"smiles": ["CCO"]},
        sequence="ACDEFG",
        options={"seed": 17, "affinity_adapter": "adapter"},
    )
    with pytest.raises(click.UsageError, match="different target"):
        prepare_screen(wrong)
    with pytest.raises(click.UsageError, match="seed"):
        build_screen_options({"affinity_adapter": "adapter"}, config_dir=tmp_path)


def test_adapter_identity_protects_resume_and_noft_has_no_adapter_key(
    example, tmp_path
):
    _, options, _, _, adapter = example
    noft = _run_provenance(
        options, runtime_version="fixture", checkpoint_sha256="a" * 64
    )
    assert "affinity_adapter" not in noft
    ftoptions = replace(options, affinity_adapter=adapter.path)
    ft = _run_provenance(
        ftoptions, runtime_version="fixture", checkpoint_sha256="a" * 64
    )
    assert ft["affinity_adapter"] == adapter.identity()
    manifest = Manifest([_record()])
    _write_existing_score(tmp_path, manifest, ft)
    with pytest.raises(click.ClickException, match="different checkpoint or runtime"):
        _validate_resume_state(
            manifest=manifest, run_dir=tmp_path, provenance=noft, override=False
        )
    changed = dict(
        ft, affinity_adapter=dict(ft["affinity_adapter"], weights_sha256="d" * 64)
    )
    with pytest.raises(click.ClickException, match="different checkpoint or runtime"):
        _validate_resume_state(
            manifest=manifest, run_dir=tmp_path, provenance=changed, override=False
        )


def test_trusted_lightning_shim_respects_explicit_restricted_load(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: calls.append(k))
    _patch_torch_checkpoint_loading(torch)
    torch.load("unused", weights_only=True)
    torch.load("unused")
    assert calls[0]["weights_only"] is True
    assert calls[1]["weights_only"] is False


def test_real_runtime_session_hook_applies_adapter_before_scoring(
    example, tmp_path, monkeypatch
):
    model, options, metadata, checkpoint, _ = example
    import boltz.main as main
    import boltz.model.models.boltz2 as boltz2
    from boltz.stfree_runtime import create_stfree_runtime_session

    base = tmp_path / "base.ckpt"
    base.write_bytes(b"CPU fixture base; never loaded as a real model")
    base_hash = hashlib.sha256(base.read_bytes()).hexdigest()
    metadata = copy.deepcopy(metadata)
    metadata["base_checkpoint_sha256"] = base_hash
    saved = torch.load(checkpoint, weights_only=True)
    saved["base_checkpoint_sha256"] = base_hash
    checkpoint2 = tmp_path / "heads2.pt"
    torch.save(saved, checkpoint2)
    adapter = import_research_heads(checkpoint2, metadata, tmp_path / "adapter2")
    before = {key: value.clone() for key, value in model.state_dict().items()}
    for flag in (
        "affinity_coordinate_free",
        "skip_run_structure",
        "skip_confidence_forward",
        "skip_pdistogram_forward",
    ):
        setattr(model, flag, True)
    monkeypatch.setattr(main, "download_stfree", lambda *a, **k: None)
    calls = []

    def load_model(path, **kwargs):
        assert path == base
        assert kwargs["map_location"] == "cpu"
        assert kwargs["skip_run_structure"] is True
        calls.append(kwargs)
        return model

    monkeypatch.setattr(boltz2.Boltz2, "load_from_checkpoint", load_model)
    options = replace(
        options,
        affinity_checkpoint=base,
        affinity_adapter=adapter.path,
        accelerator="cpu",
        num_workers=0,
    )
    with torch.enable_grad():
        session = create_stfree_runtime_session(options)
    torch.set_grad_enabled(True)
    assert len(calls) == 1
    assert session.adapter_identity == adapter.identity()
    for key, value in model.state_dict().items():
        assert torch.equal(
            value,
            before[key] + 0.25 if key.startswith("affinity_module") else before[key],
        )


def test_reference_contract_binds_crop_parameters_not_only_reference_file(
    example, tmp_path
):
    _, options, metadata, checkpoint, _ = example
    metadata = copy.deepcopy(metadata)
    metadata["receptor_context"] = dict(
        mode="reference_contiguous_blocks",
        reference_sha256="d" * 64,
        crop_config_sha256="e" * 64,
    )
    adapter = import_research_heads(checkpoint, metadata, tmp_path / "cropped")
    correct = replace(
        options,
        receptor_reference_sha256="d" * 64,
        receptor_crop_config_sha256="e" * 64,
    )
    validate_contract(adapter, correct)
    with pytest.raises(ValueError, match="crop configuration"):
        validate_contract(
            adapter, replace(correct, receptor_crop_config_sha256="f" * 64)
        )


class UnsafePickle:
    def __reduce__(self):
        # Harmless arithmetic, but any eval deserialization must be rejected.
        return eval, ("2 + 2",)


def test_trusted_lightning_shim_handles_explicit_unspecified_default(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "load", lambda *a, **kw: calls.append(kw))
    _patch_torch_checkpoint_loading(torch)
    torch.load("unused", weights_only=None)
    torch.load("unused", weights_only=True)
    torch.load("unused", weights_only=False)
    assert [call["weights_only"] for call in calls] == [False, True, False]


def test_adapter_restricted_unpickler_is_not_bypassed_by_runtime_shim(
    example, monkeypatch
):
    *_, adapter = example
    payload = dict(adapter.state)
    payload["affinity_module1.weight"] = UnsafePickle()
    torch.save({"state_dict": payload}, adapter.path / "weights.pt")
    manifest = json.loads((adapter.path / "manifest.json").read_text())
    manifest["weights_sha256"] = hashlib.sha256(
        (adapter.path / "weights.pt").read_bytes()
    ).hexdigest()
    (adapter.path / "manifest.json").write_text(json.dumps(manifest))
    # Restore torch.load after testing the trusted-checkpoint shim.
    with monkeypatch.context() as ctx:
        ctx.setattr(torch, "load", torch.load)
        _patch_torch_checkpoint_loading(torch)
        with pytest.raises(ValueError, match="Restricted tensor load failed"):
            load_adapter(adapter.path)


def test_duplicate_manifest_keys_are_rejected(example):
    *_, adapter = example
    path = adapter.path / "manifest.json"
    text = path.read_text()
    path.write_text(text.replace("{", '{"schema_version": 9,', 1))
    with pytest.raises(ValueError, match="Invalid adapter manifest"):
        load_adapter(adapter.path)


def test_nonfinite_in_memory_state_rejected_before_model_mutation(example):
    model, _, _, _, adapter = example
    before = {key: value.clone() for key, value in model.state_dict().items()}
    adapter.state["affinity_module2.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        apply_adapter(model, adapter)
    assert all(
        torch.equal(before[key], value) for key, value in model.state_dict().items()
    )

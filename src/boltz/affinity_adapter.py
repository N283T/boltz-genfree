"""Versioned affinity-only GenFree adapters with fail-closed CPU validation.

A manifest describes declared training provenance. Hashes prove artifact
identity, not that arbitrary user-supplied provenance is scientifically true.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

KIND = "boltz_genfree_affinity_adapter"
PREFIXES = ("affinity_module1.", "affinity_module2.")
ROUTE_KEYS = (
    "affinity_recycling_steps",
    "ligand_reference_conformer",
    "affinity_conditioner_mode",
    "affinity_mw_correction",
    "subsample_msa",
    "num_subsampled_msa",
    "max_msa_seqs",
    "seed",
    "etkdg_seed",
)
_MAX_WEIGHTS_BYTES = 128 * 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class AffinityAdapter:
    path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    weights_sha256: str
    state: dict[str, torch.Tensor]

    def identity(self) -> dict[str, object]:
        """Immutable adapter identity for resume checks (no machine-local path)."""
        return {
            "kind": KIND,
            "schema_version": 1,
            "manifest_sha256": self.manifest_sha256,
            "weights_sha256": self.weights_sha256,
            "base_checkpoint_sha256": self.manifest["base_checkpoint_sha256"],
            "target": self.manifest["target"],
        }


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Validate the scientifically relevant contract before loading weights."""
    _sha(metadata.get("base_checkpoint_sha256"), "base_checkpoint_sha256")
    target = _mapping(metadata.get("target"), "target")
    _sha(target.get("sequence_sha256"), "target.sequence_sha256")
    if not isinstance(target.get("name"), str) or not target["name"].strip():
        raise ValueError("target.name must be nonempty")
    route = _mapping(metadata.get("feature_route"), "feature_route")
    if set(route) != set(ROUTE_KEYS):
        raise ValueError(f"feature_route requires exactly: {', '.join(ROUTE_KEYS)}")
    if (
        type(route["affinity_recycling_steps"]) is not int
        or not 0 <= route["affinity_recycling_steps"] <= 5
    ):
        raise ValueError("Invalid affinity_recycling_steps")
    if route["ligand_reference_conformer"] not in {"skip", "rdkit_2d", "etkdg"}:
        raise ValueError("Invalid ligand_reference_conformer")
    if route["affinity_conditioner_mode"] not in {"original", "folded"}:
        raise ValueError("Invalid affinity_conditioner_mode")
    for key in ("affinity_mw_correction", "subsample_msa"):
        if type(route[key]) is not bool:
            raise ValueError(f"feature_route.{key} must be boolean")
    for key in ("num_subsampled_msa", "max_msa_seqs"):
        if type(route[key]) is not int or route[key] < 1:
            raise ValueError(f"feature_route.{key} must be a positive integer")
    if type(route["seed"]) is not int:
        raise ValueError("An explicit integer input seed is required")
    if (
        type(route["etkdg_seed"]) is not int
        or not 0 <= route["etkdg_seed"] <= 2_147_483_647
    ):
        raise ValueError("Invalid etkdg_seed")
    receptor = _mapping(metadata.get("receptor_context"), "receptor_context")
    if receptor.get("mode") not in {"complete_input", "reference_contiguous_blocks"}:
        raise ValueError("Invalid receptor_context.mode")
    if receptor["mode"] == "reference_contiguous_blocks":
        _sha(receptor.get("reference_sha256"), "receptor_context.reference_sha256")
        _sha(receptor.get("crop_config_sha256"), "receptor_context.crop_config_sha256")
    elif (
        receptor.get("reference_sha256") is not None
        or receptor.get("crop_config_sha256") is not None
    ):
        raise ValueError("Complete input must not declare a crop reference")
    training = _mapping(metadata.get("training"), "training")
    if type(training.get("seed")) is not int:
        raise ValueError("training.seed must be an integer")
    _sha(training.get("label_split_sha256"), "training.label_split_sha256")
    _sha(training.get("source_plan_sha256"), "training.source_plan_sha256")
    if (
        training.get("frozen_upstream") is not True
        or training.get("input_context") != "independent_genfree"
    ):
        raise ValueError(
            "Only frozen-upstream, independent GenFree contexts are supported"
        )
    if (
        metadata.get("distogram_mode") != "zero_distogram_features"
        or metadata.get("structure_generation_active") is not False
    ):
        raise ValueError(
            "Adapter must declare zero distogram and no candidate structure generation"
        )


def _validate_state(state: object) -> dict[str, torch.Tensor]:
    values = _mapping(state, "state_dict")
    if not values or not all(
        isinstance(key, str) and key.startswith(PREFIXES) for key in values
    ):
        raise ValueError("Adapter may contain only affinity_module1/2 tensors")
    if not all(any(k.startswith(prefix) for k in values) for prefix in PREFIXES):
        raise ValueError("Both complete affinity modules are required")
    for key, value in values.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.is_complex()
        ):
            raise ValueError(f"Invalid adapter tensor: {key}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Nonfinite adapter tensor: {key}")
    return values


def _specs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, object]]:
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(state.items())
    }


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate adapter JSON key: {key}")
        result[key] = value
    return result


def _restricted_load(raw: bytes) -> object:
    try:
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Restricted tensor load failed: {error}") from error


def load_adapter(path: Path) -> AffinityAdapter:
    """Load and hash the same bytes using PyTorch's restricted unpickler."""
    path = path.expanduser().resolve()
    manifest_path, weights_path = path / "manifest.json", path / "weights.pt"
    if not path.is_dir() or not manifest_path.is_file() or not weights_path.is_file():
        raise ValueError(
            f"Adapter must be a directory with manifest.json and weights.pt: {path}"
        )
    if (
        manifest_path.stat().st_size > _MAX_MANIFEST_BYTES
        or weights_path.stat().st_size > _MAX_WEIGHTS_BYTES
    ):
        raise ValueError("Adapter exceeds size limit")
    manifest_bytes, weights_bytes = (
        manifest_path.read_bytes(),
        weights_path.read_bytes(),
    )
    try:
        manifest = _mapping(
            json.loads(manifest_bytes, object_pairs_hook=_unique_json_object),
            "manifest",
        )
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("Invalid adapter manifest JSON") from error
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("kind") != KIND
    ):
        raise ValueError("Unsupported GenFree adapter schema/kind")
    validate_metadata(manifest)
    weights_hash = hashlib.sha256(weights_bytes).hexdigest()
    if _sha(manifest.get("weights_sha256"), "weights_sha256") != weights_hash:
        raise ValueError("Adapter weights SHA256 mismatch")
    # Always explicit True, including when Lightning's trusted-load shim is active.
    saved = _restricted_load(weights_bytes)
    if not isinstance(saved, dict) or set(saved) != {"state_dict"}:
        raise ValueError("Adapter weights require only a state_dict")
    state = _validate_state(saved["state_dict"])
    if manifest.get("tensors") != _specs(state):
        raise ValueError("Adapter tensor inventory mismatch")
    return AffinityAdapter(
        path, manifest, hashlib.sha256(manifest_bytes).hexdigest(), weights_hash, state
    )


def validate_contract(
    adapter: AffinityAdapter,
    options: object,
    *,
    base_checkpoint_sha256: str | None = None,
    require_target: bool = True,
) -> None:
    """Reject incompatible base weights, target, receptor context and input route."""
    manifest = adapter.manifest
    if (
        base_checkpoint_sha256 is not None
        and manifest["base_checkpoint_sha256"] != base_checkpoint_sha256
    ):
        raise ValueError("Adapter base checkpoint SHA256 mismatch")
    for key in ROUTE_KEYS:
        actual = getattr(options, key)
        if (
            type(actual) is not type(manifest["feature_route"][key])
            or actual != manifest["feature_route"][key]
        ):
            raise ValueError(f"Adapter feature route mismatch: {key}")
    if require_target:
        target_hash = getattr(options, "target_sequence_sha256", None)
        if target_hash != manifest["target"]["sequence_sha256"]:
            raise ValueError(
                "Adapter target sequence SHA256 mismatch or unverified target"
            )
        reference = getattr(options, "receptor_reference_sha256", None)
        receptor = manifest["receptor_context"]
        expected = receptor.get("reference_sha256")
        if reference != expected:
            raise ValueError("Adapter receptor reference SHA256 mismatch")
        if getattr(options, "receptor_crop_config_sha256", None) != receptor.get(
            "crop_config_sha256"
        ):
            raise ValueError("Adapter receptor crop configuration SHA256 mismatch")


def apply_adapter(model: object, adapter: AffinityAdapter) -> None:
    """Validate both modules completely before touching any model tensor."""
    _validate_state(adapter.state)
    modules = [
        getattr(model, "affinity_module1", None),
        getattr(model, "affinity_module2", None),
    ]
    if any(not isinstance(module, torch.nn.Module) for module in modules):
        raise ValueError("Model must contain both affinity modules")
    expected = {
        prefix + key: value
        for prefix, module in zip(PREFIXES, modules, strict=True)
        for key, value in module.state_dict().items()
    }
    if set(adapter.state) != set(expected):
        raise ValueError("Adapter/model affinity tensor names mismatch")
    for key, value in expected.items():
        actual = adapter.state[key]
        if actual.shape != value.shape or actual.dtype != value.dtype:
            raise ValueError(f"Adapter/model tensor shape or dtype mismatch: {key}")
    for prefix, module in zip(PREFIXES, modules, strict=True):
        module.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in adapter.state.items()
                if k.startswith(prefix)
            },
            strict=True,
        )


def import_research_heads(
    checkpoint: Path, metadata: dict[str, Any], destination: Path
) -> AffinityAdapter:
    """Convert the existing research head-only format without invoking a model."""
    validate_metadata(metadata)
    forbidden = {
        "kind",
        "schema_version",
        "weights_sha256",
        "tensors",
        "source_checkpoint_sha256",
    }
    if forbidden & metadata.keys():
        raise ValueError("Metadata must not override generated adapter identity fields")
    if checkpoint.stat().st_size > _MAX_WEIGHTS_BYTES:
        raise ValueError("Research checkpoint exceeds size limit")
    raw = checkpoint.read_bytes()
    saved = _restricted_load(raw)
    if (
        not isinstance(saved, dict)
        or saved.get("base_checkpoint_sha256") != metadata["base_checkpoint_sha256"]
    ):
        raise ValueError("Research checkpoint/base metadata mismatch")
    if saved.get("plan_sha256") != metadata["training"]["source_plan_sha256"]:
        raise ValueError("Research checkpoint/plan metadata mismatch")
    original = _mapping(saved.get("state_dict"), "research state_dict")
    state = {}
    for key, value in original.items():
        if not isinstance(key, str) or not key.startswith(("members.0.", "members.1.")):
            raise ValueError("Research weights contain non-affinity tensors")
        prefix = "members.0." if key.startswith("members.0.") else "members.1."
        renamed = (PREFIXES[0] if prefix == "members.0." else PREFIXES[1]) + key[
            len(prefix) :
        ]
        state[renamed] = (
            value.detach().cpu().contiguous()
            if isinstance(value, torch.Tensor)
            else value
        )
    state = _validate_state(state)
    buffer = io.BytesIO()
    torch.save({"state_dict": state}, buffer)
    weights = buffer.getvalue()
    if len(weights) > _MAX_WEIGHTS_BYTES:
        raise ValueError("Adapter exceeds size limit")
    manifest = dict(
        metadata,
        kind=KIND,
        schema_version=1,
        weights_sha256=hashlib.sha256(weights).hexdigest(),
        tensors=_specs(state),
        source_checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
    )
    destination = destination.expanduser().resolve()
    # Never overwrite an existing adapter or checkpoint.
    destination.mkdir(parents=True, exist_ok=False)
    try:
        (destination / "weights.pt").write_bytes(weights)
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        return load_adapter(destination)
    except Exception:
        for name in ("weights.pt", "manifest.json"):
            (destination / name).unlink(missing_ok=True)
        destination.rmdir()
        raise

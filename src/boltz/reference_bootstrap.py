"""Create a reusable receptor-crop reference from one predicted holo complex."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import click
import yaml
from rdkit import Chem

_TOP_LEVEL_KEYS = {"version", "target", "anchor", "output_dir", "options"}
_TARGET_KEYS = {"id", "sequence", "msa"}
_ANCHOR_KEYS = {"id", "smiles"}
_OPTION_KEYS = {
    "cache",
    "checkpoint",
    "devices",
    "accelerator",
    "num_workers",
    "recycling_steps",
    "sampling_steps",
    "diffusion_samples",
    "max_parallel_samples",
    "step_scale",
    "seed",
    "preprocessing_threads",
    "max_msa_seqs",
    "no_kernels",
    "flash_attn",
    "msa_server_url",
    "msa_pairing_strategy",
}


@dataclass(frozen=True)
class BootstrapOptions:
    """Validated options for one Boltz-2 holo prediction."""

    cache: Path
    checkpoint: Path
    devices: int | list[int]
    accelerator: str
    num_workers: int
    recycling_steps: int
    sampling_steps: int
    diffusion_samples: int
    max_parallel_samples: int | None
    step_scale: float
    seed: int | None
    preprocessing_threads: int
    max_msa_seqs: int
    no_kernels: bool
    flash_attn: bool
    msa_server_url: str
    msa_pairing_strategy: str


@dataclass(frozen=True)
class PreparedBootstrapReference:
    """Prepared paths and normalized settings for a reference bootstrap."""

    config_path: Path
    output_dir: Path
    input_path: Path
    runtime_dir: Path
    reference_path: Path
    manifest_path: Path
    plan_path: Path
    status_path: Path
    target_sequence: str
    target_id: str
    anchor_id: str
    auto_msa: bool
    options: BootstrapOptions


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise click.UsageError(f"{label} must be a YAML mapping.")
    return value


def _resolve_path(value: object, base_dir: Path, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise click.UsageError(f"{label} must be a non-empty path.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise click.UsageError(f"{label} must be a {qualifier} integer.")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise click.UsageError(f"{label} must be true or false.")
    return value


def _validate_devices(value: object) -> int | list[int]:
    if isinstance(value, bool):
        raise click.UsageError("options.devices must be a positive integer or list.")
    if isinstance(value, int):
        return _positive_int(value, "options.devices")
    if isinstance(value, list) and value and all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in value
    ):
        if len(set(value)) != len(value):
            raise click.UsageError("options.devices must not contain duplicates.")
        return value
    raise click.UsageError(
        "options.devices must be a positive integer or a list of GPU indices."
    )


def _build_options(raw: dict[str, Any], base_dir: Path) -> BootstrapOptions:
    unknown = set(raw) - _OPTION_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise click.UsageError(f"Unknown bootstrap options: {names}")

    cache = _resolve_path(raw.get("cache", "~/.boltz"), base_dir, "options.cache")
    checkpoint = _resolve_path(
        raw.get("checkpoint", cache / "boltz2_conf.ckpt"),
        base_dir,
        "options.checkpoint",
    )
    accelerator = raw.get("accelerator", "gpu")
    if accelerator not in {"gpu", "cpu", "mps"}:
        raise click.UsageError("options.accelerator must be gpu, cpu, or mps.")
    step_scale = raw.get("step_scale", 1.5)
    if isinstance(step_scale, bool) or not isinstance(step_scale, (int, float)):
        raise click.UsageError("options.step_scale must be a positive number.")
    step_scale = float(step_scale)
    if step_scale <= 0:
        raise click.UsageError("options.step_scale must be a positive number.")
    seed = raw.get("seed", 20260718)
    if seed is not None:
        seed = _positive_int(seed, "options.seed", allow_zero=True)
    max_parallel = raw.get("max_parallel_samples")
    if max_parallel is not None:
        max_parallel = _positive_int(
            max_parallel,
            "options.max_parallel_samples",
        )
    msa_server_url = raw.get("msa_server_url", "https://api.colabfold.com")
    msa_pairing_strategy = raw.get("msa_pairing_strategy", "greedy")
    if not isinstance(msa_server_url, str) or not msa_server_url.strip():
        raise click.UsageError("options.msa_server_url must be a non-empty string.")
    if not isinstance(msa_pairing_strategy, str) or not msa_pairing_strategy.strip():
        raise click.UsageError(
            "options.msa_pairing_strategy must be a non-empty string."
        )
    return BootstrapOptions(
        cache=cache,
        checkpoint=checkpoint,
        devices=_validate_devices(raw.get("devices", 1)),
        accelerator=accelerator,
        num_workers=_positive_int(
            raw.get("num_workers", 1),
            "options.num_workers",
            allow_zero=True,
        ),
        recycling_steps=_positive_int(
            raw.get("recycling_steps", 3),
            "options.recycling_steps",
            allow_zero=True,
        ),
        sampling_steps=_positive_int(
            raw.get("sampling_steps", 200),
            "options.sampling_steps",
        ),
        diffusion_samples=_positive_int(
            raw.get("diffusion_samples", 1),
            "options.diffusion_samples",
        ),
        max_parallel_samples=max_parallel,
        step_scale=step_scale,
        seed=seed,
        preprocessing_threads=_positive_int(
            raw.get("preprocessing_threads", 1),
            "options.preprocessing_threads",
        ),
        max_msa_seqs=_positive_int(
            raw.get("max_msa_seqs", 8192),
            "options.max_msa_seqs",
        ),
        no_kernels=_boolean(raw.get("no_kernels", False), "options.no_kernels"),
        flash_attn=_boolean(raw.get("flash_attn", False), "options.flash_attn"),
        msa_server_url=msa_server_url.strip(),
        msa_pairing_strategy=msa_pairing_strategy.strip(),
    )


def prepare_bootstrap_reference(config_path: Path) -> PreparedBootstrapReference:
    """Validate configuration and stage one protein-anchor Boltz input."""
    config_path = config_path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = _mapping(yaml.safe_load(handle), "bootstrap config")
    unknown = set(config) - _TOP_LEVEL_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise click.UsageError(f"Unknown bootstrap config keys: {names}")
    if config.get("version", 1) != 1:
        raise click.UsageError("Only bootstrap config version 1 is supported.")
    base_dir = config_path.parent

    target = _mapping(config.get("target"), "target")
    unknown_target = set(target) - _TARGET_KEYS
    if unknown_target:
        names = ", ".join(sorted(unknown_target))
        raise click.UsageError(f"Unknown target keys: {names}")
    sequence_value = target.get("sequence")
    if not isinstance(sequence_value, str):
        raise click.UsageError("target.sequence must be a protein sequence string.")
    sequence = re.sub(r"\s+", "", sequence_value).upper()
    if not sequence or not re.fullmatch(r"[A-Z]+", sequence):
        raise click.UsageError("target.sequence must contain amino-acid letters.")
    target_id = target.get("id", "A")
    if (
        not isinstance(target_id, str)
        or not target_id.strip()
        or len(target_id.strip()) != 1
    ):
        raise click.UsageError("target.id must be a one-character chain ID.")
    target_id = target_id.strip()

    anchor = _mapping(config.get("anchor"), "anchor")
    unknown_anchor = set(anchor) - _ANCHOR_KEYS
    if unknown_anchor:
        names = ", ".join(sorted(unknown_anchor))
        raise click.UsageError(f"Unknown anchor keys: {names}")
    anchor_id = anchor.get("id", "B")
    if (
        not isinstance(anchor_id, str)
        or not anchor_id.strip()
        or len(anchor_id.strip()) != 1
    ):
        raise click.UsageError("anchor.id must be a one-character chain ID.")
    anchor_id = anchor_id.strip()
    if anchor_id == target_id:
        raise click.UsageError("target.id and anchor.id must be different.")
    smiles = anchor.get("smiles")
    if not isinstance(smiles, str) or not smiles.strip():
        raise click.UsageError("anchor.smiles must be a non-empty SMILES string.")
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        raise click.UsageError("anchor.smiles is not a valid SMILES string.")
    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

    msa_value = target.get("msa", "auto")
    auto_msa = msa_value in ("auto", None)
    if auto_msa:
        normalized_msa: str | None = None
        msa_provenance: dict[str, object] = {"mode": "auto"}
    elif msa_value == "empty":
        normalized_msa = "empty"
        msa_provenance = {"mode": "empty"}
    else:
        msa_path = _resolve_path(msa_value, base_dir, "target.msa")
        if not msa_path.is_file():
            raise click.UsageError(f"Target MSA does not exist: {msa_path}")
        normalized_msa = str(msa_path)
        msa_provenance = {
            "mode": "file",
            "path": normalized_msa,
            "sha256": _sha256(msa_path),
        }

    options = _build_options(_mapping(config.get("options", {}), "options"), base_dir)
    output_dir = _resolve_path(
        config.get("output_dir", f"{config_path.stem}_reference"),
        base_dir,
        "output_dir",
    )
    state_dir = output_dir / ".boltz-stfree"
    input_path = state_dir / "input" / "anchor.yaml"
    runtime_dir = state_dir / "runtime"
    reference_path = output_dir / "reference.pdb"
    manifest_path = state_dir / "bootstrap_manifest.json"
    plan_path = output_dir / "bootstrap_plan.json"
    status_path = output_dir / "bootstrap_status.json"

    protein: dict[str, object] = {"id": target_id, "sequence": sequence}
    if normalized_msa is not None:
        protein["msa"] = normalized_msa
    payload = {
        "version": 1,
        "sequences": [
            {"protein": protein},
            {"ligand": {"id": anchor_id, "smiles": canonical_smiles}},
        ],
    }
    rendered = yaml.safe_dump(payload, sort_keys=False)

    serializable_options = {
        **asdict(options),
        "cache": str(options.cache),
        "checkpoint": str(options.checkpoint),
    }
    normalized_config = {
        "target": {
            "id": target_id,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "length": len(sequence),
            "msa": msa_provenance,
        },
        "anchor": {"id": anchor_id, "smiles": canonical_smiles},
        "options": serializable_options,
        "output_dir": str(output_dir),
    }
    config_sha256 = hashlib.sha256(
        json.dumps(normalized_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        **normalized_config,
        "input": str(input_path),
        "reference": str(reference_path),
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise click.UsageError(
                f"Existing bootstrap state does not match this config: {manifest_path}"
            )

    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_path.is_file() and input_path.read_text(encoding="utf-8") != rendered:
        raise click.UsageError(f"Existing bootstrap input differs: {input_path}")
    input_path.write_text(rendered, encoding="utf-8")
    _atomic_write_json(manifest_path, manifest)
    plan = {
        "schema_version": 1,
        "state": "prepared",
        "target": normalized_config["target"],
        "anchor": normalized_config["anchor"],
        "prediction": {
            "model": "Boltz-2",
            "recycling_steps": options.recycling_steps,
            "sampling_steps": options.sampling_steps,
            "diffusion_samples": options.diffusion_samples,
            "seed": options.seed,
            "accelerator": options.accelerator,
            "devices": options.devices,
        },
        "input": str(input_path),
        "reference": str(reference_path),
        "manifest": str(manifest_path),
    }
    _atomic_write_json(plan_path, plan)
    if not status_path.is_file():
        _atomic_write_json(
            status_path,
            {
                "schema_version": 1,
                "state": "prepared",
                "message": "GPU structure prediction has not been started.",
                "reference": str(reference_path),
            },
        )
    return PreparedBootstrapReference(
        config_path=config_path,
        output_dir=output_dir,
        input_path=input_path,
        runtime_dir=runtime_dir,
        reference_path=reference_path,
        manifest_path=manifest_path,
        plan_path=plan_path,
        status_path=status_path,
        target_sequence=sequence,
        target_id=target_id,
        anchor_id=anchor_id,
        auto_msa=auto_msa,
        options=options,
    )


def format_bootstrap_plan(plan: dict[str, object]) -> list[str]:
    """Return a compact human-readable prepared-plan summary."""
    target = _mapping(plan["target"], "plan.target")
    anchor = _mapping(plan["anchor"], "plan.anchor")
    prediction = _mapping(plan["prediction"], "plan.prediction")
    return [
        f"Target: {target['length']} residues; MSA={target['msa']['mode']}",
        f"Anchor: {anchor['id']} ({anchor['smiles']})",
        (
            "Prediction: "
            f"samples={prediction['diffusion_samples']}, "
            f"recycles={prediction['recycling_steps']}, "
            f"sampling_steps={prediction['sampling_steps']}, "
            f"seed={prediction['seed']}"
        ),
        f"Reference output: {plan['reference']}",
    ]


def _to_pdb(structure: Any) -> str:  # noqa: C901
    """Serialize the protein-ligand StructureV2 needed by reference cropping."""
    periodic_table = Chem.GetPeriodicTable()
    lines: list[str] = []
    atom_serial = 1
    atom_serials: dict[int, int] = {}
    from boltz.data import const

    nonpolymer = const.chain_type_ids["NONPOLYMER"]
    for chain_index, chain in enumerate(structure.chains):
        chain_name = str(chain["name"])
        if len(chain_name) != 1:
            raise ValueError("Bootstrap PDB output requires one-character chain IDs.")
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        last_residue = None
        for residue in structure.residues[res_start:res_end]:
            last_residue = residue
            res_name = str(residue["name"])[:3]
            residue_number = int(residue["res_idx"]) + 1
            atom_start = int(residue["atom_idx"])
            atom_end = atom_start + int(residue["atom_num"])
            for atom_index in range(atom_start, atom_end):
                atom = structure.atoms[atom_index]
                if not bool(atom["is_present"]):
                    continue
                record_type = (
                    "HETATM"
                    if int(chain["mol_type"]) == nonpolymer
                    else "ATOM"
                )
                atom_name = str(atom["name"])
                atom_name = atom_name if len(atom_name) == 4 else f" {atom_name}"
                element = periodic_table.GetElementSymbol(int(atom["element"])).upper()
                x, y, z = atom["coords"]
                line = (
                    f"{record_type:<6}{atom_serial:>5} {atom_name:<4} "
                    f"{res_name:>3} {chain_name:1}{residue_number:>4}    "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                    f"{1.0:>6.2f}{100.0:>6.2f}          {element:>2}  "
                )
                lines.append(line)
                atom_serials[atom_index] = atom_serial
                atom_serial += 1
        if chain_index < len(structure.chains) - 1 and last_residue is not None:
            lines.append(
                f"TER   {atom_serial:>5}      {str(last_residue['name'])[:3]:>3} "
                f"{chain_name:1}{int(last_residue['res_idx']) + 1:>4}"
            )
            atom_serial += 1
    for bond in structure.bonds:
        first = atom_serials.get(int(bond["atom_1"]))
        second = atom_serials.get(int(bond["atom_2"]))
        if first is not None and second is not None:
            lines.append(f"CONECT{first:>5}{second:>5}")
    lines.extend(["END", ""])
    return "\n".join(lines)


def run_bootstrap_reference(prepared: PreparedBootstrapReference) -> Path:
    """Run the explicitly requested GPU holo prediction and publish a reference."""
    import torch
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import BasePredictionWriter

    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest, StructureV2
    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
        _validate_checkpoint_for_load,
        check_inputs,
        download_boltz2_reference_assets,
        process_inputs,
    )
    from boltz.model.models.boltz2 import Boltz2

    options = prepared.options
    result_path = prepared.output_dir / "bootstrap_result.json"
    if prepared.reference_path.is_file() and result_path.is_file():
        existing_result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing_result.get("state") == "complete"
            and existing_result.get("reference_sha256")
            == _sha256(prepared.reference_path)
        ):
            return prepared.reference_path
        raise click.ClickException(
            "Existing bootstrap reference does not match bootstrap_result.json."
        )
    if prepared.reference_path.exists() or result_path.exists():
        raise click.ClickException(
            "Bootstrap output is incomplete; preserve it for diagnosis, then remove "
            "the incomplete reference/result before retrying."
        )
    _atomic_write_json(
        prepared.status_path,
        {
            "schema_version": 1,
            "state": "running",
            "message": "Running one Boltz-2 holo reference prediction.",
            "reference": str(prepared.reference_path),
        },
    )
    try:
        download_boltz2_reference_assets(options.cache, options.checkpoint)
        inputs = check_inputs(prepared.input_path, skip_bad_inputs=False)
        process_inputs(
            data=inputs,
            out_dir=prepared.runtime_dir,
            ccd_path=options.cache / "ccd.pkl",
            mol_dir=options.cache / "mols",
            use_msa_server=prepared.auto_msa,
            msa_server_url=options.msa_server_url,
            msa_pairing_strategy=options.msa_pairing_strategy,
            boltz2=True,
            preprocessing_threads=options.preprocessing_threads,
            max_msa_seqs=options.max_msa_seqs,
        )
        manifest = Manifest.load(prepared.runtime_dir / "processed/manifest.json")
        if len(manifest.records) != 1:
            raise click.ClickException(
                "Bootstrap preprocessing did not produce exactly one record."
            )

        class _ReferenceWriter(BasePredictionWriter):
            def __init__(self) -> None:
                super().__init__(write_interval="batch")
                self.selected_confidence: float | None = None
                self.pdb_text: str | None = None

            def write_on_batch_end(
                self,
                trainer: Trainer,  # noqa: ARG002
                pl_module: Any,  # noqa: ARG002
                prediction: dict[str, Any],
                batch_indices: list[int],  # noqa: ARG002
                batch: dict[str, Any],
                batch_idx: int,  # noqa: ARG002
                dataloader_idx: int,  # noqa: ARG002
            ) -> None:
                if prediction["exception"]:
                    raise RuntimeError("Boltz-2 reference prediction failed.")
                if len(batch["record"]) != 1:
                    raise RuntimeError("Reference writer expected one record.")
                record = batch["record"][0]
                structure = StructureV2.load(
                    prepared.runtime_dir / "processed/structures" / f"{record.id}.npz"
                )
                structure = structure.remove_invalid_chains()
                atom_mask = prediction["masks"][0].bool()
                coords = prediction["coords"][:, atom_mask]
                if "confidence_score" in prediction:
                    scores = prediction["confidence_score"].reshape(-1)
                    selected = int(torch.argmax(scores).item())
                    self.selected_confidence = float(scores[selected].item())
                else:
                    selected = 0
                selected_coords = coords[selected].detach().cpu().numpy()
                if selected_coords.shape[0] != len(structure.atoms):
                    raise RuntimeError(
                        "Predicted coordinate count does not match processed atoms."
                    )
                atoms = structure.atoms.copy()
                atoms["coords"] = selected_coords
                atoms["is_present"] = True
                residues = structure.residues.copy()
                residues["is_present"] = True
                predicted = replace(structure, atoms=atoms, residues=residues)
                self.pdb_text = _to_pdb(predicted)

        if not getattr(torch.load, "_boltz_patched", False):
            original_torch_load = torch.load

            def patched_load(*args: Any, **kwargs: Any) -> Any:
                return original_torch_load(
                    *args,
                    **{**kwargs, "weights_only": False},
                )

            patched_load._boltz_patched = True  # type: ignore[attr-defined]
            torch.load = patched_load  # type: ignore[assignment]
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision("highest")
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        if options.seed is not None:
            seed_everything(options.seed)
        for key in ("CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"):
            os.environ[key] = os.environ.get(key, "1")

        diffusion = Boltz2DiffusionParams(step_scale=options.step_scale)
        steering = BoltzSteeringParams(
            fk_steering=False,
            physical_guidance_update=False,
            contact_guidance_update=True,
        )
        predict_args = {
            "recycling_steps": options.recycling_steps,
            "sampling_steps": options.sampling_steps,
            "diffusion_samples": options.diffusion_samples,
            "max_parallel_samples": options.max_parallel_samples,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        }
        _validate_checkpoint_for_load(options.checkpoint, "Boltz-2 weights")
        map_location = {
            "cpu": "cpu",
            "mps": "mps",
            "gpu": "cuda",
        }[options.accelerator]
        model = Boltz2.load_from_checkpoint(
            options.checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location=map_location,
            diffusion_process_args=asdict(diffusion),
            ema=False,
            use_kernels=not options.no_kernels,
            use_flash_attn=options.flash_attn,
            pairformer_args=asdict(PairformerArgsV2()),
            msa_args=asdict(MSAModuleArgs()),
            steering_args=asdict(steering),
        )
        model.eval()
        processed = prepared.runtime_dir / "processed"
        data_module = Boltz2InferenceDataModule(
            manifest=manifest,
            target_dir=processed / "structures",
            msa_dir=processed / "msa",
            mol_dir=options.cache / "mols",
            num_workers=options.num_workers,
            constraints_dir=processed / "constraints",
            template_dir=processed / "templates",
            extra_mols_dir=processed / "mols",
            affinity_ligand_ref_pos_mode="etkdg",
            pin_memory=options.accelerator not in {"cpu", "mps"},
        )
        writer = _ReferenceWriter()
        trainer = Trainer(
            default_root_dir=prepared.runtime_dir,
            callbacks=[writer],
            accelerator=options.accelerator,
            devices=options.devices,
            precision=(32 if options.accelerator in {"cpu", "mps"} else "bf16-mixed"),
            logger=False,
        )
        trainer.predict(model, datamodule=data_module, return_predictions=False)
        if writer.pdb_text is None:
            raise RuntimeError("Reference prediction produced no PDB file.")
        temporary_reference = prepared.reference_path.with_name(
            f".{prepared.reference_path.name}.tmp"
        )
        temporary_reference.write_text(writer.pdb_text, encoding="utf-8")
        temporary_reference.replace(prepared.reference_path)
        reference_sha256 = _sha256(prepared.reference_path)
        result = {
            "schema_version": 1,
            "state": "complete",
            "reference": str(prepared.reference_path),
            "reference_sha256": reference_sha256,
            "checkpoint": str(options.checkpoint),
            "checkpoint_sha256": _sha256(options.checkpoint),
            "selected_confidence": writer.selected_confidence,
            "target_chain": prepared.target_id,
            "anchor_chain": prepared.anchor_id,
            "screen_crop": {
                "reference": str(prepared.reference_path),
                "reference_protein": prepared.target_id,
                "reference_ligand": prepared.anchor_id,
                "budget": "library_fixed",
            },
        }
        _atomic_write_json(result_path, result)
        _atomic_write_json(prepared.status_path, result)
        return prepared.reference_path
    except BaseException as error:
        _atomic_write_json(
            prepared.status_path,
            {
                "schema_version": 1,
                "state": "failed",
                "message": f"{type(error).__name__}: {error}",
                "reference": str(prepared.reference_path),
            },
        )
        raise

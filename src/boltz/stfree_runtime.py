"""Single-route Boltz-StFree affinity runtime."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import click

_DISTOGRAM_MODE = "zero_distogram_features"
_LIGAND_CONFORMER_MODES = {"skip", "etkdg"}


@dataclass(frozen=True)
class StFreeRuntimeOptions:
    """Supported runtime settings for full-screen affinity scoring."""

    cache: Path
    affinity_checkpoint: Path | None = None
    devices: int | list[int] = 1
    accelerator: str = "gpu"
    num_workers: int = 2
    override: bool = False
    seed: int | None = None
    preprocessing_threads: int = 1
    max_msa_seqs: int = 8192
    subsample_msa: bool = True
    num_subsampled_msa: int = 1024
    no_kernels: bool = False
    flash_attn: bool = False
    affinity_mw_correction: bool = False
    affinity_recycling_steps: int = 0
    ligand_reference_conformer: str = "skip"
    etkdg_seed: int = 42


def _patch_torch_checkpoint_loading(torch: object) -> None:
    """Allow trusted Lightning checkpoints under PyTorch 2.6 and newer."""
    torch_load = torch.load
    if getattr(torch_load, "_boltz_patched", False):
        return

    def patched_load(*args: object, **kwargs: object) -> object:
        return torch_load(*args, **{**kwargs, "weights_only": False})

    patched_load._boltz_patched = True  # type: ignore[attr-defined]
    torch.load = patched_load  # type: ignore[attr-defined]


def _validate_one_to_one_manifest(manifest: object) -> None:
    """Require exactly one protein and one affinity ligand in every record."""
    from boltz.data import const

    protein_type = const.chain_type_ids["PROTEIN"]
    ligand_type = const.chain_type_ids["NONPOLYMER"]
    failures: list[str] = []
    for record in manifest.records:
        proteins = [chain for chain in record.chains if chain.mol_type == protein_type]
        ligands = [chain for chain in record.chains if chain.mol_type == ligand_type]
        affinity = record.affinity
        binder_matches = []
        if affinity is not None:
            binder_matches = [
                chain
                for chain in ligands
                if (
                    chain.chain_id == affinity.chain_id
                    or (
                        affinity.chain_name is not None
                        and chain.chain_name == affinity.chain_name
                    )
                )
            ]
        if (
            len(record.chains) != 2
            or len(proteins) != 1
            or len(ligands) != 1
            or len(binder_matches) != 1
        ):
            failures.append(record.id)
        if record.inference_options is not None:
            pocket = record.inference_options.pocket_constraints or []
            contacts = record.inference_options.contact_constraints or []
            if pocket or contacts:
                failures.append(record.id)
        if record.templates:
            failures.append(record.id)
    if failures:
        names = ", ".join(dict.fromkeys(failures))
        raise click.ClickException(
            "Boltz-StFree accepts exactly one unconstrained protein and one "
            f"affinity ligand per record; invalid records: {names}"
        )


def _trainer_strategy(
    accelerator: str,
    devices: int | list[int],
    record_count: int,
) -> tuple[object, int | list[int]]:
    """Return a Lightning strategy and a valid device selection."""
    from pytorch_lightning.strategies import DDPStrategy

    if accelerator == "mps":
        if (isinstance(devices, int) and devices > 1) or (
            isinstance(devices, list) and len(devices) > 1
        ):
            click.echo("Warning: MPS only supports one device; using one.")
        return "auto", 1

    device_count = len(devices) if isinstance(devices, list) else devices
    if device_count <= 1:
        return "auto", devices
    if record_count < device_count:
        click.echo(
            "Warning: requested devices exceed pending records; reducing devices."
        )
        if isinstance(devices, list):
            devices = devices[: max(1, record_count)]
        else:
            devices = max(1, record_count)
    start_method = "spawn" if platform.system() == "Windows" else "fork"
    return DDPStrategy(start_method=start_method), devices


@lru_cache(maxsize=8)
def _cached_file_sha256(path_value: str, size: int, mtime_ns: int) -> str:
    """Hash a stable file identity once per process."""
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_value).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    """Return a cached digest that invalidates when file metadata changes."""
    resolved = path.resolve()
    stat = resolved.stat()
    return _cached_file_sha256(str(resolved), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=1)
def _runtime_code_sha256() -> str:
    """Fingerprint the installed Boltz Python implementation."""
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _run_provenance(
    options: StFreeRuntimeOptions,
    *,
    runtime_version: str,
    checkpoint_sha256: str,
    receptor_selection: object = "complete_input",
) -> dict[str, object]:
    """Build the immutable method identity used to protect resumed screens."""
    return {
        "schema_version": 1,
        "runtime": "boltz-stfree",
        "runtime_version": runtime_version,
        "runtime_code_sha256": _runtime_code_sha256(),
        "affinity_checkpoint_sha256": checkpoint_sha256,
        "method": {
            "input_contract": "one_protein_one_ligand",
            "receptor_selection": receptor_selection,
            "distogram_mode": _DISTOGRAM_MODE,
            "ligand_reference_conformer": {
                "mode": options.ligand_reference_conformer,
                "reference_positions": (
                    "zero"
                    if options.ligand_reference_conformer == "skip"
                    else "etkdg_v3_uff"
                ),
                "etkdg_seed": (
                    options.etkdg_seed
                    if options.ligand_reference_conformer == "etkdg"
                    else None
                ),
            },
            "structure_generation_active": False,
            "affinity_recycling_steps": options.affinity_recycling_steps,
            "affinity_mw_correction": options.affinity_mw_correction,
            "seed": options.seed,
            "max_msa_seqs": options.max_msa_seqs,
            "subsample_msa": options.subsample_msa,
            "num_subsampled_msa": options.num_subsampled_msa,
            "use_kernels": not options.no_kernels,
            "flash_attn": options.flash_attn,
        },
    }


def _receptor_crop_provenance(
    manifest: object,
    crop_dir: Path | None,
) -> object:
    """Fingerprint the exact per-record receptor selections used by a chunk."""
    if crop_dir is None:
        return "complete_input"
    digest = hashlib.sha256()
    for record in sorted(manifest.records, key=lambda item: item.id):
        path = crop_dir / f"affinity_crop_{record.id}.json"
        if not path.is_file():
            raise click.ClickException(
                f"Missing reference receptor crop for {record.id}: {path}"
            )
        digest.update(record.id.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "mode": "reference_contiguous_blocks",
        "crop_sha256": digest.hexdigest(),
        "record_count": len(manifest.records),
    }


def _validate_resume_state(
    *,
    manifest: object,
    run_dir: Path,
    provenance: dict[str, object],
    override: bool,
) -> None:
    """Reject malformed or differently-derived scores before resume filtering."""
    from boltz.data.types import get_affinity_output_id

    score_paths = []
    for record in manifest.records:
        output_id = get_affinity_output_id(record)
        score_paths.append(
            run_dir
            / "predictions"
            / record.id
            / f"affinity_{output_id}__distogram_{_DISTOGRAM_MODE}.json"
        )
    existing = [path for path in score_paths if path.exists()]
    if override:
        for path in existing:
            if not path.is_file() and not path.is_symlink():
                raise click.ClickException(
                    f"Cannot replace non-file structure-free output: {path}"
                )
            try:
                path.unlink()
            except OSError as error:
                raise click.ClickException(
                    f"Could not remove structure-free output for override: {path}"
                ) from error
        return
    if not existing:
        return

    metadata_path = run_dir / "predictions" / "affinity_run_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise click.ClickException(
            "Existing structure-free scores have missing or invalid run metadata; "
            "set options.override: true to recompute them."
        ) from error
    metadata_experiment = (
        metadata.get("experiment", {}) if isinstance(metadata, dict) else {}
    )
    if (
        not isinstance(metadata_experiment, dict)
        or metadata_experiment.get("provenance") != provenance
    ):
        raise click.ClickException(
            "Existing structure-free scores have different checkpoint or runtime "
            "provenance; set options.override: true to recompute them."
        )

    for path in existing:
        try:
            score = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise click.ClickException(
                f"Existing structure-free score is invalid: {path}; set "
                "options.override: true to recompute it."
            ) from error
        if not isinstance(score, dict):
            raise click.ClickException(
                f"Existing structure-free score does not match the expected schema: "
                f"{path}; set options.override: true to recompute it."
            )
        scalar_keys = ("affinity_pred_value", "affinity_probability_binary")
        valid_scalars = all(
            isinstance(score.get(key), (int, float))
            and not isinstance(score.get(key), bool)
            and math.isfinite(float(score[key]))
            for key in scalar_keys
        )
        experiment = score.get("experiment", {})
        if (
            not valid_scalars
            or score.get("distogram_intervention") != _DISTOGRAM_MODE
            or not isinstance(experiment, dict)
            or experiment.get("run_metadata_file") != "../affinity_run_metadata.json"
        ):
            raise click.ClickException(
                f"Existing structure-free score does not match the expected schema: "
                f"{path}; set options.override: true to recompute it."
            )


def stfree_model_kwargs(options: StFreeRuntimeOptions) -> dict[str, object]:
    """Return the fixed model route used by every Boltz-StFree record."""
    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
    )

    steering = BoltzSteeringParams()
    steering.fk_steering = False
    steering.physical_guidance_update = False
    steering.contact_guidance_update = False
    return {
        "diffusion_process_args": asdict(Boltz2DiffusionParams()),
        "pairformer_args": asdict(PairformerArgsV2()),
        "msa_args": asdict(
            MSAModuleArgs(
                subsample_msa=options.subsample_msa,
                num_subsampled_msa=options.num_subsampled_msa,
                use_paired_feature=True,
            )
        ),
        "steering_args": asdict(steering),
        "affinity_mw_correction": options.affinity_mw_correction,
        "affinity_distogram_modes": [_DISTOGRAM_MODE],
        "affinity_coordinate_free": True,
        "use_kernels": not options.no_kernels,
        "use_flash_attn": options.flash_attn,
        "skip_confidence_forward": True,
        "skip_pdistogram_forward": True,
        "skip_msa_forward": False,
        "capture_affinity_representations": False,
        "run_trunk_and_structure": True,
        "skip_run_structure": True,
    }


def run_stfree_chunk(
    *,
    data: Path,
    out_dir: Path,
    options: StFreeRuntimeOptions,
    receptor_crop_indices_dir: Path | None = None,
) -> None:
    """Score one chunk without generating or loading complex coordinates."""
    import torch
    from pytorch_lightning import Trainer, seed_everything
    from rdkit import Chem

    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    from boltz.data.write.writer import BoltzAffinityWriter
    from boltz.main import (
        BoltzProcessedInput,
        _configure_wsl_multiprocessing_tempdir,
        _resolved_input_sources,
        _runtime_version,
        _validate_checkpoint_for_load,
        check_inputs,
        download_stfree,
        filter_inputs_affinity,
        process_inputs,
    )
    from boltz.model.models.boltz2 import Boltz2

    _configure_wsl_multiprocessing_tempdir(options.num_workers)
    _patch_torch_checkpoint_loading(torch)
    if options.ligand_reference_conformer not in _LIGAND_CONFORMER_MODES:
        choices = ", ".join(sorted(_LIGAND_CONFORMER_MODES))
        raise click.UsageError(
            f"ligand_reference_conformer must be one of: {choices}."
        )
    if options.accelerator == "cpu":
        click.echo("Running on CPU; affinity scoring will be slow.")
    if options.accelerator == "mps" and not torch.backends.mps.is_available():
        raise click.UsageError("MPS was requested but is not available.")

    warnings.filterwarnings(
        "ignore",
        ".*that has Tensor Cores. To properly utilize them.*",
    )
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    if options.seed is not None:
        seed_everything(options.seed)
    for key in ("CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"):
        os.environ[key] = os.environ.get(key, "1")

    cache = options.cache.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    download_stfree(
        cache,
        include_default_affinity=options.affinity_checkpoint is None,
    )
    affinity_checkpoint = (
        options.affinity_checkpoint or cache / "boltz2_aff.ckpt"
    ).expanduser()
    _validate_checkpoint_for_load(affinity_checkpoint, "Boltz-2 affinity weights")
    checkpoint_sha256 = _file_sha256(affinity_checkpoint)

    input_paths = check_inputs(data.expanduser().resolve(), skip_bad_inputs=False)
    run_dir = out_dir.expanduser().resolve() / f"boltz_results_{data.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    mol_dir = cache / "mols"
    process_inputs(
        data=input_paths,
        out_dir=run_dir,
        ccd_path=cache / "ccd.pkl",
        mol_dir=mol_dir,
        use_msa_server=False,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        msa_server_username=None,
        msa_server_password=None,
        api_key_header=None,
        api_key_value=None,
        boltz2=True,
        preprocessing_threads=options.preprocessing_threads,
        max_msa_seqs=options.max_msa_seqs,
        skip_bad_inputs=False,
        # StFree never needs a ligand conformer during parsing. The optional
        # deterministic ETKDG conformer is generated immediately before
        # featurization instead.
        skip_ligand_conformer_generation=True,
    )

    processed_dir = run_dir / "processed"
    manifest = Manifest.load(processed_dir / "manifest.json")
    if not manifest.records:
        raise click.ClickException("No valid protein-ligand records were processed.")
    _validate_one_to_one_manifest(manifest)
    runtime_version = _runtime_version()
    receptor_selection = _receptor_crop_provenance(
        manifest,
        receptor_crop_indices_dir,
    )
    provenance = _run_provenance(
        options,
        runtime_version=runtime_version,
        checkpoint_sha256=checkpoint_sha256,
        receptor_selection=receptor_selection,
    )
    experiment_metadata = {
        "runtime": "boltz-stfree",
        "runtime_version": runtime_version,
        "provenance": provenance,
        "input_sources": _resolved_input_sources(input_paths),
        "affinity_checkpoint": str(affinity_checkpoint.resolve()),
        "affinity_checkpoint_sha256": checkpoint_sha256,
        "seed": options.seed,
        "input_contract": "one_protein_one_ligand",
        "receptor_selection": receptor_selection,
        "structure_generation_active": False,
        "generated_crop_active": receptor_crop_indices_dir is not None,
        "coordinate_derived_distogram_active": False,
        "distogram_modes": [_DISTOGRAM_MODE],
        "ligand_reference_conformer": provenance["method"][
            "ligand_reference_conformer"
        ],
        "affinity_recycling_steps": options.affinity_recycling_steps,
        "affinity_sampling_steps": 1,
        "affinity_diffusion_samples": 1,
        "affinity_batch_size": 1,
        "use_kernels": not options.no_kernels,
    }
    _validate_resume_state(
        manifest=manifest,
        run_dir=run_dir,
        provenance=provenance,
        override=options.override,
    )
    pending = filter_inputs_affinity(
        manifest=manifest,
        outdir=run_dir,
        override=options.override,
        distogram_modes=(_DISTOGRAM_MODE,),
        output_tag=None,
        require_pre_affinity=False,
    )
    if not pending.records:
        click.echo("Found existing structure-free scores for all inputs; skipping.")
        return

    processed = BoltzProcessedInput(
        manifest=pending,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            processed_dir / "constraints"
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=None,
        extra_mols_dir=(
            processed_dir / "mols" if (processed_dir / "mols").exists() else None
        ),
    )

    strategy, devices = _trainer_strategy(
        options.accelerator,
        options.devices,
        len(pending.records),
    )
    precision: int | str = 32 if options.accelerator in ("cpu", "mps") else "bf16-mixed"
    map_location = (
        "cpu"
        if options.accelerator == "cpu"
        else "mps"
        if options.accelerator == "mps"
        else "cuda"
    )
    writer = BoltzAffinityWriter(
        data_dir=processed.targets_dir,
        output_dir=run_dir / "predictions",
        write_embeddings=False,
        write_affinity_representations=False,
        output_tag=None,
        screening_minimal_output=True,
        experiment_metadata=experiment_metadata,
    )
    trainer = Trainer(
        default_root_dir=run_dir,
        strategy=strategy,
        callbacks=[writer],
        accelerator=options.accelerator,
        devices=devices,
        precision=precision,
        logger=False,
    )
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mol_dir,
        num_workers=options.num_workers,
        batch_size=1,
        constraints_dir=processed.constraints_dir,
        template_dir=None,
        extra_mols_dir=processed.extra_mols_dir,
        override_method="other",
        affinity=True,
        affinity_crop_indices_dir=None,
        affinity_receptor_crop_indices_dir=receptor_crop_indices_dir,
        affinity_structure_free=True,
        affinity_structure_free_single_sequence=False,
        affinity_sequence_window_start=0,
        affinity_ligand_ref_pos_mode=options.ligand_reference_conformer,
        affinity_ligand_ref_pos_seed=(
            options.etkdg_seed
            if options.ligand_reference_conformer == "etkdg"
            else None
        ),
        affinity_require_full_input=receptor_crop_indices_dir is None,
        pin_memory=options.accelerator not in ("cpu", "mps"),
    )
    predict_args = {
        "recycling_steps": options.affinity_recycling_steps,
        "sampling_steps": 1,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    model = Boltz2.load_from_checkpoint(
        affinity_checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location=map_location,
        ema=False,
        **stfree_model_kwargs(options),
    )
    model.eval()
    click.echo(f"Running structure-free affinity for {len(pending.records)} record(s).")
    trainer.predict(model, datamodule=data_module, return_predictions=False)

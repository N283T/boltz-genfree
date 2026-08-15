"""Single-route Boltz-StFree affinity runtime."""

from __future__ import annotations

import os
import platform
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import click


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
        "affinity_distogram_modes": ["zero_distogram_features"],
        "affinity_coordinate_free": True,
        "use_kernels": not options.no_kernels,
        "use_flash_attn": options.flash_attn,
        "skip_confidence_forward": True,
        "skip_pdistogram_forward": False,
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
        _sha256_file,
        _validate_checkpoint_for_load,
        check_inputs,
        download_stfree,
        filter_inputs_affinity,
        process_inputs,
    )
    from boltz.model.models.boltz2 import Boltz2

    _configure_wsl_multiprocessing_tempdir(options.num_workers)
    _patch_torch_checkpoint_loading(torch)
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
    download_stfree(cache)

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
        skip_ligand_conformer_generation=False,
    )

    processed_dir = run_dir / "processed"
    manifest = Manifest.load(processed_dir / "manifest.json")
    if not manifest.records:
        raise click.ClickException("No valid protein-ligand records were processed.")
    _validate_one_to_one_manifest(manifest)
    pending = filter_inputs_affinity(
        manifest=manifest,
        outdir=run_dir,
        override=options.override,
        distogram_modes=("zero_distogram_features",),
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
    affinity_checkpoint = (
        options.affinity_checkpoint or cache / "boltz2_aff.ckpt"
    ).expanduser()
    _validate_checkpoint_for_load(affinity_checkpoint, "Boltz-2 affinity weights")

    writer = BoltzAffinityWriter(
        data_dir=processed.targets_dir,
        output_dir=run_dir / "predictions",
        write_embeddings=False,
        write_affinity_representations=False,
        output_tag=None,
        screening_minimal_output=True,
        experiment_metadata={
            "runtime": "boltz-stfree",
            "runtime_version": _runtime_version(),
            "input_sources": _resolved_input_sources(input_paths),
            "affinity_checkpoint": str(affinity_checkpoint.resolve()),
            "affinity_checkpoint_sha256": _sha256_file(affinity_checkpoint),
            "seed": options.seed,
            "input_contract": "one_protein_one_ligand",
            "receptor_selection": "complete_input",
            "structure_generation_active": False,
            "generated_crop_active": False,
            "coordinate_derived_distogram_active": False,
            "distogram_modes": ["zero_distogram_features"],
            "ligand_reference_conformer": "etkdg",
            "affinity_recycling_steps": options.affinity_recycling_steps,
            "affinity_sampling_steps": 1,
            "affinity_diffusion_samples": 1,
            "affinity_batch_size": 1,
            "use_kernels": not options.no_kernels,
        },
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
        affinity_structure_free=True,
        affinity_structure_free_single_sequence=False,
        affinity_sequence_window_start=0,
        affinity_ligand_ref_pos_mode="etkdg",
        affinity_require_full_input=True,
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

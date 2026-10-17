"""Single-route Boltz-GenFree affinity runtime."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import click

from boltz.seed import record_seed

if TYPE_CHECKING:
    from pytorch_lightning import Callback

_DISTOGRAM_MODE = "zero_distogram_features"
_LIGAND_CONFORMER_MODES = {"skip", "etkdg", "rdkit_2d"}
_RECORD_SEED_SCHEME = "sha256_record_v1"
_UNUSED_STFREE_MODEL_MODULES = (
    "structure_module",
    "diffusion_conditioning",
    "confidence_module",
    "distogram_module",
    "bfactor_module",
)


@dataclass(frozen=True)
class StFreeRuntimeOptions:
    """Supported runtime settings for full-screen affinity scoring."""

    cache: Path
    affinity_checkpoint: Path | None = None
    affinity_adapter: Path | None = None
    target_sequence_sha256: str | None = None
    receptor_reference_sha256: str | None = None
    receptor_crop_config_sha256: str | None = None
    devices: int | list[int] = 1
    accelerator: str = "gpu"
    num_workers: int = 2
    affinity_batch_size: int = 1
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
    allow_extended_full_input: bool = False
    affinity_conditioner_mode: str = "original"
    etkdg_seed: int = 42


@dataclass
class StFreeRuntimeSession:
    """Model and immutable resources shared by every chunk in one screen."""

    affinity_checkpoint: Path
    checkpoint_sha256: str
    runtime_version: str
    mol_dir: Path
    model: object
    adapter_identity: dict[str, object] | None = None


def _record_msa_seed(base_seed: int, record_id: str) -> int:
    """Derive a stable CPU RNG seed for one screened record."""
    return record_seed(base_seed, record_id, purpose="msa_subsample")


def _record_seeded_msa_callback(base_seed: int) -> Callback:
    """Attach stable per-record MSA seeds to each prediction batch."""
    import torch
    from pytorch_lightning import Callback

    class RecordSeededMsaCallback(Callback):
        """Make MSA sampling independent of batching and chunk order."""

        def on_predict_batch_start(
            self,
            trainer: object,  # noqa: ARG002
            pl_module: object,  # noqa: ARG002
            batch: dict[str, object],
            batch_idx: int,  # noqa: ARG002
            dataloader_idx: int = 0,  # noqa: ARG002
        ) -> None:
            records = batch.get("record")
            if not isinstance(records, list) or not records:
                raise RuntimeError("Cannot seed MSA sampling without records.")
            seeds = []
            for record in records:
                record_id = getattr(record, "id", None)
                if not isinstance(record_id, str) or not record_id:
                    raise RuntimeError("Cannot seed MSA sampling without a record ID.")
                seeds.append(_record_msa_seed(base_seed, record_id))
            batch["msa_subsample_seeds"] = torch.tensor(seeds, dtype=torch.int64)

    return RecordSeededMsaCallback()


def _patch_torch_checkpoint_loading(torch: object) -> None:
    """Allow trusted Lightning checkpoints under PyTorch 2.6 and newer."""
    torch_load = torch.load
    if getattr(torch_load, "_boltz_patched", False):
        return

    def patched_load(*args: object, **kwargs: object) -> object:
        # Lightning may pass its unspecified default explicitly as None.
        # Preserve explicit True for the restricted adapter loader.
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return torch_load(*args, **kwargs)

    patched_load._boltz_patched = True  # type: ignore[attr-defined]
    torch.load = patched_load  # type: ignore[attr-defined]


def _prune_stfree_model(model: object) -> tuple[str, ...]:
    """Drop checkpoint modules that the fixed StFree forward path cannot call."""
    required_flags = {
        "affinity_coordinate_free": True,
        "skip_run_structure": True,
        "skip_confidence_forward": True,
        "skip_pdistogram_forward": True,
    }
    invalid = [
        name
        for name, expected in required_flags.items()
        if getattr(model, name, None) != expected
    ]
    if invalid:
        names = ", ".join(invalid)
        raise RuntimeError(f"Cannot prune a non-GenFree model; invalid flags: {names}")

    removed = []
    for name in _UNUSED_STFREE_MODEL_MODULES:
        if getattr(model, name, None) is not None:
            setattr(model, name, None)
            removed.append(name)
    return tuple(removed)


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
            "Boltz-GenFree accepts exactly one unconstrained protein and one "
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
    adapter_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the immutable method identity used to protect resumed screens."""
    if options.affinity_adapter is not None and adapter_identity is None:
        from boltz.affinity_adapter import load_adapter, validate_contract

        adapter = load_adapter(options.affinity_adapter)
        validate_contract(adapter, options, base_checkpoint_sha256=checkpoint_sha256)
        adapter_identity = adapter.identity()
    provenance = {
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
                    else (
                        "rdkit_2d"
                        if options.ligand_reference_conformer == "rdkit_2d"
                        else "etkdg_v3_uff"
                    )
                ),
                "etkdg_seed": (
                    options.etkdg_seed
                    if options.ligand_reference_conformer == "etkdg"
                    else None
                ),
            },
            "structure_generation_active": False,
            "allow_extended_full_input": options.allow_extended_full_input,
            "affinity_conditioner_mode": options.affinity_conditioner_mode,
            "affinity_recycling_steps": options.affinity_recycling_steps,
            "affinity_batch_size": options.affinity_batch_size,
            "affinity_mw_correction": options.affinity_mw_correction,
            "seed": options.seed,
            "record_seed_scheme": (
                _RECORD_SEED_SCHEME if options.seed is not None else None
            ),
            "msa_subsample_seed_scheme": (
                _RECORD_SEED_SCHEME
                if options.subsample_msa and options.seed is not None
                else None
            ),
            "max_msa_seqs": options.max_msa_seqs,
            "subsample_msa": options.subsample_msa,
            "num_subsampled_msa": options.num_subsampled_msa,
            "use_kernels": not options.no_kernels,
            "flash_attn": options.flash_attn,
        },
    }

    if adapter_identity is not None:
        provenance["affinity_adapter"] = adapter_identity
    return provenance


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
    """Return the fixed model route used by every Boltz-GenFree record."""
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


def prepare_stfree_inputs(
    *,
    input_paths: list[Path],
    run_dir: Path,
    options: StFreeRuntimeOptions,
) -> None:
    """Preprocess every supplied input in one shared runtime directory."""
    from rdkit import Chem

    from boltz.main import (
        _configure_wsl_multiprocessing_tempdir,
        download_stfree,
        process_inputs,
    )

    _configure_wsl_multiprocessing_tempdir(options.num_workers)
    if options.ligand_reference_conformer not in _LIGAND_CONFORMER_MODES:
        choices = ", ".join(sorted(_LIGAND_CONFORMER_MODES))
        raise click.UsageError(f"ligand_reference_conformer must be one of: {choices}.")
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    cache = options.cache.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    download_stfree(
        cache,
        include_default_affinity=options.affinity_checkpoint is None,
    )
    input_paths = [path.expanduser().resolve() for path in input_paths]
    run_dir = run_dir.expanduser().resolve()
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


def prepare_stfree_chunk(
    *,
    data: Path,
    out_dir: Path,
    options: StFreeRuntimeOptions,
) -> None:
    """Backward-compatible wrapper that preprocesses one input chunk."""
    from boltz.main import check_inputs

    data = data.expanduser().resolve()
    prepare_stfree_inputs(
        input_paths=check_inputs(data, skip_bad_inputs=False),
        run_dir=out_dir.expanduser().resolve() / f"boltz_results_{data.stem}",
        options=options,
    )


def create_stfree_runtime_session(
    options: StFreeRuntimeOptions,
) -> StFreeRuntimeSession:
    """Load one reusable StFree model after all chunks are preprocessed."""
    import torch
    from pytorch_lightning import seed_everything

    from boltz.main import (
        _configure_wsl_multiprocessing_tempdir,
        _runtime_version,
        _validate_checkpoint_for_load,
        download_stfree,
    )
    from boltz.model.models.boltz2 import Boltz2

    adapter = None
    if options.affinity_adapter is not None:
        from boltz.affinity_adapter import load_adapter, validate_contract

        try:
            adapter = load_adapter(options.affinity_adapter)
            validate_contract(adapter, options)
        except (OSError, ValueError) as error:
            raise click.UsageError(f"Invalid affinity adapter: {error}") from error

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
    if adapter is not None:
        try:
            validate_contract(adapter, options, base_checkpoint_sha256=checkpoint_sha256)
        except ValueError as error:
            raise click.UsageError(f"Invalid affinity adapter: {error}") from error
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
        map_location="cpu",
        ema=False,
        **stfree_model_kwargs(options),
    )
    if adapter is not None:
        from boltz.affinity_adapter import apply_adapter

        try:
            apply_adapter(model, adapter)
        except ValueError as error:
            raise click.UsageError(f"Invalid affinity adapter: {error}") from error
    from boltz.model.modules.affinity import AffinityModule

    if options.affinity_conditioner_mode not in {"original", "folded"}:
        raise click.UsageError("affinity_conditioner_mode must be original or folded.")
    for module in model.modules():
        if isinstance(module, AffinityModule):
            module.fold_zero_conditioner = options.affinity_conditioner_mode == "folded"
    _prune_stfree_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    return StFreeRuntimeSession(
        affinity_checkpoint=affinity_checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        runtime_version=_runtime_version(),
        mol_dir=cache / "mols",
        model=model,
        adapter_identity=adapter.identity() if adapter is not None else None,
    )


def run_stfree_inputs(
    *,
    input_paths: list[Path],
    run_dir: Path,
    options: StFreeRuntimeOptions,
    receptor_crop_indices_dir: Path | None = None,
    progress_callback: Callable[[list[str], bool], None] | None = None,
    session: StFreeRuntimeSession | None = None,
    preprocess: bool = True,
) -> None:
    """Score one shared manifest containing every supplied input."""
    from pytorch_lightning import Trainer

    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    from boltz.data.write.writer import BoltzAffinityWriter
    from boltz.main import (
        BoltzProcessedInput,
        _resolved_input_sources,
        filter_inputs_affinity,
    )

    input_paths = [path.expanduser().resolve() for path in input_paths]
    run_dir = run_dir.expanduser().resolve()
    if preprocess:
        prepare_stfree_inputs(
            input_paths=input_paths,
            run_dir=run_dir,
            options=options,
        )
    if session is None:
        session = create_stfree_runtime_session(options)

    if options.affinity_adapter is not None:
        from boltz.affinity_adapter import load_adapter, validate_contract

        try:
            current_adapter = load_adapter(options.affinity_adapter)
            validate_contract(current_adapter, options, base_checkpoint_sha256=session.checkpoint_sha256)
            if current_adapter.identity() != getattr(session, "adapter_identity", None):
                raise ValueError("Adapter changed after model loading; create a new session")
        except (OSError, ValueError) as error:
            raise click.UsageError(f"Invalid affinity adapter: {error}") from error
    elif getattr(session, "adapter_identity", None) is not None:
        raise click.UsageError("Cannot use an adapted session for a no-FT screen")

    processed_dir = run_dir / "processed"
    manifest = Manifest.load(processed_dir / "manifest.json")
    if not manifest.records:
        raise click.ClickException("No valid protein-ligand records were processed.")
    _validate_one_to_one_manifest(manifest)
    receptor_selection = _receptor_crop_provenance(
        manifest,
        receptor_crop_indices_dir,
    )
    provenance = _run_provenance(
        options,
        runtime_version=session.runtime_version,
        checkpoint_sha256=session.checkpoint_sha256,
        receptor_selection=receptor_selection,
        adapter_identity=getattr(session, "adapter_identity", None),
    )
    experiment_metadata = {
        "runtime": "boltz-stfree",
        "runtime_version": session.runtime_version,
        "provenance": provenance,
        "input_sources": _resolved_input_sources(input_paths),
        "affinity_checkpoint": str(session.affinity_checkpoint.resolve()),
        "affinity_checkpoint_sha256": session.checkpoint_sha256,
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
        "affinity_batch_size": options.affinity_batch_size,
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
    writer = BoltzAffinityWriter(
        data_dir=processed.targets_dir,
        output_dir=run_dir / "predictions",
        write_embeddings=False,
        write_affinity_representations=False,
        output_tag=None,
        screening_minimal_output=True,
        experiment_metadata=experiment_metadata,
        progress_callback=progress_callback,
    )
    callbacks = [writer]
    if options.subsample_msa and options.seed is not None:
        callbacks.insert(0, _record_seeded_msa_callback(options.seed))
    trainer = Trainer(
        default_root_dir=run_dir,
        strategy=strategy,
        callbacks=callbacks,
        accelerator=options.accelerator,
        devices=devices,
        precision=precision,
        logger=False,
    )
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=session.mol_dir,
        num_workers=options.num_workers,
        batch_size=options.affinity_batch_size,
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
        affinity_random_seed=options.seed,
        affinity_require_full_input=receptor_crop_indices_dir is None,
        affinity_allow_extended_full_input=options.allow_extended_full_input,
        pin_memory=options.accelerator not in ("cpu", "mps"),
    )
    click.echo(f"Running structure-free affinity for {len(pending.records)} record(s).")
    trainer.predict(session.model, datamodule=data_module, return_predictions=False)


def run_stfree_chunk(
    *,
    data: Path,
    out_dir: Path,
    options: StFreeRuntimeOptions,
    receptor_crop_indices_dir: Path | None = None,
    progress_callback: Callable[[list[str], bool], None] | None = None,
    session: StFreeRuntimeSession | None = None,
    preprocess: bool = True,
) -> None:
    """Backward-compatible wrapper that scores one physical input chunk."""
    from boltz.main import check_inputs

    data = data.expanduser().resolve()
    run_stfree_inputs(
        input_paths=check_inputs(data, skip_bad_inputs=False),
        run_dir=out_dir.expanduser().resolve() / f"boltz_results_{data.stem}",
        options=options,
        receptor_crop_indices_dir=receptor_crop_indices_dir,
        progress_callback=progress_callback,
        session=session,
        preprocess=preprocess,
    )

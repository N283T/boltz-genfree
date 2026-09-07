"""Screen-first configuration, ligand loading, and result aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import yaml
from rdkit import Chem

from boltz.data.mol import load_molecules
from boltz.data.parse.schema import standardize
from boltz.reference_crop import (
    ReferenceLigandSelector,
    load_reference_crop,
    select_receptor_tokens,
)
from boltz.stfree_runtime import StFreeRuntimeOptions

_TOP_LEVEL_KEYS = {
    "version",
    "target",
    "ligands",
    "output_dir",
    "chunk_size",
    "options",
}
_TARGET_KEYS = {"id", "sequence", "msa", "crop"}
_CROP_KEYS = {
    "reference",
    "reference_protein",
    "reference_ligand",
    "budget",
    "library_ligand_tokens",
}
_REFERENCE_LIGAND_KEYS = {"chain", "residue", "ccd"}
_LIGAND_SOURCE_KEYS = {"smiles", "csv", "sdf", "ccd"}
_MAX_SEQUENCE_PREFIX_TOKENS = 200
_MAX_AFFINITY_TOKENS = 256
_MAX_LIGAND_TOKENS = 128
_MAX_TRAINING_LIGAND_TOKENS = 56
_OPTION_ALIASES: dict[str, str] = {}
_SCREEN_OPTION_KEYS = {
    "cache",
    "affinity_checkpoint",
    "devices",
    "accelerator",
    "num_workers",
    "affinity_batch_size",
    "override",
    "seed",
    "preprocessing_threads",
    "max_msa_seqs",
    "subsample_msa",
    "num_subsampled_msa",
    "no_kernels",
    "flash_attn",
    "affinity_mw_correction",
    "affinity_recycling_steps",
    "ligand_reference_conformer",
    "allow_extended_full_input",
    "affinity_conditioner_mode",
    "etkdg_seed",
    "msa_server_url",
    "msa_pairing_strategy",
    "msa_server_username",
    "msa_server_password",
    "api_key_header",
    "api_key_value",
}


@dataclass(frozen=True)
class LigandSpec:
    """One ligand in a screen library."""

    identifier: str
    kind: str
    value: str


@dataclass(frozen=True)
class PreparedScreen:
    """Prepared chunks and normalized runtime settings."""

    config_path: Path
    output_dir: Path
    runs_dir: Path
    manifest_path: Path
    plan_path: Path
    status_path: Path
    chunk_dirs: tuple[Path, ...]
    options: dict[str, Any]
    screen_options: ScreenOptions
    target_sequence: str
    auto_msa_path: Path | None
    receptor_crop_indices_dir: Path | None


@dataclass(frozen=True)
class ScreenOptions:
    """Validated runtime and MSA-server settings."""

    runtime: StFreeRuntimeOptions
    msa_server_url: str = "https://api.colabfold.com"
    msa_pairing_strategy: str = "greedy"
    msa_server_username: str | None = None
    msa_server_password: str | None = None
    api_key_header: str | None = None
    api_key_value: str | None = None


def _utc_now() -> str:
    """Return an RFC 3339 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically publish a JSON document for concurrent readers."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise click.UsageError(f"{label} must be a YAML mapping.")
    return value


def _normalize_option_name(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise click.UsageError("Every options key must be a non-empty string.")
    normalized = name.strip().lstrip("-").replace("-", "_")
    return _OPTION_ALIASES.get(normalized, normalized)


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


def _crop_reference_provenance(path: Path, reference_sha256: str) -> dict[str, object]:
    """Identify and validate an adjacent predicted-holo bootstrap result."""
    result_path = path.parent / "bootstrap_result.json"
    if not result_path.is_file():
        return {"type": "supplied_complex"}
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise click.UsageError(
            f"Could not read crop bootstrap provenance {result_path}: {error}"
        ) from error
    if result.get("state") != "complete":
        raise click.UsageError(
            f"Crop bootstrap provenance is not complete: {result_path}"
        )
    result_reference = result.get("reference")
    if (
        not isinstance(result_reference, str)
        or Path(result_reference).resolve() != path
    ):
        raise click.UsageError(
            f"Crop bootstrap provenance references a different file: {result_path}"
        )
    if result.get("reference_sha256") != reference_sha256:
        raise click.UsageError(
            f"Crop reference checksum does not match bootstrap provenance: {path}"
        )
    return {
        "type": "predicted_holo",
        "result": str(result_path),
        "result_sha256": _sha256(result_path),
        "checkpoint": result.get("checkpoint"),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "selected_confidence": result.get("selected_confidence"),
        "target_chain": result.get("target_chain"),
        "anchor_chain": result.get("anchor_chain"),
    }


def _safe_record_id(identifier: str, index: int) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier).strip("._-")
    if not suffix:
        suffix = "ligand"
    return f"{index:06d}_{suffix[:80]}"


def _inline_smiles(value: object) -> list[LigandSpec]:
    if not isinstance(value, list) or not value:
        raise click.UsageError("ligands.smiles must be a non-empty list.")
    ligands: list[LigandSpec] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            identifier = f"ligand_{index + 1:06d}"
            smiles = item
        elif isinstance(item, dict):
            smiles = item.get("smiles")
            identifier = item.get("id", f"ligand_{index + 1:06d}")
        else:
            raise click.UsageError(
                f"ligands.smiles[{index}] must be a SMILES string or mapping."
            )
        if not isinstance(identifier, str) or not identifier.strip():
            raise click.UsageError(f"ligands.smiles[{index}].id is invalid.")
        if not isinstance(smiles, str) or not smiles.strip():
            raise click.UsageError(f"ligands.smiles[{index}].smiles is invalid.")
        ligands.append(LigandSpec(identifier.strip(), "smiles", smiles.strip()))
    return ligands


def _inline_ccd(value: object) -> list[LigandSpec]:
    if not isinstance(value, list) or not value:
        raise click.UsageError("ligands.ccd must be a non-empty list.")
    ligands: list[LigandSpec] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            code = item
            identifier = item
        elif isinstance(item, dict):
            code = item.get("ccd", item.get("code"))
            identifier = item.get("id", code)
        else:
            raise click.UsageError(
                f"ligands.ccd[{index}] must be a CCD code or mapping."
            )
        if not isinstance(code, str) or not code.strip():
            raise click.UsageError(f"ligands.ccd[{index}].ccd is invalid.")
        if not isinstance(identifier, str) or not identifier.strip():
            raise click.UsageError(f"ligands.ccd[{index}].id is invalid.")
        ligands.append(LigandSpec(identifier.strip(), "ccd", code.strip()))
    return ligands


def _csv_ligands(
    path: Path,
    *,
    id_column: str,
    smiles_column: str,
    delimiter: str | None,
) -> list[LigandSpec]:
    if not path.is_file():
        raise click.UsageError(f"Ligand CSV does not exist: {path}")
    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    if len(delimiter) != 1:
        raise click.UsageError("ligands.delimiter must be one character.")
    ligands: list[LigandSpec] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = set(reader.fieldnames or [])
        missing = {id_column, smiles_column} - fields
        if missing:
            names = ", ".join(sorted(missing))
            raise click.UsageError(f"Ligand CSV is missing columns: {names}")
        for row_number, row in enumerate(reader, start=2):
            identifier = (row.get(id_column) or "").strip()
            smiles = (row.get(smiles_column) or "").strip()
            if not identifier or not smiles:
                raise click.UsageError(
                    f"Ligand CSV row {row_number} has an empty ID or SMILES."
                )
            ligands.append(LigandSpec(identifier, "smiles", smiles))
    if not ligands:
        raise click.UsageError("Ligand CSV contains no records.")
    return ligands


def _sdf_ligands(path: Path, id_property: str | None) -> list[LigandSpec]:
    if not path.is_file():
        raise click.UsageError(f"Ligand SDF does not exist: {path}")
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    ligands: list[LigandSpec] = []
    for index, molecule in enumerate(supplier):
        if molecule is None:
            raise click.UsageError(f"Failed to parse SDF record {index + 1}: {path}")
        identifier = ""
        if id_property and molecule.HasProp(id_property):
            identifier = molecule.GetProp(id_property).strip()
        if not identifier and molecule.HasProp("_Name"):
            identifier = molecule.GetProp("_Name").strip()
        if not identifier:
            identifier = f"ligand_{index + 1:06d}"
        smiles = Chem.MolToSmiles(
            Chem.RemoveHs(molecule),
            canonical=True,
            isomericSmiles=True,
        )
        if not smiles:
            raise click.UsageError(f"SDF record {index + 1} has no valid molecule.")
        ligands.append(LigandSpec(identifier, "smiles", smiles))
    if not ligands:
        raise click.UsageError("Ligand SDF contains no records.")
    return ligands


def _load_ligands(
    config: dict[str, Any],
    base_dir: Path,
) -> tuple[list[LigandSpec], dict[str, Any]]:
    source_keys = _LIGAND_SOURCE_KEYS & set(config)
    if len(source_keys) != 1:
        choices = ", ".join(sorted(_LIGAND_SOURCE_KEYS))
        raise click.UsageError(f"ligands must define exactly one source: {choices}.")
    source = next(iter(source_keys))
    allowed_keys = {
        "smiles": {"smiles"},
        "ccd": {"ccd"},
        "csv": {"csv", "id_column", "smiles_column", "delimiter"},
        "sdf": {"sdf", "id_property"},
    }[source]
    unknown = set(config) - allowed_keys
    if unknown:
        names = ", ".join(sorted(unknown))
        raise click.UsageError(f"Unknown ligands.{source} settings: {names}")
    provenance: dict[str, Any] = {"type": source}
    if source == "smiles":
        return _inline_smiles(config[source]), provenance
    if source == "ccd":
        return _inline_ccd(config[source]), provenance

    path = _resolve_path(config[source], base_dir, f"ligands.{source}")
    if not path.is_file():
        raise click.UsageError(f"Ligand {source.upper()} does not exist: {path}")
    provenance.update({"path": str(path), "sha256": _sha256(path)})
    if source == "csv":
        id_column = config.get("id_column", "id")
        smiles_column = config.get("smiles_column", "smiles")
        delimiter = config.get("delimiter")
        if not isinstance(id_column, str) or not isinstance(smiles_column, str):
            raise click.UsageError("CSV column names must be strings.")
        ligands = _csv_ligands(
            path,
            id_column=id_column,
            smiles_column=smiles_column,
            delimiter=delimiter,
        )
    else:
        id_property = config.get("id_property")
        if id_property is not None and not isinstance(id_property, str):
            raise click.UsageError("ligands.id_property must be a string.")
        ligands = _sdf_ligands(path, id_property)
    return ligands, provenance


def _screen_cache_path(options: dict[str, Any], base_dir: Path) -> Path:
    cache = options.get("cache")
    if cache is not None:
        return _resolve_path(cache, base_dir, "options.cache")
    env_cache = os.environ.get("BOLTZ_CACHE")
    if env_cache:
        return Path(env_cache).expanduser().resolve()
    return Path("~/.boltz").expanduser().resolve()


def _smiles_token_count(ligand: LigandSpec) -> int:
    """Count ligand tokens using the same standardization as affinity parsing."""
    try:
        standardized = standardize(ligand.value)
        molecule = Chem.MolFromSmiles(standardized)
        if molecule is None:
            raise ValueError("RDKit could not parse the standardized SMILES.")
        molecule.UpdatePropertyCache(strict=True)
        molecule = Chem.AddHs(molecule)
        molecule = Chem.RemoveHs(molecule, sanitize=False)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise click.UsageError(
            f"Ligand {ligand.identifier!r} failed SMILES preprocessing: {error}"
        ) from error
    return molecule.GetNumAtoms()


def _ccd_token_counts(
    ligands: list[LigandSpec],
    *,
    cache: Path,
    ensure_ccd_cache: Callable[[Path], None] | None,
) -> dict[str, int]:
    """Load CCD molecules and return their atom-token counts."""
    codes = sorted({ligand.value for ligand in ligands})
    mol_dir = cache / "mols"
    missing = [code for code in codes if not (mol_dir / f"{code}.pkl").is_file()]
    if missing and ensure_ccd_cache is not None:
        ensure_ccd_cache(cache)
        missing = [code for code in codes if not (mol_dir / f"{code}.pkl").is_file()]
    if missing:
        shown = ", ".join(missing[:10])
        remainder = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise click.UsageError(
            "Cannot validate CCD ligand tokens because the component cache is "
            f"missing {shown}{remainder} under {mol_dir}. Run a normal screen to "
            "download the CCD cache, or set options.cache to an existing Boltz-2 "
            "cache."
        )

    molecules = load_molecules(str(mol_dir), codes)
    return {code: molecule.GetNumAtoms() for code, molecule in molecules.items()}


def _validate_full_screen(
    sequence: str,
    ligands: list[LigandSpec],
    *,
    cache: Path,
    ensure_ccd_cache: Callable[[Path], None] | None,
    reference_crop: bool = False,
    allow_extended_full_input: bool = False,
) -> list[int]:
    """Validate ligand limits and, without a reference, the complete receptor."""
    target_tokens = len(sequence)
    if (
        not reference_crop
        and not allow_extended_full_input
        and target_tokens > _MAX_SEQUENCE_PREFIX_TOKENS
    ):
        raise click.UsageError(
            "Full-screen validation failed: "
            f"target_tokens={target_tokens} exceeds the 200-token receptor limit."
        )

    ccd_ligands = [ligand for ligand in ligands if ligand.kind == "ccd"]
    ccd_counts = (
        _ccd_token_counts(
            ccd_ligands,
            cache=cache,
            ensure_ccd_cache=ensure_ccd_cache,
        )
        if ccd_ligands
        else {}
    )
    token_counts = [
        _smiles_token_count(ligand)
        if ligand.kind == "smiles"
        else ccd_counts[ligand.value]
        for ligand in ligands
    ]

    failures: list[str] = []
    warnings: list[str] = []
    for ligand, ligand_tokens in zip(ligands, token_counts, strict=True):
        reasons: list[str] = []
        if ligand_tokens > _MAX_LIGAND_TOKENS:
            reasons.append(f"ligand_tokens={ligand_tokens} exceeds 128")
        total_tokens = target_tokens + ligand_tokens
        if (
            not reference_crop
            and not allow_extended_full_input
            and total_tokens > _MAX_AFFINITY_TOKENS
        ):
            reasons.append(
                f"target_tokens={target_tokens} + ligand_tokens={ligand_tokens} "
                f"= {total_tokens} exceeds 256"
            )
        if reasons:
            failures.append(f"{ligand.identifier}: " + "; ".join(reasons))
        elif ligand_tokens > _MAX_TRAINING_LIGAND_TOKENS:
            warnings.append(f"{ligand.identifier}={ligand_tokens}")

    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures[:20])
        if len(failures) > 20:
            details += f"\n  - ... and {len(failures) - 20} more"
        raise click.UsageError(
            f"Full-screen validation failed for {len(failures)} ligand(s):\n{details}"
        )
    if warnings:
        details = ", ".join(warnings[:20])
        if len(warnings) > 20:
            details += f", ... and {len(warnings) - 20} more"
        click.echo(
            "Warning: ligand tokens above the 56-token training maximum; "
            f"affinity estimates may be inaccurate: {details}",
            err=True,
        )
    return token_counts


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    config = _mapping(loaded, "screen config")
    unknown = set(config) - _TOP_LEVEL_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise click.UsageError(f"Unknown screen config keys: {names}")
    if config.get("version", 1) != 1:
        raise click.UsageError("Only screen config version 1 is supported.")
    return config


def prepare_screen(  # noqa: C901, PLR0912, PLR0915
    config_path: Path,
    *,
    ensure_ccd_cache: Callable[[Path], None] | None = None,
) -> PreparedScreen:
    """Validate a screen config and generate deterministic Boltz YAML chunks."""
    config_path = config_path.expanduser().resolve()
    config = _load_config(config_path)
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
    if not isinstance(target_id, str) or not target_id.strip():
        raise click.UsageError("target.id must be a non-empty string.")
    target_id = target_id.strip()
    if target_id == "B":
        raise click.UsageError("target.id B is reserved for the ligand chain.")

    msa_value = target.get("msa", "auto")
    auto_msa = msa_value in ("auto", None)
    msa_provenance: dict[str, object]
    if not auto_msa and msa_value != "empty":
        msa_path = _resolve_path(msa_value, base_dir, "target.msa")
        if not msa_path.is_file():
            raise click.UsageError(f"Target MSA does not exist: {msa_path}")
        msa_value = str(msa_path)
        msa_provenance = {
            "mode": "file",
            "path": msa_value,
            "sha256": _sha256(msa_path),
        }
    elif auto_msa:
        msa_provenance = {"mode": "auto"}
    else:
        msa_provenance = {"mode": "empty"}

    raw_options = _mapping(config.get("options", {}), "options")
    options: dict[str, Any] = {}
    for raw_name, value in raw_options.items():
        name = _normalize_option_name(raw_name)
        if name in options:
            raise click.UsageError(f"Duplicate normalized option: {name}")
        options[name] = value
    if {
        "affinity_crop_indices_dir",
        "affinity_structure_free_crop_label",
    } & options.keys():
        raise click.UsageError(
            "screen is full-screen-only and does not accept affinity crop options."
        )
    screen_options = build_screen_options(options, config_dir=base_dir)

    ligands_config = _mapping(config.get("ligands"), "ligands")
    ligands, ligand_provenance = _load_ligands(ligands_config, base_dir)
    crop_config_value = target.get("crop")
    crop_config: dict[str, Any] | None = None
    crop_reference = None
    crop_reference_path: Path | None = None
    crop_reference_sha256: str | None = None
    crop_reference_provenance: dict[str, object] | None = None
    crop_policy: str | None = None
    configured_library_ligand_tokens: int | None = None
    if crop_config_value is not None:
        crop_config = _mapping(crop_config_value, "target.crop")
        unknown_crop = set(crop_config) - _CROP_KEYS
        if unknown_crop:
            names = ", ".join(sorted(unknown_crop))
            raise click.UsageError(f"Unknown target.crop settings: {names}")
        if "reference" not in crop_config:
            raise click.UsageError("target.crop.reference is required.")
        crop_reference_path = _resolve_path(
            crop_config["reference"],
            base_dir,
            "target.crop.reference",
        )
        if not crop_reference_path.is_file():
            raise click.UsageError(
                f"Crop reference does not exist: {crop_reference_path}"
            )
        crop_policy = crop_config.get("budget", "per_ligand")
        if crop_policy not in {"per_ligand", "library_fixed"}:
            raise click.UsageError(
                "target.crop.budget must be per_ligand or library_fixed."
            )
        configured_library_ligand_tokens = crop_config.get("library_ligand_tokens")
        if configured_library_ligand_tokens is not None:
            if crop_policy != "library_fixed":
                raise click.UsageError(
                    "target.crop.library_ligand_tokens requires "
                    "target.crop.budget=library_fixed."
                )
            if (
                not isinstance(configured_library_ligand_tokens, int)
                or isinstance(configured_library_ligand_tokens, bool)
                or not 1 <= configured_library_ligand_tokens <= _MAX_LIGAND_TOKENS
            ):
                raise click.UsageError(
                    "target.crop.library_ligand_tokens must be an integer "
                    f"between 1 and {_MAX_LIGAND_TOKENS}."
                )
        protein_chain = crop_config.get("reference_protein")
        if protein_chain is not None and (
            not isinstance(protein_chain, str) or not protein_chain
        ):
            raise click.UsageError(
                "target.crop.reference_protein must be a non-empty string."
            )
        reference_ligand_value = crop_config.get("reference_ligand")
        if isinstance(reference_ligand_value, str):
            if not reference_ligand_value:
                raise click.UsageError(
                    "target.crop.reference_ligand must not be empty."
                )
            ligand_selector = ReferenceLigandSelector(
                chain=reference_ligand_value,
            )
        else:
            reference_ligand = _mapping(
                reference_ligand_value,
                "target.crop.reference_ligand",
            )
            unknown_ligand = set(reference_ligand) - _REFERENCE_LIGAND_KEYS
            if unknown_ligand:
                names = ", ".join(sorted(unknown_ligand))
                raise click.UsageError(
                    f"Unknown target.crop.reference_ligand settings: {names}"
                )
            missing_ligand = _REFERENCE_LIGAND_KEYS - set(reference_ligand)
            if missing_ligand:
                names = ", ".join(sorted(missing_ligand))
                raise click.UsageError(
                    f"target.crop.reference_ligand is missing: {names}"
                )
            ligand_chain = reference_ligand["chain"]
            ligand_residue = reference_ligand["residue"]
            ligand_ccd = reference_ligand["ccd"]
            if not isinstance(ligand_chain, str) or not ligand_chain:
                raise click.UsageError(
                    "target.crop.reference_ligand.chain must be a non-empty string."
                )
            if not isinstance(ligand_residue, int) or isinstance(ligand_residue, bool):
                raise click.UsageError(
                    "target.crop.reference_ligand.residue must be an integer."
                )
            if not isinstance(ligand_ccd, str) or not ligand_ccd:
                raise click.UsageError(
                    "target.crop.reference_ligand.ccd must be a non-empty string."
                )
            ligand_selector = ReferenceLigandSelector(
                chain=ligand_chain,
                residue=ligand_residue,
                ccd=ligand_ccd.upper(),
            )
        try:
            crop_reference = load_reference_crop(
                crop_reference_path,
                sequence,
                protein_chain=protein_chain,
                ligand=ligand_selector,
            )
        except ValueError as error:
            raise click.UsageError(str(error)) from error
        crop_reference_sha256 = _sha256(crop_reference_path)
        crop_reference_provenance = _crop_reference_provenance(
            crop_reference_path,
            crop_reference_sha256,
        )
    if crop_reference is not None and screen_options.runtime.allow_extended_full_input:
        raise click.UsageError(
            "allow_extended_full_input cannot be combined with target.crop."
        )
    ligand_token_counts = _validate_full_screen(
        sequence,
        ligands,
        cache=_screen_cache_path(options, base_dir),
        ensure_ccd_cache=ensure_ccd_cache,
        reference_crop=crop_reference is not None,
        allow_extended_full_input=screen_options.runtime.allow_extended_full_input,
    )
    if (
        configured_library_ligand_tokens is not None
        and configured_library_ligand_tokens < max(ligand_token_counts)
    ):
        raise click.UsageError(
            "target.crop.library_ligand_tokens must be at least the largest "
            "ligand in this screen "
            f"({max(ligand_token_counts)})."
        )

    chunk_size = config.get("chunk_size", 1000)
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        raise click.UsageError("chunk_size must be a positive integer.")

    output_value = config.get("output_dir", f"{config_path.stem}_screen")
    output_dir = _resolve_path(output_value, base_dir, "output_dir")
    state_dir = output_dir / ".boltz-stfree"
    inputs_dir = state_dir / "inputs"
    receptor_crop_indices_dir = (
        state_dir / "crops" if crop_reference is not None else None
    )
    runs_dir = output_dir / "runs"
    manifest_path = state_dir / "screen_manifest.json"
    plan_path = output_dir / "screen_plan.json"
    status_path = output_dir / "screen_status.json"
    auto_msa_path = state_dir / "msa" / "target.csv" if auto_msa else None
    if auto_msa_path is not None:
        msa_provenance["path"] = str(auto_msa_path)

    records: list[dict[str, Any]] = []
    expected_files: dict[Path, str] = {}
    chunk_dirs: list[Path] = []
    library_ligand_tokens = (
        configured_library_ligand_tokens or max(ligand_token_counts)
        if crop_reference is not None and crop_policy == "library_fixed"
        else None
    )
    receptor_indices_by_ligand_tokens: dict[int, tuple[int, ...]] = {}
    for index, (ligand, ligand_tokens) in enumerate(
        zip(ligands, ligand_token_counts, strict=True),
        start=1,
    ):
        chunk_index = (index - 1) // chunk_size
        chunk_dir = inputs_dir / f"chunk_{chunk_index:05d}"
        if not chunk_dirs or chunk_dirs[-1] != chunk_dir:
            chunk_dirs.append(chunk_dir)
        record_id = _safe_record_id(ligand.identifier, index)
        protein: dict[str, object] = {"id": target_id, "sequence": sequence}
        if msa_value == "empty":
            protein["msa"] = "empty"
        elif auto_msa_path is not None:
            protein["msa"] = str(auto_msa_path)
        elif isinstance(msa_value, str):
            protein["msa"] = msa_value
        ligand_entity = {"id": "B", ligand.kind: ligand.value}
        payload = {
            "version": 1,
            "sequences": [
                {"protein": protein},
                {"ligand": ligand_entity},
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        rendered = yaml.safe_dump(payload, sort_keys=False)
        input_path = chunk_dir / f"{record_id}.yaml"
        expected_files[input_path] = rendered
        record = {
            "record_id": record_id,
            "ligand_id": ligand.identifier,
            "ligand_kind": ligand.kind,
            "ligand_value": ligand.value,
            "ligand_tokens": ligand_tokens,
            "chunk": chunk_dir.name,
        }
        if crop_reference is not None:
            reserved_ligand_tokens = (
                library_ligand_tokens
                if library_ligand_tokens is not None
                else ligand_tokens
            )
            try:
                receptor_indices = receptor_indices_by_ligand_tokens.get(
                    reserved_ligand_tokens
                )
                if receptor_indices is None:
                    receptor_indices = select_receptor_tokens(
                        crop_reference,
                        sequence,
                        ligand_tokens=reserved_ligand_tokens,
                    )
                    receptor_indices_by_ligand_tokens[reserved_ligand_tokens] = (
                        receptor_indices
                    )
            except ValueError as error:
                raise click.UsageError(
                    f"Reference crop failed for {ligand.identifier}: {error}"
                ) from error
            assert receptor_crop_indices_dir is not None
            crop_path = receptor_crop_indices_dir / f"affinity_crop_{record_id}.json"
            crop_document = {
                "schema_version": 1,
                "record_id": record_id,
                "selection": "reference_contiguous_blocks",
                "budget": crop_policy,
                "neighborhood_size": 10,
                "reference": str(crop_reference_path),
                "reference_sha256": crop_reference_sha256,
                "reference_source": crop_reference_provenance,
                "reference_protein": crop_reference.protein_chain,
                "reference_ligand": {
                    "chain": crop_reference.ligand_chain,
                    "residue": crop_reference.ligand_residue,
                    "ccd": crop_reference.ligand_ccd,
                },
                "ligand_tokens_reserved": reserved_ligand_tokens,
                "receptor_token_idx": list(receptor_indices),
            }
            expected_files[crop_path] = json.dumps(crop_document, indent=2) + "\n"
            record["crop"] = {
                "budget": crop_policy,
                "receptor_tokens": len(receptor_indices),
                "path": str(crop_path),
            }
        records.append(record)

    input_config = {key: value for key, value in config.items() if key != "options"}
    input_config["preprocessing"] = {
        "component_cache": str(screen_options.runtime.cache),
        "max_msa_seqs": screen_options.runtime.max_msa_seqs,
    }
    if auto_msa:
        input_config["preprocessing"]["auto_msa"] = {
            "server_url": screen_options.msa_server_url,
            "pairing_strategy": screen_options.msa_pairing_strategy,
        }
    manifest = {
        "schema_version": 2,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(
            json.dumps(
                input_config,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "target": {
            "id": target_id,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "length": len(sequence),
            "msa": msa_provenance,
        },
        "ligand_source": ligand_provenance,
        "chunk_size": chunk_size,
        "output_dir": str(output_dir),
        "records": records,
    }
    if crop_reference is not None:
        assert crop_reference_path is not None
        manifest["target"]["crop"] = {
            "selection": "reference_contiguous_blocks",
            "reference": str(crop_reference_path),
            "reference_sha256": crop_reference_sha256,
            "reference_source": crop_reference_provenance,
            "protein_chain": crop_reference.protein_chain,
            "ligand": {
                "chain": crop_reference.ligand_chain,
                "residue": crop_reference.ligand_residue,
                "ccd": crop_reference.ligand_ccd,
            },
            "budget": crop_policy,
            "neighborhood_size": 10,
        }
        if library_ligand_tokens is not None:
            manifest["target"]["crop"]["library_ligand_tokens"] = library_ligand_tokens
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise click.UsageError(
                f"Existing screen state does not match this config: {manifest_path}"
            )

    expected_paths = set(expected_files)
    if inputs_dir.is_dir():
        found_paths = set(inputs_dir.glob("chunk_*/*.yaml"))
        unexpected = found_paths - expected_paths
        if unexpected:
            raise click.UsageError(
                f"Generated input directory contains unexpected files: {inputs_dir}"
            )
    if receptor_crop_indices_dir is not None and receptor_crop_indices_dir.is_dir():
        expected_crop_paths = {
            path for path in expected_paths if path.parent == receptor_crop_indices_dir
        }
        unexpected_crops = set(receptor_crop_indices_dir.glob("affinity_crop_*.json"))
        unexpected_crops -= expected_crop_paths
        if unexpected_crops:
            raise click.UsageError(
                "Generated crop directory contains unexpected files: "
                f"{receptor_crop_indices_dir}"
            )
    for path, rendered in expected_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.read_text(encoding="utf-8") != rendered:
            raise click.UsageError(f"Generated input differs from config: {path}")
        if not path.is_file():
            path.write_text(rendered, encoding="utf-8")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    runs_dir.mkdir(parents=True, exist_ok=True)
    prepared = PreparedScreen(
        config_path=config_path,
        output_dir=output_dir,
        runs_dir=runs_dir,
        manifest_path=manifest_path,
        plan_path=plan_path,
        status_path=status_path,
        chunk_dirs=tuple(chunk_dirs),
        options=options,
        screen_options=screen_options,
        target_sequence=sequence,
        auto_msa_path=auto_msa_path,
        receptor_crop_indices_dir=receptor_crop_indices_dir,
    )
    _atomic_write_json(plan_path, build_screen_plan(prepared, manifest))
    return prepared


def ensure_auto_msa(
    prepared: PreparedScreen,
    *,
    screen_options: ScreenOptions,
    compute_msa_fn: Any,
) -> None:
    """Generate one shared automatic MSA and verify it on later resumes."""
    msa_path = prepared.auto_msa_path
    if msa_path is None:
        return
    metadata_path = msa_path.with_suffix(".metadata.json")
    sequence_sha256 = hashlib.sha256(prepared.target_sequence.encode()).hexdigest()
    server_url = screen_options.msa_server_url
    pairing_strategy = screen_options.msa_pairing_strategy

    def write_metadata(*, recovered: bool = False) -> None:
        metadata = {
            "schema_version": 1,
            "path": str(msa_path),
            "sha256": _sha256(msa_path),
            "sequence_sha256": sequence_sha256,
            "server_url": server_url,
            "pairing_strategy": pairing_strategy,
            "recovered_without_prior_metadata": recovered,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    if msa_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("sequence_sha256") != sequence_sha256:
            raise click.ClickException(
                f"Shared MSA sequence does not match the screen: {msa_path}"
            )
        if metadata.get("sha256") != _sha256(msa_path):
            raise click.ClickException(
                f"Shared MSA content changed after generation: {msa_path}"
            )
        if (
            metadata.get("server_url") != server_url
            or metadata.get("pairing_strategy") != pairing_strategy
        ):
            raise click.ClickException(
                f"Shared MSA source settings do not match the screen state: {msa_path}"
            )
        return
    if msa_path.is_file():
        with msa_path.open(encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        if first_line != "key,sequence":
            raise click.ClickException(f"Shared MSA is incomplete: {msa_path}")
        write_metadata(recovered=True)
        return
    if metadata_path.exists() and not msa_path.is_file():
        raise click.ClickException(
            f"Shared MSA is missing but provenance remains: {msa_path}"
        )

    msa_path.parent.mkdir(parents=True, exist_ok=True)
    username = screen_options.msa_server_username or os.environ.get(
        "BOLTZ_MSA_USERNAME"
    )
    password = screen_options.msa_server_password or os.environ.get(
        "BOLTZ_MSA_PASSWORD"
    )
    api_key_value = screen_options.api_key_value or os.environ.get("MSA_API_KEY_VALUE")
    compute_msa_fn(
        data={"target": prepared.target_sequence},
        target_id="screen",
        msa_dir=msa_path.parent,
        msa_server_url=server_url,
        msa_pairing_strategy=pairing_strategy,
        msa_server_username=username,
        msa_server_password=password,
        api_key_header=screen_options.api_key_header,
        api_key_value=api_key_value,
    )
    if not msa_path.is_file():
        raise click.ClickException(
            f"MSA generation did not create the expected file: {msa_path}"
        )
    write_metadata()


def _integer_option(
    options: dict[str, Any],
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = options.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise click.UsageError(f"options.{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise click.UsageError(f"options.{name} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise click.UsageError(f"options.{name} must be at most {maximum}.")
    return value


def _boolean_option(options: dict[str, Any], name: str, default: bool) -> bool:
    value = options.get(name, default)
    if not isinstance(value, bool):
        raise click.UsageError(f"options.{name} must be true or false.")
    return value


def _optional_string(options: dict[str, Any], name: str) -> str | None:
    value = options.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise click.UsageError(f"options.{name} must be a non-empty string.")
    return value


def _devices_option(value: object) -> int | list[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(value, str):
        try:
            parsed = [int(item.strip()) for item in value.split(",")]
        except ValueError as error:
            raise click.UsageError("options.devices is invalid.") from error
        if len(parsed) == 1 and parsed[0] > 0:
            return parsed[0]
        if len(parsed) > 1 and all(device >= 0 for device in parsed):
            return parsed
    if (
        isinstance(value, list)
        and value
        and all(
            isinstance(device, int) and not isinstance(device, bool) and device >= 0
            for device in value
        )
    ):
        return value
    raise click.UsageError(
        "options.devices must be a positive count or a list of GPU indices."
    )


def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def build_screen_options(
    raw_options: dict[str, Any],
    *,
    config_dir: Path,
) -> ScreenOptions:
    """Validate the deliberately small Boltz-StFree option surface."""
    unknown = set(raw_options) - _SCREEN_OPTION_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise click.UsageError(f"Unknown Boltz-StFree options: {names}")

    accelerator = raw_options.get("accelerator", "gpu")
    if accelerator not in {"gpu", "cpu", "mps"}:
        raise click.UsageError("options.accelerator must be gpu, cpu, or mps.")
    devices = _devices_option(raw_options.get("devices", 1))
    if accelerator in {"cpu", "mps"} and devices != 1:
        raise click.UsageError(
            f"options.devices must be 1 when accelerator is {accelerator}."
        )
    cache = _screen_cache_path(raw_options, config_dir)
    checkpoint_value = raw_options.get("affinity_checkpoint")
    checkpoint = (
        _resolve_path(checkpoint_value, config_dir, "options.affinity_checkpoint")
        if checkpoint_value is not None
        else None
    )
    if checkpoint is not None and not checkpoint.is_file():
        raise click.UsageError(
            f"options.affinity_checkpoint does not exist: {checkpoint}"
        )
    seed = raw_options.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise click.UsageError("options.seed must be an integer or null.")
    ligand_reference_conformer = raw_options.get(
        "ligand_reference_conformer",
        "skip",
    )
    if ligand_reference_conformer not in {"skip", "etkdg", "rdkit_2d"}:
        raise click.UsageError(
            "options.ligand_reference_conformer must be skip, etkdg, or rdkit_2d."
        )
    affinity_batch_size = _integer_option(
        raw_options,
        "affinity_batch_size",
        1,
        minimum=1,
    )
    subsample_msa = _boolean_option(raw_options, "subsample_msa", True)

    conditioner_mode = raw_options.get("affinity_conditioner_mode", "original")
    if conditioner_mode not in {"original", "folded"}:
        raise click.UsageError(
            "options.affinity_conditioner_mode must be original or folded."
        )
    runtime = StFreeRuntimeOptions(
        cache=cache,
        affinity_checkpoint=checkpoint,
        devices=devices,
        accelerator=accelerator,
        num_workers=_integer_option(raw_options, "num_workers", 2, minimum=0),
        affinity_batch_size=affinity_batch_size,
        override=_boolean_option(raw_options, "override", False),
        seed=seed,
        preprocessing_threads=_integer_option(
            raw_options,
            "preprocessing_threads",
            _cpu_count(),
            minimum=1,
        ),
        max_msa_seqs=_integer_option(
            raw_options,
            "max_msa_seqs",
            8192,
            minimum=1,
        ),
        subsample_msa=subsample_msa,
        num_subsampled_msa=_integer_option(
            raw_options,
            "num_subsampled_msa",
            1024,
            minimum=1,
        ),
        no_kernels=_boolean_option(raw_options, "no_kernels", False),
        flash_attn=_boolean_option(raw_options, "flash_attn", False),
        affinity_mw_correction=_boolean_option(
            raw_options,
            "affinity_mw_correction",
            False,
        ),
        affinity_recycling_steps=_integer_option(
            raw_options,
            "affinity_recycling_steps",
            0,
            minimum=0,
            maximum=5,
        ),
        ligand_reference_conformer=ligand_reference_conformer,
        affinity_conditioner_mode=conditioner_mode,
        allow_extended_full_input=_boolean_option(
            raw_options, "allow_extended_full_input", False
        ),
        etkdg_seed=_integer_option(
            raw_options,
            "etkdg_seed",
            42,
            minimum=0,
            maximum=2_147_483_647,
        ),
    )
    server_url = raw_options.get("msa_server_url", "https://api.colabfold.com")
    pairing = raw_options.get("msa_pairing_strategy", "greedy")
    if not isinstance(server_url, str) or not server_url:
        raise click.UsageError("options.msa_server_url must be a non-empty string.")
    if pairing not in {"greedy", "complete"}:
        raise click.UsageError(
            "options.msa_pairing_strategy must be greedy or complete."
        )
    return ScreenOptions(
        runtime=runtime,
        msa_server_url=server_url,
        msa_pairing_strategy=pairing,
        msa_server_username=_optional_string(raw_options, "msa_server_username"),
        msa_server_password=_optional_string(raw_options, "msa_server_password"),
        api_key_header=_optional_string(raw_options, "api_key_header"),
        api_key_value=_optional_string(raw_options, "api_key_value"),
    )


def _screen_score_matches(
    runs_dir: Path,
    record: dict[str, object],
) -> list[Path]:
    """Return structure-free score files for one screen manifest record."""
    record_id = str(record["record_id"])
    prediction_dirs = (
        runs_dir / "boltz_results_screen" / "predictions" / record_id,
        runs_dir / f"boltz_results_{record['chunk']}" / "predictions" / record_id,
    )
    return sorted(
        score
        for prediction_dir in prediction_dirs
        for score in prediction_dir.glob(
            "affinity_*__distogram_zero_distogram_features*.json"
        )
    )


def build_screen_plan(
    prepared: PreparedScreen,
    manifest: dict[str, object],
) -> dict[str, object]:
    """Build the durable, non-secret execution plan for a prepared screen."""
    records = manifest["records"]
    assert isinstance(records, list)
    token_counts = [int(record["ligand_tokens"]) for record in records]
    chunk_counts: dict[str, int] = {}
    receptor_counts: list[int] = []
    for record in records:
        chunk = str(record["chunk"])
        chunk_counts[chunk] = chunk_counts.get(chunk, 0) + 1
        crop = record.get("crop")
        if isinstance(crop, dict):
            receptor_counts.append(int(crop["receptor_tokens"]))

    runtime = prepared.screen_options.runtime
    target = manifest["target"]
    assert isinstance(target, dict)
    ligand_source = manifest["ligand_source"]
    assert isinstance(ligand_source, dict)
    receptor_selection: dict[str, object]
    crop = target.get("crop")
    if isinstance(crop, dict):
        receptor_selection = {
            "mode": "reference_crop",
            "budget": crop["budget"],
            "receptor_tokens_min": min(receptor_counts),
            "receptor_tokens_max": max(receptor_counts),
            "reference": crop["reference"],
            "reference_sha256": crop["reference_sha256"],
        }
        if "library_ligand_tokens" in crop:
            receptor_selection["library_ligand_tokens"] = crop["library_ligand_tokens"]
    else:
        receptor_selection = {
            "mode": "complete_input",
            "receptor_tokens": target["length"],
        }

    checkpoint = runtime.affinity_checkpoint or runtime.cache / "boltz2_aff.ckpt"
    return {
        "schema_version": 1,
        "config": str(prepared.config_path),
        "manifest": str(prepared.manifest_path),
        "target": {
            "id": target["id"],
            "sequence_length": target["length"],
            "msa": target["msa"],
            "receptor_selection": receptor_selection,
        },
        "library": {
            "source": ligand_source,
            "records": len(records),
            "ligand_tokens_min": min(token_counts),
            "ligand_tokens_max": max(token_counts),
            "above_training_token_limit": sum(
                count > _MAX_TRAINING_LIGAND_TOKENS for count in token_counts
            ),
        },
        "execution": {
            "chunk_size": manifest["chunk_size"],
            "chunk_count": len(chunk_counts),
            "runtime_layout": "unified",
            "preprocessing_passes": 1,
            "prediction_passes": 1,
            "chunks": [
                {"name": name, "records": count} for name, count in chunk_counts.items()
            ],
            "accelerator": runtime.accelerator,
            "devices": runtime.devices,
            "num_workers": runtime.num_workers,
            "affinity_batch_size": runtime.affinity_batch_size,
            "preprocessing_workers": runtime.preprocessing_threads,
            "seed": runtime.seed,
            "subsample_msa": runtime.subsample_msa,
            "num_subsampled_msa": runtime.num_subsampled_msa,
            "affinity_recycling_steps": runtime.affinity_recycling_steps,
            "ligand_reference_conformer": runtime.ligand_reference_conformer,
            "allow_extended_full_input": runtime.allow_extended_full_input,
            "affinity_conditioner_mode": runtime.affinity_conditioner_mode,
            "checkpoint": str(checkpoint.expanduser().resolve()),
        },
        "outputs": {
            "directory": str(prepared.output_dir),
            "status": str(prepared.status_path),
            "scores": str(prepared.output_dir / "scores.csv"),
        },
    }


def format_screen_plan(plan: dict[str, object]) -> tuple[str, ...]:
    """Format a concise terminal summary of a screen plan."""
    target = plan["target"]
    library = plan["library"]
    execution = plan["execution"]
    outputs = plan["outputs"]
    assert isinstance(target, dict)
    assert isinstance(library, dict)
    assert isinstance(execution, dict)
    assert isinstance(outputs, dict)
    receptor = target["receptor_selection"]
    msa = target["msa"]
    assert isinstance(receptor, dict)
    assert isinstance(msa, dict)
    receptor_label = str(receptor["mode"])
    if receptor_label == "reference_crop":
        receptor_label += (
            f" ({receptor['receptor_tokens_min']}-"
            f"{receptor['receptor_tokens_max']} receptor tokens)"
        )
    else:
        receptor_label += f" ({receptor['receptor_tokens']} receptor tokens)"
    return (
        f"Target: {target['sequence_length']} residues, {receptor_label}, "
        f"MSA {msa['mode']}",
        f"Library: {library['records']} records, ligand tokens "
        f"{library['ligand_tokens_min']}-{library['ligand_tokens_max']}, "
        f"{library['above_training_token_limit']} above training limit",
        f"Execution: {execution['chunk_count']} chunk(s) of at most "
        f"{execution['chunk_size']}, {execution['accelerator']} on "
        f"{execution['devices']}, batch {execution['affinity_batch_size']}, "
        f"seed {execution['seed']}",
        f"Status: {outputs['status']}",
    )


class ScreenProgress:
    """Publish resumable screen progress as an atomic JSON snapshot."""

    def __init__(
        self,
        prepared: PreparedScreen,
        *,
        ignore_existing: bool = False,
    ) -> None:
        self.prepared = prepared
        self.manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
        records = self.manifest["records"]
        self.records = {str(record["record_id"]): record for record in records}
        self.chunk_order = [path.name for path in prepared.chunk_dirs]
        self.record_chunks = {
            record_id: str(record["chunk"])
            for record_id, record in self.records.items()
        }
        self.chunk_record_ids = {chunk: [] for chunk in self.chunk_order}
        for record_id, chunk in self.record_chunks.items():
            self.chunk_record_ids[chunk].append(record_id)
        self.chunk_completed = {chunk: 0 for chunk in self.chunk_order}
        self.chunk_failed = {chunk: 0 for chunk in self.chunk_order}
        self.completed: set[str] = set()
        self.failed: set[str] = set()
        if not ignore_existing:
            for record_id, record in self.records.items():
                if len(_screen_score_matches(prepared.runs_dir, record)) == 1:
                    self._set_record_state(record_id, failed=False)
        self.state = "prepared"
        self.message: str | None = None
        self.current_chunk: str | None = None
        self.started_at: str | None = None
        self._publish()

    def _set_record_state(self, record_id: str, *, failed: bool) -> None:
        chunk = self.record_chunks[record_id]
        if failed:
            if record_id in self.completed:
                self.completed.remove(record_id)
                self.chunk_completed[chunk] -= 1
            if record_id not in self.failed:
                self.failed.add(record_id)
                self.chunk_failed[chunk] += 1
        else:
            if record_id not in self.completed:
                self.completed.add(record_id)
                self.chunk_completed[chunk] += 1
            if record_id in self.failed:
                self.failed.remove(record_id)
                self.chunk_failed[chunk] -= 1

    def _refresh_chunk(self, chunk: str) -> None:
        for record_id in self.chunk_record_ids[chunk]:
            record = self.records[record_id]
            matches = _screen_score_matches(self.prepared.runs_dir, record)
            if len(matches) == 1:
                self._set_record_state(record_id, failed=False)

    def _snapshot(self) -> dict[str, object]:
        chunks = []
        for chunk in self.chunk_order:
            expected = len(self.chunk_record_ids[chunk])
            completed = self.chunk_completed[chunk]
            failed = self.chunk_failed[chunk]
            if completed == expected:
                state = "complete"
            elif chunk == self.current_chunk:
                state = "running"
            elif completed or failed:
                state = "partial"
            else:
                state = "pending"
            chunks.append(
                {
                    "name": chunk,
                    "state": state,
                    "expected": expected,
                    "completed": completed,
                    "failed": failed,
                }
            )
        expected = len(self.records)
        completed = len(self.completed)
        failed = len(self.failed)
        return {
            "schema_version": 1,
            "state": self.state,
            "message": self.message,
            "started_at": self.started_at,
            "updated_at": _utc_now(),
            "expected": expected,
            "completed": completed,
            "failed": failed,
            "pending": expected - completed,
            "percent_complete": round(100 * completed / expected, 2),
            "current_chunk": self.current_chunk,
            "chunks": chunks,
            "plan": str(self.prepared.plan_path),
            "scores": str(self.prepared.output_dir / "scores.csv"),
        }

    def _publish(self) -> None:
        _atomic_write_json(self.prepared.status_path, self._snapshot())

    def set_stage(self, state: str, message: str | None = None) -> None:
        """Publish a screen-level stage transition."""
        if self.started_at is None and state != "prepared":
            self.started_at = _utc_now()
        self.state = state
        self.message = message
        self._publish()

    def start_chunk(self, chunk: str) -> None:
        """Mark one chunk as active."""
        self.current_chunk = chunk
        self.set_stage("running", f"Running {chunk}")

    def record_batch(self, record_ids: list[str], failed: bool) -> None:
        """Publish completion or failure for a prediction batch."""
        chunks = {
            self.record_chunks[record_id]
            for record_id in record_ids
            if record_id in self.record_chunks
        }
        if chunks:
            self.current_chunk = min(chunks, key=self.chunk_order.index)
        for record_id in record_ids:
            if record_id not in self.records:
                raise ValueError(
                    f"Unknown screen record in progress update: {record_id}"
                )
            self._set_record_state(record_id, failed=failed)
        self._publish()

    def finish_chunk(self, chunk: str) -> None:
        """Rescan a finished chunk and publish its durable result count."""
        self._refresh_chunk(chunk)
        self.current_chunk = None
        self._publish()

    def complete(self, scores_path: Path) -> None:
        """Publish successful screen completion."""
        self.completed = set(self.records)
        self.failed.clear()
        self.chunk_completed = {
            chunk: len(record_ids)
            for chunk, record_ids in self.chunk_record_ids.items()
        }
        self.chunk_failed = {chunk: 0 for chunk in self.chunk_order}
        self.current_chunk = None
        self.set_stage("complete", f"Scores written to {scores_path}")

    def fail(self, error: BaseException) -> None:
        """Publish a terminal failure or interruption without hiding the error."""
        self.current_chunk = None
        state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        self.set_stage(state, f"{type(error).__name__}: {error}")


def invalidate_screen_aggregates(prepared: PreparedScreen) -> None:
    """Remove published aggregates before an explicit override starts."""
    for name in (
        "scores.csv",
        "screen_summary.json",
        ".scores.csv.tmp",
        ".screen_summary.json.tmp",
    ):
        path = prepared.output_dir / name
        if not path.exists():
            continue
        if not path.is_file() and not path.is_symlink():
            raise click.ClickException(f"Cannot replace non-file screen output: {path}")
        try:
            path.unlink()
        except OSError as error:
            raise click.ClickException(
                f"Could not invalidate screen output for override: {path}"
            ) from error


def collect_screen_scores(prepared: PreparedScreen) -> Path:
    """Aggregate compact score JSON files into one CSV and summary JSON."""
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for record in manifest["records"]:
        matches = _screen_score_matches(prepared.runs_dir, record)
        if not matches:
            missing.append(record["record_id"])
            continue
        if len(matches) != 1:
            raise click.ClickException(
                f"Multiple structure-free scores found for {record['record_id']}."
            )
        score = json.loads(matches[0].read_text(encoding="utf-8"))
        rows.append(
            {
                **record,
                "affinity_pred_value": score["affinity_pred_value"],
                "affinity_probability_binary": score["affinity_probability_binary"],
            }
        )

    scores_path = prepared.output_dir / "scores.csv"
    columns = [
        "record_id",
        "ligand_id",
        "ligand_kind",
        "ligand_value",
        "ligand_tokens",
        "chunk",
        "crop_budget",
        "receptor_tokens",
        "affinity_pred_value",
        "affinity_probability_binary",
    ]
    scores_temp = prepared.output_dir / ".scores.csv.tmp"
    with scores_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            crop = row.pop("crop", None)
            row["crop_budget"] = crop["budget"] if crop else "complete_input"
            row["receptor_tokens"] = (
                crop["receptor_tokens"] if crop else manifest["target"]["length"]
            )
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "expected": len(manifest["records"]),
        "completed": len(rows),
        "missing": missing,
        "scores": str(scores_path),
    }
    summary_path = prepared.output_dir / "screen_summary.json"
    summary_temp = prepared.output_dir / ".screen_summary.json.tmp"
    summary_temp.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    scores_temp.replace(scores_path)
    summary_temp.replace(summary_path)
    if missing:
        raise click.ClickException(
            f"Screen is incomplete: {len(rows)}/{len(manifest['records'])} scores."
        )
    return scores_path

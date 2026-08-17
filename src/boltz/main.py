from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import socket
import tarfile
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import click

if TYPE_CHECKING:
    from boltz.data.types import Manifest

MOL_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"

_PROCESSED_MSA_CACHE_VERSION = 1
_MSA_HASH_CACHE: dict[tuple[str, int, int, int], str] = {}
_MSA_LOCK_POLL_SECONDS = 0.05
_MSA_LOCK_TIMEOUT_SECONDS = 3600.0
_PROCESS_INPUT_WORKER: Callable[[Path], Optional[str]] | None = None

BOLTZ2_REFERENCE_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
]

BOLTZ2_AFFINITY_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_aff.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
]


def _runtime_version() -> str:
    """Return this distribution version with source-tree fallbacks."""
    try:
        return _pkg_version("boltz-stfree")
    except PackageNotFoundError:
        try:
            return _pkg_version("boltz-community")
        except PackageNotFoundError:
            try:
                return _pkg_version("boltz")
            except PackageNotFoundError:
                return "0+unknown"


def _resolved_path_for_message(path: Path) -> Path:
    """Return a best-effort resolved path for user-facing error messages."""
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser()


def _checkpoint_needs_download(path: Path, description: str) -> bool:
    """Return True when a cached checkpoint should be downloaded again."""
    resolved = _resolved_path_for_message(path)

    if not path.exists():
        return True
    if not path.is_file():
        msg = f"{description} checkpoint path is not a file: {resolved}"
        raise RuntimeError(msg)

    try:
        size = path.stat().st_size
    except OSError as e:
        msg = f"Could not read {description} checkpoint at {resolved}: {e}"
        raise RuntimeError(msg) from e

    if size == 0:
        click.echo(
            f"Found empty {description} checkpoint at {resolved}; re-downloading."
        )
        return True

    return False


def _validate_checkpoint_for_load(path: Path, description: str) -> None:
    """Raise a clear CLI error for missing or empty checkpoint files."""
    resolved = _resolved_path_for_message(path)

    if not path.exists():
        msg = (
            f"{description} checkpoint not found at {resolved}. "
            "Delete any partial file and retry so boltz can re-download it, "
            "or pass a valid checkpoint path."
        )
        raise click.ClickException(msg)
    if not path.is_file():
        msg = f"{description} checkpoint path is not a file: {resolved}"
        raise click.ClickException(msg)

    try:
        size = path.stat().st_size
    except OSError as e:
        msg = f"Could not read {description} checkpoint at {resolved}: {e}"
        raise click.ClickException(msg) from e

    if size == 0:
        msg = (
            f"{description} checkpoint is empty: {resolved}. "
            "Delete the file and retry so boltz can re-download it, "
            "or pass a valid checkpoint path."
        )
        raise click.ClickException(msg)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_input_sources(data: Union[Path, List[Path]]) -> list[str]:
    """Return stable absolute paths after input expansion."""
    paths = [data] if isinstance(data, Path) else data
    return [str(path.resolve()) for path in paths]


def _available_cpu_count() -> int:
    """Return the number of CPUs available to this process.

    Respects cgroup/taskset limits on Linux via os.sched_getaffinity.
    Falls back to os.cpu_count on macOS or on error.
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _configure_wsl_multiprocessing_tempdir(num_workers: int) -> Optional[Path]:
    """Keep multiprocessing Unix sockets off Windows-mounted WSL paths."""
    if num_workers < 1 or "microsoft" not in platform.release().lower():
        return None
    current = Path(tempfile.gettempdir())
    if not str(current).startswith("/mnt/"):
        return None
    linux_tmp = Path("/tmp")
    if not linux_tmp.is_dir() or not os.access(linux_tmp, os.W_OK | os.X_OK):
        return None
    os.environ["TMPDIR"] = str(linux_tmp)
    # gettempdir() caches its result, so update the parent process as well as
    # the environment inherited by DataLoader workers.
    tempfile.tempdir = str(linux_tmp)
    click.echo(f"WSL multiprocessing temp directory: {current} -> {linux_tmp}")
    return linux_tmp


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path
    constraints_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    extra_mols_dir: Optional[Path] = None


@dataclass
class PairformerArgsV2:
    """Pairformer arguments."""

    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False


@dataclass
class MSAModuleArgs:
    """MSA module arguments."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024


@dataclass
class Boltz2DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True


@dataclass
class BoltzSteeringParams:
    """Steering parameters."""

    fk_steering: bool = False
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    physical_guidance_update: bool = False
    contact_guidance_update: bool = True
    num_gd_steps: int = 20


def _download(url: str, dest: str) -> None:
    """Download a file with proper redirect handling.

    Uses requests instead of urllib to handle HuggingFace redirects
    and to use certifi SSL certificates (avoids macOS SSL errors).
    """
    import requests

    response = requests.get(url, stream=True, allow_redirects=True, timeout=300)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def download_boltz2_ccd(cache: Path) -> None:
    """Download and extract the Boltz-2 CCD component cache."""
    cache.mkdir(parents=True, exist_ok=True)
    mols = cache / "mols"
    tar_mols = cache / "mols.tar"
    if not mols.exists():
        if not tar_mols.exists():
            click.echo(
                f"Downloading the CCD data to {tar_mols}. "
                "This may take a bit of time. You may change the cache directory "
                "with the --cache flag."
            )
            _download(MOL_URL, str(tar_mols))
        click.echo(
            f"Extracting the CCD data to {mols}. "
            "This may take a bit of time. You may change the cache directory "
            "with the --cache flag."
        )
        with tarfile.open(str(tar_mols), "r") as tar:
            tar.extractall(cache)  # noqa: S202


def download_stfree(
    cache: Path,
    *,
    include_default_affinity: bool = True,
) -> None:
    """Download component data and, when needed, the default affinity checkpoint."""
    download_boltz2_ccd(cache)
    if not include_default_affinity:
        return
    affinity_model = cache / "boltz2_aff.ckpt"
    if _checkpoint_needs_download(affinity_model, "Boltz-2 affinity weights"):
        click.echo(
            f"Downloading the Boltz-2 affinity weights to {affinity_model}. You may "
            "change the cache directory with options.cache."
        )
        for index, url in enumerate(BOLTZ2_AFFINITY_URL_WITH_FALLBACK):
            try:
                _download(url, str(affinity_model))
                _validate_checkpoint_for_load(
                    affinity_model,
                    "Boltz-2 affinity weights",
                )
                break
            except Exception as error:  # noqa: BLE001
                if index == len(BOLTZ2_AFFINITY_URL_WITH_FALLBACK) - 1:
                    msg = (
                        "Failed to download the affinity checkpoint from all URLs. "
                        f"Last error: {error}"
                    )
                    raise RuntimeError(msg) from error


def download_boltz2_reference_assets(cache: Path, checkpoint: Path) -> None:
    """Ensure assets for an explicitly requested holo reference prediction."""
    download_boltz2_ccd(cache)
    default_checkpoint = (cache / "boltz2_conf.ckpt").resolve()
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint != default_checkpoint:
        _validate_checkpoint_for_load(checkpoint, "Boltz-2 weights")
        return
    if not _checkpoint_needs_download(checkpoint, "Boltz-2 weights"):
        return
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Downloading the Boltz-2 structure weights to {checkpoint}.")
    for index, url in enumerate(BOLTZ2_REFERENCE_URL_WITH_FALLBACK):
        try:
            _download(url, str(checkpoint))
            _validate_checkpoint_for_load(checkpoint, "Boltz-2 weights")
            return
        except Exception as error:  # noqa: BLE001
            if index == len(BOLTZ2_REFERENCE_URL_WITH_FALLBACK) - 1:
                msg = (
                    "Failed to download the Boltz-2 structure checkpoint from "
                    f"all URLs. Last error: {error}"
                )
                raise RuntimeError(msg) from error


def get_cache_path() -> str:
    """Determine the cache path, prioritising the BOLTZ_CACHE environment variable.

    Returns
    -------
    str: Path
        Path to use for boltz cache location.

    """
    env_cache = os.environ.get("BOLTZ_CACHE")
    if env_cache:
        resolved_cache = Path(env_cache).expanduser().resolve()
        if not resolved_cache.is_absolute():
            msg = f"BOLTZ_CACHE must be an absolute path, got: {env_cache}"
            raise ValueError(msg)
        return str(resolved_cache)

    return str(Path("~/.boltz").expanduser())


def check_inputs(data: Path, skip_bad_inputs: bool = False) -> list[Path]:
    """Check the input data and output directory.

    Parameters
    ----------
    data : Path
        The input data.
    skip_bad_inputs : bool
        If True, filter out invalid entries and warn instead of aborting.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = sorted(data.glob("*"))

        # Filter out non .fasta or .yaml files
        valid = []
        for d in data:
            if d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
            elif d.suffix.lower() not in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
            else:
                valid.append(d)
                continue

            if not skip_bad_inputs:
                raise RuntimeError(msg)
            click.echo(f"Skipping: {msg}")

        data = valid
    else:
        data = [data]

    return data


def filter_inputs_affinity(
    manifest: Manifest,
    outdir: Path,
    override: bool = False,
    distogram_modes: tuple[str, ...] = ("control",),
    output_tag: Optional[str] = None,
    require_pre_affinity: bool = True,
) -> Manifest:
    """Check the input data and output directory for affinity.

    Parameters
    ----------
    manifest : Manifest
        The manifest.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    Manifest
        The manifest of the filtered input data.

    """
    from boltz.data.types import Manifest, get_affinity_output_id

    click.echo("Checking input data for affinity.")

    pred_dir = outdir / "predictions"

    def has_affinity_outputs(record):
        output_id = get_affinity_output_id(record)
        record_dir = pred_dir / record.id
        legacy_control = record_dir / f"affinity_{output_id}.json"
        return all(
            (
                record_dir
                / (
                    f"affinity_{output_id}__distogram_{mode}"
                    f"{'__run_' + output_tag if output_tag else ''}.json"
                )
            ).exists()
            or (mode == "control" and output_tag is None and legacy_control.exists())
            for mode in distogram_modes
        )

    existing = [
        record
        for record in manifest.records
        if record.affinity and has_affinity_outputs(record)
    ]

    # Remove them from the input data
    if existing and not override:
        existing_keys = {
            (record.id, get_affinity_output_id(record)) for record in existing
        }
        manifest = Manifest(
            [
                record
                for record in manifest.records
                if (record.id, get_affinity_output_id(record)) not in existing_keys
            ]
        )
        num_skipped = len(existing)
        msg = (
            f"Found some existing affinity predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "affinity predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = "Found existing affinity predictions, will override."
        click.echo(msg)

    # Skip records whose structure prediction failed (no pre_affinity npz)
    missing = (
        [
            r
            for r in manifest.records
            if r.affinity
            and not (pred_dir / r.id / f"pre_affinity_{r.id}.npz").exists()
        ]
        if require_pre_affinity
        else []
    )
    if missing:
        missing_ids = {r.id for r in missing}
        manifest = Manifest([r for r in manifest.records if r.id not in missing_ids])
        click.echo(
            f"Warning: {len(missing)} record(s) requested affinity but have no "
            f"structure prediction (pre_affinity_*.npz missing). Skipping: "
            f"{', '.join(r.id for r in missing)}"
        )

    return manifest


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.
    msa_server_username : str, optional
        Username for basic authentication with MSA server.
    msa_server_password : str, optional
        Password for basic authentication with MSA server.
    api_key_header : str, optional
        Custom header key for API key authentication (default: X-API-Key).
    api_key_value : str, optional
        Custom header value for API key authentication (overrides --api_key if set).

    """
    from boltz.data import const
    from boltz.data.msa.mmseqs2 import run_mmseqs2

    click.echo(f"Calling MSA server for target {target_id} with {len(data)} sequences")
    click.echo(f"MSA server URL: {msa_server_url}")
    click.echo(f"MSA pairing strategy: {msa_pairing_strategy}")

    # Construct auth headers if API key header/value is provided
    auth_headers = None
    if api_key_value:
        key = api_key_header if api_key_header else "X-API-Key"
        value = api_key_value
        auth_headers = {key: value}
        click.echo(f"Using API key authentication for MSA server (header: {key})")
    elif msa_server_username and msa_server_password:
        click.echo("Using basic authentication for MSA server")
    else:
        click.echo("No authentication provided for MSA server")

    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            auth_headers=auth_headers,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        auth_headers=auth_headers,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))


def _msa_content_sha256(path: Path) -> str:
    """Hash an MSA once per worker while detecting changes during the read."""
    resolved = path.resolve()
    before = resolved.stat()
    cache_key = (
        str(resolved),
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    cached = _MSA_HASH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    if (
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        msg = f"MSA file changed while it was being read: {resolved}"
        raise RuntimeError(msg)

    result = digest.hexdigest()
    _MSA_HASH_CACHE[cache_key] = result
    return result


def _processed_msa_id(msa_path: Path, max_msa_seqs: int) -> str:
    """Return a content-addressed ID for one processed MSA artifact."""
    source_digest = _msa_content_sha256(msa_path)
    provenance = (
        f"boltz-processed-msa-v{_PROCESSED_MSA_CACHE_VERSION}\0"
        f"{msa_path.suffix.lower()}\0{max_msa_seqs}\0{source_digest}"
    )
    return f"shared_{hashlib.sha256(provenance.encode()).hexdigest()}"


def _msa_lock_owner_is_running(lock_path: Path) -> bool:
    """Return whether a cache lock belongs to a live process on this host."""
    try:
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return True
    if owner.get("host") != socket.gethostname():
        return True
    try:
        os.kill(int(owner["pid"]), 0)
    except ProcessLookupError:
        return False
    except (OSError, ValueError, TypeError):
        return True
    return True


def _acquire_msa_cache_lock(lock_path: Path, processed_path: Path) -> int | None:
    """Acquire an inter-process cache lock, or return after another writer wins."""
    started = time.monotonic()
    payload = json.dumps({"pid": os.getpid(), "host": socket.gethostname()}).encode()
    while True:
        if processed_path.exists():
            return None
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not _msa_lock_owner_is_running(lock_path):
                with suppress(FileNotFoundError):
                    lock_path.unlink()
                continue
            if time.monotonic() - started >= _MSA_LOCK_TIMEOUT_SECONDS:
                msg = f"Timed out waiting for shared MSA cache: {processed_path}"
                raise TimeoutError(msg) from None
            time.sleep(_MSA_LOCK_POLL_SECONDS)
            continue
        try:
            os.write(descriptor, payload)
        except BaseException:
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)
            raise
        return descriptor


def _prepare_shared_msa(
    msa_path: Path,
    processed_msa_dir: Path,
    max_msa_seqs: int,
) -> str:
    """Parse an MSA into one atomic, content-addressed artifact."""
    from boltz.data.parse.a3m import parse_a3m  # noqa: PLC0415
    from boltz.data.parse.csv import parse_csv  # noqa: PLC0415

    msa_id = _processed_msa_id(msa_path, max_msa_seqs)
    processed = processed_msa_dir / f"{msa_id}.npz"
    if processed.exists():
        return msa_id

    lock_path = processed.with_suffix(".lock")
    descriptor = _acquire_msa_cache_lock(lock_path, processed)
    if descriptor is None:
        return msa_id

    temporary_path: Path | None = None
    try:
        if msa_path.suffix.lower() == ".a3m":
            msa = parse_a3m(msa_path, taxonomy=None, max_seqs=max_msa_seqs)
        elif msa_path.suffix.lower() == ".csv":
            msa = parse_csv(msa_path, max_seqs=max_msa_seqs)
        else:
            msg = f"MSA file {msa_path} not supported, only a3m or csv."
            raise RuntimeError(msg)

        with tempfile.NamedTemporaryFile(
            dir=processed_msa_dir,
            prefix=f".{msa_id}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        msa.dump(temporary_path)
        temporary_path.replace(processed)
        temporary_path = None
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return msa_id


def process_input(  # noqa: C901, PLR0912, PLR0915, D103
    path: Path,
    ccd: dict,
    msa_dir: Path,
    mol_dir: Path,
    boltz2: bool,
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str],
    msa_server_password: Optional[str],
    api_key_header: Optional[str],
    api_key_value: Optional[str],
    max_msa_seqs: int,
    processed_msa_dir: Path,
    processed_constraints_dir: Path,
    processed_templates_dir: Path,
    processed_mols_dir: Path,
    structure_dir: Path,
    records_dir: Path,
    skip_bad_inputs: bool = False,
    skip_ligand_conformer_generation: bool = False,
) -> Optional[str]:
    try:
        # Parse data
        if path.suffix.lower() in (".fa", ".fas", ".fasta"):
            from boltz.data.parse.fasta import parse_fasta

            target = parse_fasta(path, ccd, mol_dir, boltz2)
        elif path.suffix.lower() in (".yml", ".yaml"):
            from boltz.data.parse.yaml import parse_yaml

            target = parse_yaml(
                path,
                ccd,
                mol_dir,
                boltz2,
                skip_ligand_conformer_generation=(skip_ligand_conformer_generation),
            )
        elif path.is_dir():
            msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
            raise RuntimeError(msg)  # noqa: TRY301
        else:
            msg = (
                f"Unable to parse filetype {path.suffix}, "
                "please provide a .fasta or .yaml file."
            )
            raise RuntimeError(msg)  # noqa: TRY301

        from rdkit import Chem

        from boltz.data import const

        # Get target id
        target_id = target.record.id

        # Get all MSA ids and decide whether to generate MSA
        to_generate = {}
        prot_id = const.chain_type_ids["PROTEIN"]
        for chain in target.record.chains:
            # Add to generate list, assigning entity id
            if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                entity_id = chain.entity_id
                msa_id = f"{target_id}_{entity_id}"
                to_generate[msa_id] = target.sequences[entity_id]
                chain.msa_id = msa_dir / f"{msa_id}.csv"

            # We do not support msa generation for non-protein chains
            elif chain.msa_id == 0:
                chain.msa_id = -1

        # Generate MSA
        if to_generate and not use_msa_server:
            msg = "Missing MSA's in input and --use_msa_server flag not set."
            raise RuntimeError(msg)  # noqa: TRY301

        if to_generate:
            msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
            click.echo(msg)
            compute_msa(
                data=to_generate,
                target_id=target_id,
                msa_dir=msa_dir,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
                msa_server_username=msa_server_username,
                msa_server_password=msa_server_password,
                api_key_header=api_key_header,
                api_key_value=api_key_value,
            )

        # Parse MSA data
        msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
        msa_id_map = {}
        for msa_id in msas:
            # Check that raw MSA exists
            msa_path = Path(msa_id)
            if not msa_path.exists():
                msg = f"MSA file {msa_path} not found."
                raise FileNotFoundError(msg)  # noqa: TRY301

            msa_id_map[msa_id] = _prepare_shared_msa(
                msa_path,
                processed_msa_dir,
                max_msa_seqs,
            )

        # Modify records to point to processed MSA
        for c in target.record.chains:
            if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                c.msa_id = msa_id_map[c.msa_id]

        # Dump templates
        for template_id, template in target.templates.items():
            name = f"{target.record.id}_{template_id}.npz"
            template_path = processed_templates_dir / name
            template.dump(template_path)

        # Dump constraints
        constraints_path = processed_constraints_dir / f"{target.record.id}.npz"
        target.residue_constraints.dump(constraints_path)

        # Dump extra molecules
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        with (processed_mols_dir / f"{target.record.id}.pkl").open("wb") as f:
            pickle.dump(target.extra_mols, f)

        # Dump structure
        struct_path = structure_dir / f"{target.record.id}.npz"
        target.structure.dump(struct_path)

        # Dump record
        record_path = records_dir / f"{target.record.id}.json"
        target.record.dump(record_path)

        return None

    except Exception as e:  # noqa: BLE001
        if not skip_bad_inputs:
            raise
        return f"{path.name}: {e}"


def _initialize_process_input_worker(
    worker: Callable[[Path], Optional[str]],
) -> None:
    """Retain heavyweight shared preprocessing state once per worker."""
    global _PROCESS_INPUT_WORKER  # noqa: PLW0603
    _PROCESS_INPUT_WORKER = worker


def _run_process_input_worker(path: Path) -> Optional[str]:
    """Process one path using worker-local shared receptor and CCD state."""
    if _PROCESS_INPUT_WORKER is None:
        msg = "Preprocessing worker was not initialized."
        raise RuntimeError(msg)
    return _PROCESS_INPUT_WORKER(path)


def process_inputs(
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    mol_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 8192,
    use_msa_server: bool = False,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
    boltz2: bool = False,
    preprocessing_threads: int = 1,
    skip_bad_inputs: bool = False,
    skip_ligand_conformer_generation: bool = False,
) -> Manifest:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 8192.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.
    msa_server_username : str, optional
        Username for basic authentication with MSA server, by default None.
    msa_server_password : str, optional
        Password for basic authentication with MSA server, by default None.
    api_key_header : str, optional
        Custom header key for API key authentication (default: X-API-Key).
    api_key_value : str, optional
        Custom header value for API key authentication (overrides --api_key if set).
    boltz2: bool, optional
        Whether to use Boltz2, by default False.
    preprocessing_threads: int, optional
        The number of threads to use for preprocessing, by default 1.

    Returns
    -------
    Manifest
        The manifest of the processed input data.

    """
    from tqdm import tqdm

    from boltz.data.mol import load_canonicals
    from boltz.data.types import Manifest, Record

    # Validate mutually exclusive authentication methods
    has_basic_auth = msa_server_username and msa_server_password
    has_api_key = api_key_value is not None

    if has_basic_auth and has_api_key:
        raise ValueError(
            "Cannot use both basic authentication (--msa_server_username/--msa_server_password) "
            "and API key authentication (--api_key_header/--api_key_value). Please use only one authentication method."
        )

    # Check if records exist at output path
    records_dir = out_dir / "processed" / "records"
    if records_dir.exists():
        # Load existing records
        existing = [Record.load(p) for p in sorted(records_dir.glob("*.json"))]
        processed_ids = {record.id for record in existing}

        # Filter to missing only
        data = [d for d in data if d.stem not in processed_ids]

        # Nothing to do, update the manifest and return
        if data:
            click.echo(
                f"Found {len(existing)} existing processed inputs, skipping them."
            )
        else:
            click.echo("All inputs are already processed.")
            updated_manifest = Manifest(existing)
            updated_manifest.dump(out_dir / "processed" / "manifest.json")
            return updated_manifest

    # Create output directories
    msa_dir = out_dir / "msa"
    records_dir = out_dir / "processed" / "records"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    processed_constraints_dir = out_dir / "processed" / "constraints"
    processed_templates_dir = out_dir / "processed" / "templates"
    processed_mols_dir = out_dir / "processed" / "mols"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    processed_constraints_dir.mkdir(parents=True, exist_ok=True)
    processed_templates_dir.mkdir(parents=True, exist_ok=True)
    processed_mols_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    if boltz2:
        ccd = load_canonicals(mol_dir)
    else:
        with ccd_path.open("rb") as file:
            ccd = pickle.load(file)  # noqa: S301

    # Create partial function
    process_input_partial = partial(
        process_input,
        ccd=ccd,
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        boltz2=boltz2,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        max_msa_seqs=max_msa_seqs,
        skip_ligand_conformer_generation=skip_ligand_conformer_generation,
        processed_msa_dir=processed_msa_dir,
        processed_constraints_dir=processed_constraints_dir,
        processed_templates_dir=processed_templates_dir,
        processed_mols_dir=processed_mols_dir,
        structure_dir=structure_dir,
        records_dir=records_dir,
        skip_bad_inputs=skip_bad_inputs,
    )

    # Parse input data
    preprocessing_threads = min(preprocessing_threads, len(data))
    click.echo(f"Processing {len(data)} inputs with {preprocessing_threads} threads.")

    if preprocessing_threads > 1 and len(data) > 1:
        # Process one representative before forking. Besides creating the shared
        # MSA atomically, this primes the immutable receptor cache inherited by
        # fork-based workers used for large Linux/WSL screens.
        errors = [process_input_partial(data[0])]
        remaining = data[1:]
        with Pool(
            preprocessing_threads,
            initializer=_initialize_process_input_worker,
            initargs=(process_input_partial,),
        ) as pool:
            errors.extend(
                tqdm(
                    pool.imap(_run_process_input_worker, remaining),
                    total=len(remaining),
                )
            )
    else:
        errors = []
        for path in tqdm(data):
            errors.append(process_input_partial(path))

    # Report failures
    failures = [e for e in errors if e is not None]
    if failures:
        click.echo(f"\n{len(failures)} input(s) failed:")
        for msg in failures:
            click.echo(f"  {msg}")

    # Load all records and write manifest
    records = [Record.load(p) for p in sorted(records_dir.glob("*.json"))]
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")
    return manifest


@click.group()
@click.version_option(version=_runtime_version(), prog_name="boltz-stfree")
def cli() -> None:
    """Boltz-StFree structure-free affinity screening runtime."""
    return


@cli.command(context_settings={"show_default": True})
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--prepare-only",
    is_flag=True,
    help="Validate the config and generate chunked Boltz inputs without inference.",
)
def screen(config: Path, prepare_only: bool) -> None:
    """Screen one protein target against a ligand library."""
    from boltz.screen import (
        ScreenProgress,
        collect_screen_scores,
        ensure_auto_msa,
        format_screen_plan,
        invalidate_screen_aggregates,
        prepare_screen,
    )
    from boltz.stfree_runtime import (
        create_stfree_runtime_session,
        prepare_stfree_inputs,
        run_stfree_inputs,
    )

    prepared = prepare_screen(
        config,
        ensure_ccd_cache=None if prepare_only else download_boltz2_ccd,
    )
    screen_options = prepared.screen_options
    click.echo(
        f"Prepared {len(prepared.chunk_dirs)} chunk(s) in {prepared.output_dir}."
    )
    plan = json.loads(prepared.plan_path.read_text(encoding="utf-8"))
    click.echo(f"Screen plan: {prepared.plan_path}")
    for line in format_screen_plan(plan):
        click.echo(f"  {line}")
    if prepare_only:
        ScreenProgress(prepared)
        click.echo(f"Manifest: {prepared.manifest_path}")
        return

    if screen_options.runtime.override:
        invalidate_screen_aggregates(prepared)
    progress = ScreenProgress(
        prepared,
        ignore_existing=screen_options.runtime.override,
    )
    click.echo(f"Live status: {prepared.status_path}")
    try:
        if prepared.auto_msa_path is not None:
            progress.set_stage("preparing_msa", "Preparing shared target MSA")
            click.echo("Preparing one shared target MSA for all screen chunks.")
            ensure_auto_msa(
                prepared,
                screen_options=screen_options,
                compute_msa_fn=compute_msa,
            )

        input_paths = [
            input_path
            for chunk_dir in prepared.chunk_dirs
            for input_path in sorted(chunk_dir.glob("*.yaml"))
        ]
        runtime_dir = prepared.runs_dir / "boltz_results_screen"
        progress.set_stage("preprocessing_inputs", "Preprocessing all screen inputs")
        click.echo(f"Preprocessing all {len(input_paths)} screen inputs in one pass.")
        prepare_stfree_inputs(
            input_paths=input_paths,
            run_dir=runtime_dir,
            options=screen_options.runtime,
        )

        click.echo("Loading one StFree model for the complete screen.")
        runtime_session = create_stfree_runtime_session(screen_options.runtime)
        progress.set_stage("running", "Running complete screen")
        click.echo(f"Running one unified screen for {len(input_paths)} inputs.")
        run_stfree_inputs(
            input_paths=input_paths,
            run_dir=runtime_dir,
            options=screen_options.runtime,
            receptor_crop_indices_dir=prepared.receptor_crop_indices_dir,
            progress_callback=progress.record_batch,
            session=runtime_session,
            preprocess=False,
        )
        scores_path = collect_screen_scores(prepared)
        progress.complete(scores_path)
        click.echo(f"Screen complete: {scores_path}")
    except BaseException as error:
        progress.fail(error)
        raise


@cli.command(name="bootstrap-reference", context_settings={"show_default": True})
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--prepare-only",
    is_flag=True,
    help="Validate and stage the holo input without loading a model or using a GPU.",
)
def bootstrap_reference(config: Path, prepare_only: bool) -> None:
    """Create one reusable crop reference from a target and anchor ligand."""
    from boltz.reference_bootstrap import (
        format_bootstrap_plan,
        prepare_bootstrap_reference,
        run_bootstrap_reference,
    )

    prepared = prepare_bootstrap_reference(config)
    plan = json.loads(prepared.plan_path.read_text(encoding="utf-8"))
    click.echo(f"Bootstrap plan: {prepared.plan_path}")
    for line in format_bootstrap_plan(plan):
        click.echo(f"  {line}")
    if prepare_only:
        click.echo("Prepared only; no model was loaded and no GPU work was started.")
        click.echo(f"Manifest: {prepared.manifest_path}")
        return
    reference = run_bootstrap_reference(prepared)
    click.echo(f"Reference complete: {reference}")


if __name__ == "__main__":
    cli()

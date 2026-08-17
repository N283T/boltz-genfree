"""Tests for the shared preprocessing and CLI support used by ``screen``."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import click
import numpy as np
import pytest

import boltz.data.parse.schema as schema_module
import boltz.main as main_module
from boltz.data.parse.csv import parse_csv
from boltz.data.types import MSA, ChainInfo, Record, StructureInfo
from boltz.main import (
    _available_cpu_count,
    _configure_wsl_multiprocessing_tempdir,
    _prepare_shared_msa,
    _resolved_input_sources,
    _validate_checkpoint_for_load,
    check_inputs,
    process_input,
    process_inputs,
)


def _write_test_msa(path: Path) -> None:
    path.write_text(
        "key,sequence\n-1,QLEDSEVE\n9606,QLEDSEVE\n-1,QLEDSEVA\n",
        encoding="utf-8",
    )


def test_standard_receptor_polymer_is_parsed_once_per_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated standard receptors should reuse their immutable parsed chain."""
    schema_module._POLYMER_PARSE_CACHE.clear()  # noqa: SLF001
    parsed_chain = object()
    calls = 0

    def fake_parse_polymer(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return parsed_chain

    monkeypatch.setattr(schema_module, "parse_polymer", fake_parse_polymer)
    components = {}
    arguments = {
        "sequence": ["ALA", "GLY"],
        "raw_sequence": "AG",
        "entity": "0",
        "chain_type": 0,
        "components": components,
        "cyclic": False,
        "mol_dir": tmp_path,
    }

    first = schema_module._parse_polymer_cached(**arguments)  # noqa: SLF001
    second = schema_module._parse_polymer_cached(**arguments)  # noqa: SLF001

    assert first is parsed_chain
    assert second is parsed_chain
    assert calls == 1
    schema_module._POLYMER_PARSE_CACHE.clear()  # noqa: SLF001


def test_process_inputs_resume_preserves_legacy_msa_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing records and their per-record MSA artifacts remain resumable."""
    input_path = tmp_path / "legacy.yaml"
    input_path.write_text("not read during resume", encoding="utf-8")
    output = tmp_path / "output"
    records_dir = output / "processed" / "records"
    msa_dir = output / "processed" / "msa"
    records_dir.mkdir(parents=True)
    msa_dir.mkdir()
    legacy_msa = msa_dir / "legacy_0.npz"
    legacy_contents = b"legacy-msa-remains-untouched"
    legacy_msa.write_bytes(legacy_contents)
    record = Record(
        id="legacy",
        structure=StructureInfo(num_chains=1),
        chains=[
            ChainInfo(
                chain_id=0,
                chain_name="A",
                mol_type=0,
                cluster_id=-1,
                msa_id="legacy_0",
                num_residues=2,
                entity_id=0,
            )
        ],
        interfaces=[],
    )
    record.dump(records_dir / "legacy.json")
    monkeypatch.setattr("boltz.data.mol.load_canonicals", lambda _: {})

    manifest = process_inputs(
        data=[input_path],
        out_dir=output,
        ccd_path=tmp_path / "unused.pkl",
        mol_dir=tmp_path / "mols",
        msa_server_url="",
        msa_pairing_strategy="greedy",
        boltz2=True,
    )

    assert [item.id for item in manifest.records] == ["legacy"]
    assert manifest.records[0].chains[0].msa_id == "legacy_0"
    assert legacy_msa.read_bytes() == legacy_contents


def test_processed_msa_cache_is_content_addressed_and_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal MSA content should produce one byte-identical processed artifact."""
    first_source = tmp_path / "first.csv"
    second_source = tmp_path / "second.csv"
    _write_test_msa(first_source)
    _write_test_msa(second_source)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    main_module._MSA_HASH_CACHE.clear()  # noqa: SLF001

    calls = 0

    def counted_parse(path: Path, max_seqs: int) -> MSA:
        nonlocal calls
        calls += 1
        return parse_csv(path, max_seqs=max_seqs)

    monkeypatch.setattr("boltz.data.parse.csv.parse_csv", counted_parse)
    first_id = _prepare_shared_msa(first_source, processed_dir, max_msa_seqs=2)
    second_id = _prepare_shared_msa(second_source, processed_dir, max_msa_seqs=2)

    assert first_id == second_id
    assert calls == 1
    expected = parse_csv(first_source, max_seqs=2)
    actual = MSA.load(processed_dir / f"{first_id}.npz")
    np.testing.assert_array_equal(actual.sequences, expected.sequences)
    np.testing.assert_array_equal(actual.deletions, expected.deletions)
    np.testing.assert_array_equal(actual.residues, expected.residues)


def test_processed_msa_cache_includes_max_sequences(
    tmp_path: Path,
) -> None:
    """Changing a parse limit must not reuse an incompatible artifact."""
    source = tmp_path / "target.csv"
    _write_test_msa(source)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    main_module._MSA_HASH_CACHE.clear()  # noqa: SLF001

    one_sequence = _prepare_shared_msa(source, processed_dir, max_msa_seqs=1)
    two_sequences = _prepare_shared_msa(source, processed_dir, max_msa_seqs=2)

    assert one_sequence != two_sequences
    assert {one_sequence, two_sequences} == {
        path.stem for path in processed_dir.glob("shared_*.npz")
    }


def test_processed_msa_cache_parses_once_under_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent workers should elect exactly one MSA parser."""
    source = tmp_path / "target.csv"
    _write_test_msa(source)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    main_module._MSA_HASH_CACHE.clear()  # noqa: SLF001
    counter_lock = threading.Lock()
    calls = 0

    def slow_counted_parse(path: Path, max_seqs: int) -> MSA:
        nonlocal calls
        with counter_lock:
            calls += 1
        time.sleep(0.1)
        return parse_csv(path, max_seqs=max_seqs)

    monkeypatch.setattr("boltz.data.parse.csv.parse_csv", slow_counted_parse)
    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = list(
            executor.map(
                lambda _: _prepare_shared_msa(
                    source,
                    processed_dir,
                    max_msa_seqs=2,
                ),
                range(16),
            )
        )

    assert len(set(ids)) == 1
    assert calls == 1
    assert not list(processed_dir.glob("*.lock"))
    assert len(list(processed_dir.glob("shared_*.npz"))) == 1


def test_processed_msa_cache_recovers_dead_worker_lock(tmp_path: Path) -> None:
    """Resume should remove a lock whose local owner process no longer exists."""
    source = tmp_path / "target.csv"
    _write_test_msa(source)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    main_module._MSA_HASH_CACHE.clear()  # noqa: SLF001
    msa_id = main_module._processed_msa_id(source, max_msa_seqs=2)  # noqa: SLF001
    lock_path = processed_dir / f"{msa_id}.lock"
    lock_path.write_text(
        json.dumps({"pid": 2**31 - 1, "host": socket.gethostname()}),
        encoding="utf-8",
    )

    result = _prepare_shared_msa(source, processed_dir, max_msa_seqs=2)

    assert result == msa_id
    assert (processed_dir / f"{msa_id}.npz").is_file()
    assert not lock_path.exists()


def test_wsl_multiprocessing_tempdir_moves_off_windows_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("boltz.main.platform.release", lambda: "microsoft-WSL2")
    monkeypatch.setattr(
        "boltz.main.tempfile.gettempdir",
        lambda: "/mnt/c/Users/test/AppData/Local/Temp",
    )
    monkeypatch.setattr("boltz.main.tempfile.tempdir", None)
    monkeypatch.delenv("TMPDIR", raising=False)

    result = _configure_wsl_multiprocessing_tempdir(2)

    assert result == Path("/tmp")
    assert os.environ["TMPDIR"] == "/tmp"
    assert __import__("boltz.main", fromlist=["tempfile"]).tempfile.tempdir == "/tmp"


def test_wsl_multiprocessing_tempdir_ignores_single_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("boltz.main.platform.release", lambda: "microsoft-WSL2")

    assert _configure_wsl_multiprocessing_tempdir(0) is None


def test_resolved_input_sources_accepts_expanded_input_list(tmp_path: Path) -> None:
    inputs = [tmp_path / "a.yaml", tmp_path / "b.yaml"]

    assert _resolved_input_sources(inputs) == [str(path.resolve()) for path in inputs]


class TestAvailableCpuCount:
    """CPU discovery must respect task and container affinity."""

    def test_returns_positive_int(self) -> None:
        result = _available_cpu_count()

        assert isinstance(result, int)
        assert result >= 1

    def test_falls_back_when_affinity_is_unavailable(self) -> None:
        if hasattr(os, "sched_getaffinity"):
            with patch("os.sched_getaffinity", side_effect=OSError("mocked")):
                result = _available_cpu_count()
        else:
            result = _available_cpu_count()

        assert isinstance(result, int)
        assert result >= 1


class TestCheckpointValidation:
    """Checkpoint validation should fail early with clear errors."""

    def test_rejects_empty_checkpoint_before_load(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "boltz2_aff.ckpt"
        checkpoint.write_bytes(b"")

        with pytest.raises(click.ClickException, match="checkpoint is empty"):
            _validate_checkpoint_for_load(checkpoint, "Boltz-2 affinity weights")

    def test_accepts_non_empty_checkpoint(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "boltz2_aff.ckpt"
        checkpoint.write_bytes(b"not-empty")

        _validate_checkpoint_for_load(checkpoint, "Boltz-2 affinity weights")


class TestCheckInputs:
    """Input discovery is shared by every StFree chunk."""

    def test_single_file_returned_as_list(self, tmp_path: Path) -> None:
        input_path = tmp_path / "input.yaml"
        input_path.write_text("dummy")

        assert check_inputs(input_path) == [input_path]

    def test_directory_filters_valid_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "b.fasta").write_text("f")
        (tmp_path / "a.yaml").write_text("y")

        names = [path.name for path in check_inputs(tmp_path)]

        assert names == ["a.yaml", "b.fasta"]

    def test_directory_aborts_on_bad_extension_by_default(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "good.yaml").write_text("y")
        (tmp_path / "bad.txt").write_text("t")

        with pytest.raises(RuntimeError, match="Unable to parse filetype"):
            check_inputs(tmp_path)

    def test_skip_bad_inputs_filters_invalid_entries(self, tmp_path: Path) -> None:
        (tmp_path / "good.fasta").write_text("f")
        (tmp_path / "bad.txt").write_text("t")
        (tmp_path / "subdir").mkdir()

        result = check_inputs(tmp_path, skip_bad_inputs=True)

        assert [path.name for path in result] == ["good.fasta"]


class TestProcessInputSkipBadInputs:
    """Malformed per-record input may be reported instead of aborting a library."""

    @staticmethod
    def _call(path: Path, *, skip_bad_inputs: bool) -> str | None:
        return process_input(
            path=path,
            ccd={},
            msa_dir=Path("/unused"),
            mol_dir=Path("/unused"),
            boltz2=True,
            use_msa_server=False,
            msa_server_url="",
            msa_pairing_strategy="greedy",
            msa_server_username=None,
            msa_server_password=None,
            api_key_header=None,
            api_key_value=None,
            max_msa_seqs=1,
            processed_msa_dir=Path("/unused"),
            processed_constraints_dir=Path("/unused"),
            processed_templates_dir=Path("/unused"),
            processed_mols_dir=Path("/unused"),
            structure_dir=Path("/unused"),
            records_dir=Path("/unused"),
            skip_bad_inputs=skip_bad_inputs,
        )

    def test_bad_input_raises_by_default(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("not valid yaml: [")

        with pytest.raises(Exception):
            self._call(bad, skip_bad_inputs=False)

    def test_bad_input_returns_error_with_skip(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("not valid yaml: [")

        result = self._call(bad, skip_bad_inputs=True)

        assert result is not None
        assert "bad.yaml" in result

    def test_unsupported_extension_returns_error_with_skip(
        self,
        tmp_path: Path,
    ) -> None:
        bad = tmp_path / "data.txt"
        bad.write_text("hello")

        result = self._call(bad, skip_bad_inputs=True)

        assert result is not None
        assert "data.txt" in result

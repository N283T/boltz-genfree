"""Tests for the shared preprocessing and CLI support used by ``screen``."""

import os
from pathlib import Path
from unittest.mock import patch

import click
import pytest

from boltz.main import (
    _available_cpu_count,
    _configure_wsl_multiprocessing_tempdir,
    _resolved_input_sources,
    _validate_checkpoint_for_load,
    check_inputs,
    process_input,
)


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

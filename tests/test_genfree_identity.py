"""Public naming changes must not alter scientific identity or safety gates."""
from importlib.metadata import entry_points, version

import pytest
from click.testing import CliRunner

import boltz.main as main
from boltz.seed import record_seed
from boltz.stfree_runtime import StFreeRuntimeOptions, _run_provenance


def test_new_and_legacy_commands_share_the_same_cli():
    scripts = {p.name: p.value for p in entry_points(group="console_scripts")}
    assert scripts["boltz-genfree"] == scripts["boltz-stfree"] == "boltz.main:cli"
    assert scripts["boltz-genfree-fix-macos-libomp"] == scripts["boltz-stfree-fix-macos-libomp"]
    result = CliRunner().invoke(main.cli, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"boltz-genfree, version {version('boltz-genfree')}"


@pytest.mark.parametrize("available", ["boltz-genfree", "boltz-stfree", "boltz-community", "boltz", None])
def test_version_prefers_new_distribution_with_legacy_fallback(monkeypatch, available):
    calls = []

    def fake_version(name):
        calls.append(name)
        if name == available:
            return "test-version"
        raise main.PackageNotFoundError(name)

    monkeypatch.setattr(main, "_pkg_version", fake_version)
    assert main._runtime_version() == ("test-version" if available else "0+unknown")
    expected = ["boltz-genfree", "boltz-stfree", "boltz-community", "boltz"]
    assert calls == (expected[:expected.index(available)+1] if available else expected)


def test_legacy_record_seed_namespace_and_runtime_identity_are_retained(tmp_path):
    import hashlib

    payload = "\0".join(["boltz-stfree-record-seed-v1", "msa", "17", "record-a"]).encode()
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
    assert record_seed(17, "record-a", purpose="msa") == expected
    provenance = _run_provenance(StFreeRuntimeOptions(cache=tmp_path),
        runtime_version="test-version", checkpoint_sha256="f"*64)
    assert provenance["runtime"] == "boltz-stfree"
    assert provenance["runtime_code_sha256"]
    assert provenance["method"]["distogram_mode"] == "zero_distogram_features"

"""Write named float32 tensors as a tensor bundle (manifest layout modelled on LocalFold's).

Layout: ``<name>.json`` holds ``{"formatVersion": 1, "tensors": {...}}`` and
``<name>.bin`` holds the raw little-endian float32 data. Each tensor record is
``{"file", "byteOffset", "dtype": "float32", "shape"}``, which is what
LocalFold's ``FileTensorStore`` (shared/bundles/tensor-store.js) reads and what
``webgpu/src/bundle.js`` reads here. Integer and boolean tensors are stored as
float32 so that one reader covers everything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def write_bundle(
    directory: Path,
    name: str,
    tensors: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write ``tensors`` to ``directory/name.{json,bin}``; return the manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    offset = 0
    with (directory / f"{name}.bin").open("wb") as handle:
        for key, value in tensors.items():
            array = _to_numpy(value)
            handle.write(array.tobytes())
            records[key] = {
                "file": f"{name}.bin",
                "byteOffset": offset,
                "dtype": "float32",
                "shape": list(array.shape),
            }
            offset += array.nbytes
    manifest = {"formatVersion": 1, "meta": meta or {}, "tensors": records}
    path = directory / f"{name}.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path

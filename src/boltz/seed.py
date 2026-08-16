"""Stable per-record seeds for recoverable screening."""

from __future__ import annotations

import hashlib


def record_seed(base_seed: int, record_id: str, *, purpose: str) -> int:
    """Derive a stable signed-64-bit seed from a screen record identity."""
    payload = (
        f"boltz-stfree-record-seed-v1\0{purpose}\0{base_seed}\0{record_id}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )

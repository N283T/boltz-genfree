"""Convert Boltz-2 affinity checkpoint tensors to a WebGPU weight pack.

Reads ``affinity_module1.*`` and ``affinity_module2.*`` from the checkpoint and
writes a LocalFold-compatible tensor bundle (see ``tensor_bundle.py``). Tensor
names use ``/`` separators and a ``affinity1`` / ``affinity2`` root, for example
``affinity1/pairformer_stack/layers/0/tri_mul_out/p_in/weight``. Shapes are kept
in the PyTorch ``(out, in)`` layout; the JS side reads them as such.

The pack carries the module configuration in ``meta`` so the JS loader needs no
Python-side hyper-parameters. Weights stay float32 (8.7M parameters, ~35 MB).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch
from tensor_bundle import write_bundle

PREFIXES = {"affinity_module1.": "affinity1", "affinity_module2.": "affinity2"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_config(state: dict[str, torch.Tensor], root: str, hparams: dict[str, Any]):
    """Derive widths from tensor shapes; take block counts from the checkpoint."""
    prefix = f"{root}."
    embed = state[f"{prefix}dist_bin_pairwise_embed.weight"]
    s_in = state[f"{prefix}s_to_z_prod_in1.weight"]
    heads_out = state[f"{prefix}affinity_heads.affinity_out_mlp.2.weight"]
    tri_bias = state[f"{prefix}pairformer_stack.layers.0.tri_att_start.linear.weight"]
    q = state[f"{prefix}pairformer_stack.layers.0.tri_att_start.mha.linear_q.weight"]
    layers = {
        int(k.split(".")[3])
        for k in state
        if k.startswith(f"{prefix}pairformer_stack.layers.")
    }
    num_heads = int(tri_bias.shape[0])
    return {
        "numDistBins": int(embed.shape[0]),
        "tokenZ": int(embed.shape[1]),
        "tokenS": int(s_in.shape[1]),
        "headsTokenS": int(heads_out.shape[0]),
        "pairformerBlocks": len(layers),
        "triHeads": num_heads,
        "triHeadWidth": int(q.shape[0]) // num_heads,
        "conditionerTransitions": len(
            {
                int(k.split(".")[3])
                for k in state
                if k.startswith(f"{prefix}pairwise_conditioner.transitions.")
            }
        ),
        "boundaries": [float(x) for x in state[f"{prefix}boundaries"]],
    }


def convert(checkpoint: Path, out_dir: Path, name: str = "genfree-affinity") -> Path:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    hparams = ckpt.get("hyper_parameters", {})
    tensors: dict[str, torch.Tensor] = {}
    config: dict[str, Any] = {}
    for prefix, root in PREFIXES.items():
        found = False
        for key, value in state.items():
            if key.startswith(prefix):
                tensors[root + "/" + key[len(prefix) :].replace(".", "/")] = value
                found = True
        if not found:
            raise SystemExit(f"checkpoint has no tensors with prefix {prefix!r}")
        config[root] = module_config(state, prefix[:-1], hparams)
    meta = {
        "kind": "boltz-genfree-affinity-pack",
        "source": checkpoint.name,
        "sourceSha256": sha256_of(checkpoint),
        "affinityMwCorrection": bool(hparams.get("affinity_mw_correction", False)),
        "modules": config,
    }
    return write_bundle(out_dir, name, tensors, meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path.home() / ".boltz" / "boltz2_aff.ckpt"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("webgpu/packs"))
    args = parser.parse_args()
    print(convert(args.checkpoint, args.out_dir))


if __name__ == "__main__":
    main()

"""Convert the Boltz-2 trunk in ``boltz2_aff.ckpt`` to a float32 pack with PyTorch names.

Unlike ``convert_trunk_weights.py`` (LocalFold's haiku-style layout), this keeps the
original PyTorch parameter names (only ``.`` becomes ``/``) and layouts: no transposes,
no interleaving, no zeroed or swapped rows. Linear weights stay ``(out, in)``.

Included: everything needed to run Boltz-2 up to the affinity module input
(``input_embedder`` incl. atom encoders, ``s_init``, ``z_init_1/2``, ``rel_pos``,
``token_bonds``, ``token_bonds_type``, ``contact_conditioning``, ``s_norm``/``z_norm``,
``s_recycle``/``z_recycle``, ``msa_module``, ``pairformer_module``).

Excluded: ``template_module`` (not in this checkpoint), ``distogram_module``,
``confidence_module``, ``structure_module``, ``diffusion_conditioning`` and the
``affinity_module*`` weights (see ``convert_weights.py``).

Architecture dimensions are written to the manifest ``meta`` so loaders need no
hard-coded numbers.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch
from tensor_bundle import write_bundle

PREFIXES = (
    "input_embedder.",
    "s_init.",
    "z_init_1.",
    "z_init_2.",
    "rel_pos.",
    "token_bonds.",
    "token_bonds_type.",
    "contact_conditioning.",
    "s_norm.",
    "z_norm.",
    "s_recycle.",
    "z_recycle.",
    "msa_module.",
    "pairformer_module.",
)
EXCLUDED = (
    "template_module.",
    "distogram_module.",
    "confidence_module.",
    "structure_module.",
    "diffusion_conditioning.",
    "affinity_module",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_layers(state: dict[str, torch.Tensor], prefix: str) -> int:
    ids = {int(k[len(prefix) :].split(".")[0]) for k in state if k.startswith(prefix)}
    return len(ids)


def architecture(state: dict[str, torch.Tensor], hparams: dict[str, Any]):
    msa_args = hparams.get("msa_args", {})
    pf_args = hparams.get("pairformer_args", {})
    embed = hparams.get("embedder_args", {})
    atom_prefix = "input_embedder.atom_attention_encoder.atom_encoder.diffusion_transformer.layers."
    return {
        "tokenS": int(state["s_init.weight"].shape[0]),
        "tokenZ": int(state["z_init_1.weight"].shape[0]),
        "msaS": int(state["msa_module.s_proj.weight"].shape[0]),
        "msaFeatures": int(state["msa_module.msa_proj.weight"].shape[1]),
        "msaBlocks": count_layers(state, "msa_module.layers."),
        "pairformerBlocks": count_layers(state, "pairformer_module.layers."),
        "pairformerHeads": int(pf_args.get("num_heads", 16)),
        "msaPairwiseHeads": int(msa_args.get("pairwise_num_heads", 4)),
        "msaPairwiseHeadWidth": int(msa_args.get("pairwise_head_width", 32)),
        "atomS": int(
            state["input_embedder.atom_encoder.embed_atom_features.weight"].shape[0]
        ),
        "atomFeatures": int(
            state["input_embedder.atom_encoder.embed_atom_features.weight"].shape[1]
        ),
        "atomEncoderBlocks": count_layers(state, atom_prefix),
        "atomEncoderHeads": int(embed.get("atom_encoder_heads", 4)),
        "numMethods": int(
            state["input_embedder.method_conditioning_init.weight"].shape[0]
        ),
        "numTokenTypes": int(state["input_embedder.res_type_encoding.weight"].shape[1]),
        "numBondTypes": int(state["token_bonds_type.weight"].shape[0]),
        "relPosFeatures": int(state["rel_pos.linear_layer.weight"].shape[1]),
    }


def convert(checkpoint: Path, out_dir: Path, name: str = "boltz2-trunk-own") -> Path:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    hparams = ckpt.get("hyper_parameters", {})
    tensors: dict[str, torch.Tensor] = {}
    unclassified: list[str] = []
    for key, value in state.items():
        if key.startswith(PREFIXES):
            tensors[key.replace(".", "/")] = value
        elif not key.startswith(EXCLUDED):
            unclassified.append(key)
    if unclassified:
        print(
            f"warning: {len(unclassified)} unclassified tensors, e.g. {unclassified[:5]}"
        )
    for prefix in PREFIXES:
        if not any(k.startswith(prefix.replace(".", "/")) for k in tensors):
            raise SystemExit(f"checkpoint has no tensors with prefix {prefix!r}")
    meta = {
        "kind": "boltz-trunk-own-pack",
        "model": "boltz2",
        "source": checkpoint.name,
        "sourceSha256": sha256_of(checkpoint),
        "architecture": architecture(state, hparams),
    }
    return write_bundle(out_dir, name, tensors, meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path.home() / ".boltz" / "boltz2_aff.ckpt"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("webgpu/packs/boltz2-trunk-own")
    )
    args = parser.parse_args()
    print(convert(args.checkpoint, args.out_dir))


if __name__ == "__main__":
    main()

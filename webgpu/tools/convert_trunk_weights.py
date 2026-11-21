"""Convert the Boltz-2 trunk in ``boltz2_aff.ckpt`` to LocalFold's boltz2 tensor layout.

GenFree runs the trunk that ships inside ``boltz2_aff.ckpt`` (``Boltz2.load_from_checkpoint``
loads the whole model from that one file). LocalFold's WebGPU trunk reads Boltz-2 weights in
the AF3-lineage layout of its ``boltz2`` bundle (haiku-style names, layer stacks with the
block as leading axis, fused and interleaved projections). This script writes the same
tensors, float32, straight from the checkpoint, under the names LocalFold's loaders
(``shared/af3/weights/``) read. The mapping was derived by matching tensor values against
LocalFold's int5 bundle; ``compare_trunk_pack.mjs`` re-runs that check.

Layout rules (all verified against the int5 bundle, relRMS ~0.04 = int5 round-off):

- ``nn.Linear`` weights are ``(out, in)`` in PyTorch and ``(in, out)`` here (transposed).
- Triangle q/k projections and the triangle-attention gate keep ``(out, in)``;
  q/k are split ``(heads, head_dim, in)``, v is ``(in, heads, head_dim)``.
- Triangle multiplication ``projection`` / ``gate`` interleave the two halves of
  ``p_in`` / ``g_in``: column ``2k`` is row ``k`` and column ``2k+1`` is row ``128+k``
  (outgoing); the incoming product has the two halves swapped.
- The distogram head's ``half_logits`` bias is half of the PyTorch bias (the head sums the
  logits with their transpose).
- Transition ``transition1`` is ``cat([fc1.T, fc2.T], axis=1)``.
- Residue-type axes: LocalFold's 31 classes are Boltz's 33 tokens without ``<pad>`` and
  ``DN``; the DNA unknown is carried as ``boltz2_dn_*`` = column ``DN`` minus column ``N``.
- The atom encoder's ``embed_pair_offsets`` is the negated ``embed_atompair_ref_pos``.
- The conditioned transition's ``swish_gate`` halves are swapped.

The template module is not part of this checkpoint (``use_templates: False``), so its
tensors are written as zeros (norm scales as ones). Its ``u_proj`` is zero, so the template
term LocalFold adds for an input without templates is exactly zero, as in GenFree.

Only the trunk is converted (embedder, atom encoder, MSA module, pairformer, distogram head).
The diffusion head, the confidence head and the structure module are not used by the
structure-free route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

TOKEN_COLUMN = [
    *range(2, 22),
    22,
    1,
    23,
    24,
    25,
    26,
    28,
    29,
    30,
    31,
    27,
]  # LocalFold class -> Boltz token
BOLTZ_GAP, BOLTZ_N, BOLTZ_DN = 1, 27, 32
SHARD_BYTES = 256 * 1024 * 1024
ROOT = "diffuser/"


def interleave(weight: np.ndarray, swap: bool = False) -> np.ndarray:
    """``(2d, in)`` -> ``(in, 2d)``: column 2k = row k, column 2k+1 = row d+k (swapped if ``swap``)."""
    half = weight.shape[0] // 2
    first, second = (
        (weight[half:], weight[:half]) if swap else (weight[:half], weight[half:])
    )
    out = np.empty((weight.shape[1], weight.shape[0]), dtype=weight.dtype)
    out[:, 0::2] = first.T
    out[:, 1::2] = second.T
    return out


def pair_track(get, prefix: str, out: dict[str, np.ndarray]) -> None:
    """Pair-track tensors shared by the MSA module's blocks and the pairformer blocks."""
    for tag, src in (("1", "tri_att_start"), ("2", "tri_att_end")):
        o = f"pair_attention{tag}/"
        out[o + "act_norm/offset"] = get(f"{src}.layer_norm.bias")
        out[o + "act_norm/scale"] = get(f"{src}.layer_norm.weight")
        out[o + "gating_query/weights"] = get(f"{src}.mha.linear_g.weight")
        out[o + "k_projection/weights"] = get(f"{src}.mha.linear_k.weight").reshape(
            4, 32, 128
        )
        out[o + "q_projection/weights"] = get(f"{src}.mha.linear_q.weight").reshape(
            4, 32, 128
        )
        out[o + "v_projection/weights"] = get(f"{src}.mha.linear_v.weight").T.reshape(
            128, 4, 32
        )
        out[o + "output_projection/weights"] = get(f"{src}.mha.linear_o.weight").T
        out[o + "pair_bias_projection/weights"] = get(f"{src}.linear.weight").T
    out["pair_transition/input_layer_norm/offset"] = get("transition_z.norm.bias")
    out["pair_transition/input_layer_norm/scale"] = get("transition_z.norm.weight")
    out["pair_transition/transition1/weights"] = np.concatenate(
        [get("transition_z.fc1.weight").T, get("transition_z.fc2.weight").T], axis=1
    )
    out["pair_transition/transition2/weights"] = get("transition_z.fc3.weight").T
    for tag, src, swap in (
        ("outgoing", "tri_mul_out", False),
        ("incoming", "tri_mul_in", True),
    ):
        o = f"triangle_multiplication_{tag}/"
        out[o + "left_norm_input/offset"] = get(f"{src}.norm_in.bias")
        out[o + "left_norm_input/scale"] = get(f"{src}.norm_in.weight")
        out[o + "center_norm/offset"] = get(f"{src}.norm_out.bias")
        out[o + "center_norm/scale"] = get(f"{src}.norm_out.weight")
        out[o + "projection/weights"] = interleave(get(f"{src}.p_in.weight"), swap)
        out[o + "gate/weights"] = interleave(get(f"{src}.g_in.weight"), swap)
        out[o + "gating_linear/weights"] = get(f"{src}.g_out.weight").T
        out[o + "output_projection/weights"] = get(f"{src}.p_out.weight").T


def msa_block(get) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    pair_track(lambda k: get("pairformer_layer." + k), "", out)
    out["msa_attention1/act_norm/offset"] = get("pair_weighted_averaging.norm_m.bias")
    out["msa_attention1/act_norm/scale"] = get("pair_weighted_averaging.norm_m.weight")
    out["msa_attention1/pair_norm/offset"] = get("pair_weighted_averaging.norm_z.bias")
    out["msa_attention1/pair_norm/scale"] = get("pair_weighted_averaging.norm_z.weight")
    out["msa_attention1/gating_query/weights"] = get(
        "pair_weighted_averaging.proj_g.weight"
    ).T
    out["msa_attention1/output_projection/weights"] = get(
        "pair_weighted_averaging.proj_o.weight"
    ).T
    out["msa_attention1/pair_logits/weights"] = get(
        "pair_weighted_averaging.proj_z.weight"
    ).T
    out["msa_attention1/v_projection/weights"] = get(
        "pair_weighted_averaging.proj_m.weight"
    ).T.reshape(64, 8, 32)
    out["msa_transition/input_layer_norm/offset"] = get("msa_transition.norm.bias")
    out["msa_transition/input_layer_norm/scale"] = get("msa_transition.norm.weight")
    out["msa_transition/transition1/weights"] = np.concatenate(
        [get("msa_transition.fc1.weight").T, get("msa_transition.fc2.weight").T], axis=1
    )
    out["msa_transition/transition2/weights"] = get("msa_transition.fc3.weight").T
    out["outer_product_mean/layer_norm_input/offset"] = get(
        "outer_product_mean.norm.bias"
    )
    out["outer_product_mean/layer_norm_input/scale"] = get(
        "outer_product_mean.norm.weight"
    )
    out["outer_product_mean/left_projection/weights"] = get(
        "outer_product_mean.proj_a.weight"
    ).T
    out["outer_product_mean/right_projection/weights"] = get(
        "outer_product_mean.proj_b.weight"
    ).T
    out["outer_product_mean/output_w"] = get(
        "outer_product_mean.proj_o.weight"
    ).T.reshape(32, 32, 128)
    out["outer_product_mean/output_b"] = get("outer_product_mean.proj_o.bias")
    return out


def pairformer_block(get) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    pair_track(get, "", out)
    out["single_attention_layer_norm/offset"] = get("pre_norm_s.bias")
    out["single_attention_layer_norm/scale"] = get("pre_norm_s.weight")
    out["single_attention_gating_query/weights"] = get("attention.proj_g.weight").T
    out["single_attention_k_projection/weights"] = get(
        "attention.proj_k.weight"
    ).T.reshape(384, 16, 24)
    out["single_attention_q_projection/weights"] = get(
        "attention.proj_q.weight"
    ).T.reshape(384, 16, 24)
    out["single_attention_q_projection/bias"] = get("attention.proj_q.bias").reshape(
        16, 24
    )
    out["single_attention_v_projection/weights"] = get(
        "attention.proj_v.weight"
    ).T.reshape(384, 16, 24)
    out["single_attention_transition2/weights"] = get("attention.proj_o.weight").T
    out["single_pair_logits_norm/offset"] = get("attention.proj_z.0.bias")
    out["single_pair_logits_norm/scale"] = get("attention.proj_z.0.weight")
    out["single_pair_logits_projection/weights"] = get("attention.proj_z.1.weight").T
    out["single_transition/input_layer_norm/offset"] = get("transition_s.norm.bias")
    out["single_transition/input_layer_norm/scale"] = get("transition_s.norm.weight")
    out["single_transition/transition1/weights"] = np.concatenate(
        [get("transition_s.fc1.weight").T, get("transition_s.fc2.weight").T], axis=1
    )
    out["single_transition/transition2/weights"] = get("transition_s.fc3.weight").T
    return out


def atom_block(get) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    out["adaptive_zero_cond/weights"] = get("output_projection.0.weight").T
    out["adaptive_zero_cond/bias"] = get("output_projection.0.bias")
    out["ffw_a_to_b/weights"] = get("transition.a_to_b.weight").T
    out["ffw_adaptive_zero_cond/weights"] = get(
        "transition.output_projection.0.weight"
    ).T
    out["ffw_adaptive_zero_cond/bias"] = get("transition.output_projection.0.bias")
    out["ffw_single_cond_bias/weights"] = get("transition.adaln.s_bias.weight").T
    out["ffw_single_cond_layer_norm/scale"] = get("transition.adaln.s_norm.weight")
    out["ffw_single_cond_scale/weights"] = get("transition.adaln.s_scale.weight").T
    out["ffw_single_cond_scale/bias"] = get("transition.adaln.s_scale.bias")
    gate = get("transition.swish_gate.0.weight")
    out["ffw_transition1/weights"] = np.concatenate([gate[256:], gate[:256]], axis=0).T
    out["ffw_transition2/weights"] = get("transition.b_to_a.weight").T
    out["gating_query/weights"] = get("pair_bias_attn.proj_g.weight").T
    out["k_projection/weights"] = get("pair_bias_attn.proj_k.weight").T.reshape(
        128, 4, 32
    )
    out["q_projection/weights"] = get("pair_bias_attn.proj_q.weight").T.reshape(
        128, 4, 32
    )
    out["q_projection/bias"] = get("pair_bias_attn.proj_q.bias").reshape(4, 32)
    out["v_projection/weights"] = get("pair_bias_attn.proj_v.weight").T.reshape(
        128, 4, 32
    )
    out["transition2/weights"] = get("pair_bias_attn.proj_o.weight").T
    for q in ("k", "q"):
        out[f"{q}single_cond_bias/weights"] = get("adaln.s_bias.weight").T
        out[f"{q}single_cond_layer_norm/scale"] = get("adaln.s_norm.weight")
        out[f"{q}single_cond_scale/weights"] = get("adaln.s_scale.weight").T
        out[f"{q}single_cond_scale/bias"] = get("adaln.s_scale.bias")
    return out


def stack(prefix: str, count: int, build, state) -> dict[str, np.ndarray]:
    blocks = [build(lambda k, l=l: state[prefix.format(l=l) + k]) for l in range(count)]
    return {name: np.stack([b[name] for b in blocks]) for name in blocks[0]}


def convert(
    state: dict[str, np.ndarray], template_shapes: dict[str, list[int]]
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    e = ROOT + "evoformer/"
    for name, tensor in stack("msa_module.layers.{l}.", 4, msa_block, state).items():
        out[e + "__layer_stack_no_per_layer/msa_stack/" + name] = tensor
    for name, tensor in stack(
        "pairformer_module.layers.{l}.", 64, pairformer_block, state
    ).items():
        out[e + "__layer_stack_no_per_layer_1/trunk_pairformer/" + name] = tensor
    atom = "evoformer_conditioning_atom_transformer_encoder"
    for name, tensor in stack(
        "input_embedder.atom_attention_encoder.atom_encoder.diffusion_transformer.layers.{l}.",
        3,
        atom_block,
        state,
    ).items():
        out[f"{ROOT}{atom}/__layer_stack_with_per_layer/{atom}{name}"] = tensor

    ie = "input_embedder."
    ae = ie + "atom_encoder."
    out[ROOT + "evoformer_conditioning_embed_atom_features_bias"] = state[
        ae + "embed_atom_features.bias"
    ]
    features = state[ae + "embed_atom_features.weight"]  # (128, 3 + 1 + 128 + 256)
    cuts = {
        "ref_pos": (0, 3),
        "ref_charge": (3, 4),
        "ref_element": (4, 132),
        "ref_atom_name": (132, 388),
    }
    for key, (lo, hi) in cuts.items():
        out[f"{ROOT}evoformer_conditioning_embed_{key}/weights"] = features[:, lo:hi].T
    out[ROOT + "evoformer_conditioning_embed_ref_mask/weights"] = np.zeros(
        (1, 128), np.float32
    )
    offsets = -state[ae + "embed_atompair_ref_pos.weight"].T
    distances = state[ae + "embed_atompair_ref_dist.weight"].T
    for suffix in ("", "_1"):
        out[f"{ROOT}evoformer_conditioning_embed_pair_offsets{suffix}/weights"] = (
            offsets
        )
        out[f"{ROOT}evoformer_conditioning_embed_pair_distances{suffix}/weights"] = (
            distances
        )
        out[f"{ROOT}evoformer_conditioning_single_to_pair_cond_col{suffix}/weights"] = (
            state[ae + "c_to_p_trans_k.1.weight"].T
        )
        out[f"{ROOT}evoformer_conditioning_single_to_pair_cond_row{suffix}/weights"] = (
            state[ae + "c_to_p_trans_q.1.weight"].T
        )
    out[ROOT + "evoformer_conditioning_embed_pair_offsets_valid/weights"] = state[
        ae + "embed_atompair_mask.weight"
    ].T
    for i, src in enumerate((1, 3, 5), start=1):
        out[f"{ROOT}evoformer_conditioning_pair_mlp_{i}/weights"] = state[
            f"{ae}p_mlp.{src}.weight"
        ].T
    out[f"{ROOT}{atom}/pair_input_layer_norm/scale"] = state[
        ie + "atom_enc_proj_z.0.weight"
    ]
    out[f"{ROOT}{atom}/pair_logits_projection/weights"] = state[
        ie + "atom_enc_proj_z.1.weight"
    ].T.reshape(16, 3, 4)
    out[ROOT + "evoformer_conditioning_project_atom_features_for_aggr/weights"] = state[
        ie + "atom_attention_encoder.atom_to_token_trans.0.weight"
    ].T

    res = state[ie + "res_type_encoding.weight"]  # (384, 33)
    prof = state[ie + "msa_profile_encoding.weight"]  # (384, 34)
    out[ROOT + "boltz2_res_type_encoding/weights"] = res[:, TOKEN_COLUMN].T
    out[ROOT + "boltz2_msa_profile_encoding/weights"] = prof[:, [*TOKEN_COLUMN, 33]].T
    out[ROOT + "boltz2_dn_res_type"] = res[:, BOLTZ_DN] - res[:, BOLTZ_N]
    out[ROOT + "boltz2_dn_msa_profile"] = prof[:, BOLTZ_DN] - prof[:, BOLTZ_N]
    out[ROOT + "boltz2_method_conditioning/weights"] = state[
        ie + "method_conditioning_init.weight"
    ]
    out[ROOT + "boltz2_modified_conditioning/weights"] = state[
        ie + "modified_conditioning_init.weight"
    ]
    out[ROOT + "boltz2_mol_type_conditioning/weights"] = state[
        ie + "mol_type_conditioning_init.weight"
    ]
    out[ROOT + "boltz2_cyclic_conditioning/weights"] = state[
        ie + "cyclic_conditioning_init.weight"
    ].T

    msa_in = state["msa_module.msa_proj.weight"]  # (64, 36)
    msa_rows = np.concatenate(
        [msa_in[:, TOKEN_COLUMN].T, np.zeros((1, 64), np.float32), msa_in[:, 33:36].T],
        axis=0,
    )
    out[e + "msa_activations/weights"] = msa_rows
    out[e + "boltz2_dn_msa"] = msa_in[:, BOLTZ_DN] - msa_in[:, BOLTZ_N]
    out[e + "extra_msa_target_feat/weights"] = state["msa_module.s_proj.weight"].T
    out[e + "left_single/weights"] = state["z_init_1.weight"].T
    out[e + "right_single/weights"] = state["z_init_2.weight"].T
    out[e + "single_activations/weights"] = state["s_init.weight"].T
    out[e + "prev_embedding/weights"] = state["z_recycle.weight"].T
    out[e + "prev_embedding_layer_norm/scale"] = state["z_norm.weight"]
    out[e + "prev_embedding_layer_norm/offset"] = state["z_norm.bias"]
    out[e + "prev_single_embedding/weights"] = state["s_recycle.weight"].T
    out[e + "prev_single_embedding_layer_norm/scale"] = state["s_norm.weight"]
    out[e + "prev_single_embedding_layer_norm/offset"] = state["s_norm.bias"]
    out[e + "bond_embedding/weights"] = state["token_bonds.weight"].T
    out[e + "token_bonds_type_embed/weights"] = state["token_bonds_type.weight"]
    out[e + "~_relative_encoding/position_activations/weights"] = state[
        "rel_pos.linear_layer.weight"
    ].T
    cc = "contact_conditioning."
    out[e + "contact_encoding_unselected"] = state[cc + "encoding_unselected"]
    out[e + "contact_encoding_unspecified"] = state[cc + "encoding_unspecified"]
    out[e + "contact_encoder/weights"] = state[cc + "encoder.weight"].T
    out[e + "contact_encoder/bias"] = state[cc + "encoder.bias"]
    out[e + "contact_fourier/weights"] = state[cc + "fourier_embedding.proj.weight"].T
    out[e + "contact_fourier/bias"] = state[cc + "fourier_embedding.proj.bias"]
    out[ROOT + "distogram_head/half_logits/weights"] = state[
        "distogram_module.distogram.weight"
    ].T
    out[ROOT + "distogram_head/half_logits/bias"] = (
        state["distogram_module.distogram.bias"] / 2
    )

    for name, shape in template_shapes.items():
        is_scale = name.endswith("/scale")
        out[name] = (np.ones if is_scale else np.zeros)(shape, np.float32)
    return out


def template_tensor_shapes() -> dict[str, list[int]]:
    """Shapes of the (unused) template module, as LocalFold's loader expects them."""
    shapes: dict[str, list[int]] = {}
    t = ROOT + "evoformer/template_embedding/"
    for name, shape in (
        ("a_proj/weights", [109, 64]),
        ("u_proj/weights", [64, 128]),
        ("v_norm/offset", [64]),
        ("v_norm/scale", [64]),
        ("z_norm/offset", [128]),
        ("z_norm/scale", [128]),
        ("z_proj/weights", [128, 64]),
    ):
        shapes[t + name] = shape
    stack_name = t + "__layer_stack_no_per_layer/tmpl_pairformer/"
    c = 64
    for tag in ("1", "2"):
        o = f"pair_attention{tag}/"
        shapes[stack_name + o + "act_norm/offset"] = [2, c]
        shapes[stack_name + o + "act_norm/scale"] = [2, c]
        shapes[stack_name + o + "gating_query/weights"] = [2, 128, c]
        shapes[stack_name + o + "k_projection/weights"] = [2, 4, 32, c]
        shapes[stack_name + o + "q_projection/weights"] = [2, 4, 32, c]
        shapes[stack_name + o + "v_projection/weights"] = [2, c, 4, 32]
        shapes[stack_name + o + "output_projection/weights"] = [2, 128, c]
        shapes[stack_name + o + "pair_bias_projection/weights"] = [2, c, 4]
    shapes[stack_name + "pair_transition/input_layer_norm/offset"] = [2, c]
    shapes[stack_name + "pair_transition/input_layer_norm/scale"] = [2, c]
    shapes[stack_name + "pair_transition/transition1/weights"] = [2, c, 512]
    shapes[stack_name + "pair_transition/transition2/weights"] = [2, 256, c]
    for tag in ("outgoing", "incoming"):
        o = f"triangle_multiplication_{tag}/"
        for norm in ("center_norm", "left_norm_input"):
            shapes[stack_name + o + norm + "/offset"] = [2, c]
            shapes[stack_name + o + norm + "/scale"] = [2, c]
        shapes[stack_name + o + "gate/weights"] = [2, c, 128]
        shapes[stack_name + o + "projection/weights"] = [2, c, 128]
        shapes[stack_name + o + "gating_linear/weights"] = [2, c, c]
        shapes[stack_name + o + "output_projection/weights"] = [2, c, c]
    return shapes


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sharded(
    out_dir: Path, tensors: dict[str, np.ndarray], meta: dict[str, Any]
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    shard, offset, handle = 0, 0, None
    for name, array in tensors.items():
        data = np.ascontiguousarray(array, dtype=np.float32)
        if handle is None or (offset > 0 and offset + data.nbytes > SHARD_BYTES):
            if handle is not None:
                handle.close()
            shard += 1
            offset = 0
            handle = (out_dir / f"weights-{shard:02d}.bin").open("wb")
        assert handle is not None
        handle.write(data.tobytes())
        records[name] = {
            "file": f"weights-{shard:02d}.bin",
            "byteOffset": offset,
            "dtype": "float32",
            "shape": list(data.shape),
        }
        offset += data.nbytes
    assert handle is not None
    handle.close()
    manifest = {
        "formatVersion": 1,
        "source": "Boltz-2 trunk converted from boltz2_aff.ckpt (boltz-genfree webgpu/tools/convert_trunk_weights.py)",
        "model": {"name": "boltz2", "recycles": 0},
        "meta": meta,
        "tensors": records,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path.home() / ".boltz" / "boltz2_aff.ckpt"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("webgpu/packs/boltz2-trunk-f32")
    )
    args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {k: v.detach().float().numpy() for k, v in ckpt["state_dict"].items()}
    tensors = convert(state, template_tensor_shapes())
    meta = {
        "checkpoint": args.checkpoint.name,
        "checkpointSha256": sha256_of(args.checkpoint),
        "tensors": len(tensors),
    }
    path = write_sharded(args.out_dir, tensors, meta)
    total = sum(t.nbytes for t in tensors.values())
    print(f"wrote {path}: {len(tensors)} tensors, {total / 2**20:.0f} MiB float32")


if __name__ == "__main__":
    main()

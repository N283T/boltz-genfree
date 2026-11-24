"""Map LocalFold's int5 Boltz-2 bundle onto the PyTorch-named trunk pack.

LocalFold (sokrypton/localfold, ``boltz2-int5`` on Hugging Face) ships the Boltz-2 weights with AF3/Haiku
names and layouts (layer-stacked tensors, ``(in, out)`` weights, factored head axes, fused projections) as an
int5 + float16-scale/zero bundle. The trunk modules read PyTorch names and layouts
(``webgpu/src/trunk/own-weights.js``). This script derives, for every tensor of the f32 trunk pack
(``tools/convert_boltz_trunk.py``), a small recipe over LocalFold tensors and writes
``src/trunk/localfold-map.json``, which ``src/trunk/localfold-weights.js`` applies at load time.

Every recipe is verified here: the int5 tensors are decoded, the recipe applied, and the result compared with the
f32 pack (relative L2 error; int5 noise is ~0.04, a wrong mapping gives ~1.0 or more).

Recipe language (JSON; the JS loader implements exactly this):
  node  := {"src": "${ALIAS}name", "layer": true?, "ops": [op...]?}      one LocalFold tensor (``layer``: take layer L)
         | {"zeros": [shape...], "ops": ...?}                            a tensor with no counterpart in the bundle
         | {"concat": axis, "parts": [node...], "ops": ...?}
         | {"add": [node, node], "ops": ...?}
  op    := ["reshape", shape] | ["transpose", perm] | ["slice", axis, start, end] | ["flip", axis]
         | ["take", axis, indices] | ["scale", factor]
A rule is {"pack": name (``{L}`` = layer index), "layers": N|null, "shape": [...], "recipe": node, "note"?: str}.

Usage (from webgpu/):
  python tools/map_localfold_weights.py --manifest MANIFEST.json --shards-dir DIR
      [--pack packs/boltz2-trunk-own/boltz2-trunk-own.json] [--out src/trunk/localfold-map.json]
      [--write-fixture test/fixtures/localfold-int5.json]
"""

# ruff: noqa: FBT003  (recipe builders pass the "stacked tensor" flag positionally)
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WEBGPU = HERE.parent

ALIASES = {
    "TOP": "diffuser/",
    "EV": "diffuser/evoformer/",
    "MSA": "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/",
    "PF": "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/",
    "ET": "diffuser/evoformer_conditioning_atom_transformer_encoder/",
    "AT": "diffuser/evoformer_conditioning_atom_transformer_encoder/__layer_stack_with_per_layer/"
    "evoformer_conditioning_atom_transformer_encoder",
    "EC": "diffuser/evoformer_conditioning_",
}


# --------------------------------------------------------------------------- int5 decoding


class Bundle:
    """Reads LocalFold manifest tensors from local shard files (int5 general path of dtype.js readTensorRange)."""

    def __init__(self, manifest: Path, shards: Path):
        doc = json.loads(manifest.read_text())
        self.records = doc["tensors"]
        self.shards = shards
        self._maps: dict[str, np.memmap] = {}

    def _bytes(self, file: str) -> np.memmap:
        if file not in self._maps:
            self._maps[file] = np.memmap(self.shards / file, dtype=np.uint8, mode="r")
        return self._maps[file]

    def get(self, name: str, layer: int | None = None) -> np.ndarray:
        rec = self.records[name]
        shape = rec["shape"]
        data = self._bytes(rec["file"])
        count = int(np.prod(shape))
        if rec["dtype"] == "float32":
            raw = np.frombuffer(
                bytes(data[rec["byteOffset"] : rec["byteOffset"] + 4 * count]),
                dtype="<f4",
            )
            out = raw.reshape(shape)
        elif rec["dtype"] == "int5":
            out = decode_int5(
                data,
                rec["byteOffset"],
                rec["scaleOffset"],
                rec["zeroOffset"],
                rec["block"],
                count,
            ).reshape(shape)
        else:
            raise ValueError(f"{name}: unsupported dtype {rec['dtype']}")
        return out if layer is None else out[layer]


def decode_int5(
    data, code_at: int, scale_at: int, zero_at: int, block: int, count: int
) -> np.ndarray:
    """Groups of ``block`` elements, 5-bit codes packed LSB-first, value = code * scale[g] + zero[g] (float16 g)."""
    groups = -(-count // block)
    group_bytes = block * 5 // 8
    codes = np.frombuffer(
        bytes(data[code_at : code_at + groups * group_bytes + 1]), dtype=np.uint8
    )
    if codes.size < groups * group_bytes + 1:
        codes = np.concatenate(
            [codes, np.zeros(groups * group_bytes + 1 - codes.size, np.uint8)]
        )
    scale = np.frombuffer(
        bytes(data[scale_at : scale_at + 2 * groups]), dtype="<f2"
    ).astype(np.float32)
    zero = np.frombuffer(
        bytes(data[zero_at : zero_at + 2 * groups]), dtype="<f2"
    ).astype(np.float32)
    index = np.arange(count)
    group = index // block
    bit = group * group_bytes * 8 + (index - group * block) * 5
    byte = bit >> 3
    value = (
        (codes[byte].astype(np.int32) | (codes[byte + 1].astype(np.int32) << 8))
        >> (bit & 7)
    ) & 31
    return (value * scale[group] + zero[group]).astype(np.float32)


def encode_int5(values: np.ndarray, block: int = 32):
    """Pure-Python reference encoder (asymmetric per group, float16 scale/zero) for the JS test fixture."""
    flat = np.asarray(values, dtype=np.float32).ravel()
    groups = -(-flat.size // block)
    padded = np.zeros(groups * block, np.float32)
    padded[: flat.size] = flat
    scales, zeros, packed = [], [], bytearray()
    for g in range(groups):
        chunk = padded[g * block : (g + 1) * block]
        lo, hi = float(chunk.min()), float(chunk.max())
        zero = np.float16(lo)
        scale = np.float16((hi - lo) / 31 if hi > lo else 1.0)
        scales.append(scale)
        zeros.append(zero)
        acc = 0
        for i, v in enumerate(chunk):
            code = int(np.clip(np.rint((float(v) - float(zero)) / float(scale)), 0, 31))
            acc |= code << (i * 5)
        packed += acc.to_bytes(block * 5 // 8, "little")
    return bytes(packed), np.array(scales, np.float16), np.array(zeros, np.float16)


# --------------------------------------------------------------------------- recipe evaluation


def expand(src: str) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: ALIASES[m.group(1)], src)


def apply_ops(x: np.ndarray, ops) -> np.ndarray:
    for op in ops or []:
        kind = op[0]
        if kind == "reshape":
            x = x.reshape(op[1])
        elif kind == "transpose":
            x = np.ascontiguousarray(x.transpose(op[1]))
        elif kind == "slice":
            index = [slice(None)] * x.ndim
            index[op[1]] = slice(op[2], op[3])
            x = np.ascontiguousarray(x[tuple(index)])
        elif kind == "flip":
            x = np.ascontiguousarray(np.flip(x, op[1]))
        elif kind == "take":
            x = np.ascontiguousarray(np.take(x, op[2], axis=op[1]))
        elif kind == "scale":
            x = (x * np.float32(op[1])).astype(np.float32)
        else:
            raise ValueError(f"unknown op {kind}")
    return x


def evaluate(node, bundle: Bundle, layer: int | None) -> np.ndarray:
    if "src" in node:
        x = bundle.get(expand(node["src"]), layer if node.get("layer") else None)
    elif "zeros" in node:
        x = np.zeros(node["zeros"], np.float32)
    elif "concat" in node:
        x = np.concatenate(
            [evaluate(p, bundle, layer) for p in node["parts"]], axis=node["concat"]
        )
    elif "add" in node:
        a, b = (evaluate(p, bundle, layer) for p in node["add"])
        x = (a + b).astype(np.float32)
    else:
        raise ValueError(f"bad node {node}")
    return apply_ops(x, node.get("ops"))


# --------------------------------------------------------------------------- the mapping


def leaf(src: str, ops=None, layer: bool = False) -> dict:
    node: dict = {"src": src}
    if layer:
        node["layer"] = True
    if ops:
        node["ops"] = ops
    return node


T2 = [["transpose", [1, 0]]]


def lin(src: str, extra=None, layer: bool = True) -> dict:
    """(in, out) -> PyTorch (out, in)."""
    return leaf(src, [*(extra or []), *T2], layer)


def norm_rules(
    pack: str, src: str, layers: int | None, layer: bool, bias: str = "bias"
) -> list:
    return [
        rule(f"{pack}/weight", leaf(f"{src}/scale", layer=layer), layers),
        rule(f"{pack}/{bias}", leaf(f"{src}/offset", layer=layer), layers),
    ]


def rule(
    pack: str, recipe: dict, layers: int | None = None, note: str | None = None
) -> dict:
    out = {"pack": pack, "layers": layers, "recipe": recipe}
    if note:
        out["note"] = note
    return out


def pair_block(pack: str, base: str, layers: int) -> list:
    """Triangle multiplication / triangle attention / pair transition (shared by the MSA module and the pairformer)."""
    rules: list = []
    n = layers
    for tm, lf, flip in (
        ("tri_mul_out", "triangle_multiplication_outgoing", False),
        ("tri_mul_in", "triangle_multiplication_incoming", True),
    ):
        src = f"{base}{lf}"
        # (128, 256) projections interleave the two halves along the output axis: column 2k is a_k, 2k+1 is b_k
        # (incoming: swapped); PyTorch stacks [a (128 rows); b (128 rows)].
        inter = (
            [["reshape", [128, 128, 2]]]
            + ([["flip", 2]] if flip else [])
            + [["transpose", [2, 1, 0]], ["reshape", [256, 128]]]
        )
        rules += [
            rule(
                f"{pack}/{tm}/p_in/weight",
                leaf(f"{src}/projection/weights", inter, True),
                n,
            ),
            rule(
                f"{pack}/{tm}/g_in/weight", leaf(f"{src}/gate/weights", inter, True), n
            ),
            rule(f"{pack}/{tm}/g_out/weight", lin(f"{src}/gating_linear/weights"), n),
            rule(
                f"{pack}/{tm}/p_out/weight", lin(f"{src}/output_projection/weights"), n
            ),
            *norm_rules(f"{pack}/{tm}/norm_in", f"{src}/left_norm_input", n, True),
            *norm_rules(f"{pack}/{tm}/norm_out", f"{src}/center_norm", n, True),
        ]
    for ta, lf in (
        ("tri_att_start", "pair_attention1"),
        ("tri_att_end", "pair_attention2"),
    ):
        src = f"{base}{lf}"
        flat = [["reshape", [128, 128]]]
        rules += [
            *norm_rules(f"{pack}/{ta}/layer_norm", f"{src}/act_norm", n, True),
            rule(
                f"{pack}/{ta}/linear/weight",
                lin(f"{src}/pair_bias_projection/weights"),
                n,
            ),
            # q/k: (heads, head_dim, in) is already (out, in); v: (in, heads, head_dim); gate/output: (in, out)
            rule(
                f"{pack}/{ta}/mha/linear_q/weight",
                leaf(f"{src}/q_projection/weights", flat, True),
                n,
            ),
            rule(
                f"{pack}/{ta}/mha/linear_k/weight",
                leaf(f"{src}/k_projection/weights", flat, True),
                n,
            ),
            rule(
                f"{pack}/{ta}/mha/linear_v/weight",
                lin(f"{src}/v_projection/weights", flat),
                n,
            ),
            rule(
                f"{pack}/{ta}/mha/linear_g/weight",
                leaf(f"{src}/gating_query/weights", layer=True),
                n,
            ),
            rule(
                f"{pack}/{ta}/mha/linear_o/weight",
                lin(f"{src}/output_projection/weights"),
                n,
            ),
        ]
    rules += swiglu(f"{pack}/transition_z", f"{base}pair_transition", 512, n)
    return rules


def swiglu(pack: str, src: str, hidden: int, layers: int) -> list:
    return [
        *norm_rules(f"{pack}/norm", f"{src}/input_layer_norm", layers, True),
        rule(
            f"{pack}/fc1/weight",
            lin(f"{src}/transition1/weights", [["slice", 1, 0, hidden]]),
            layers,
        ),
        rule(
            f"{pack}/fc2/weight",
            lin(f"{src}/transition1/weights", [["slice", 1, hidden, 2 * hidden]]),
            layers,
        ),
        rule(f"{pack}/fc3/weight", lin(f"{src}/transition2/weights"), layers),
    ]


def build_rules() -> list:
    rules: list = []
    # ---- pairformer (64 layers)
    pf, n = "pairformer_module/layers/{L}", 64
    heads = [["reshape", [384, 384]]]
    rules += norm_rules(f"{pf}/pre_norm_s", "${PF}single_attention_layer_norm", n, True)
    rules += [
        rule(
            f"{pf}/attention/proj_q/weight",
            lin("${PF}single_attention_q_projection/weights", heads),
            n,
        ),
        rule(
            f"{pf}/attention/proj_q/bias",
            leaf("${PF}single_attention_q_projection/bias", [["reshape", [384]]], True),
            n,
        ),
        rule(
            f"{pf}/attention/proj_k/weight",
            lin("${PF}single_attention_k_projection/weights", heads),
            n,
        ),
        rule(
            f"{pf}/attention/proj_v/weight",
            lin("${PF}single_attention_v_projection/weights", heads),
            n,
        ),
        rule(
            f"{pf}/attention/proj_g/weight",
            lin("${PF}single_attention_gating_query/weights"),
            n,
        ),
        rule(
            f"{pf}/attention/proj_o/weight",
            lin("${PF}single_attention_transition2/weights"),
            n,
        ),
        *norm_rules(
            f"{pf}/attention/proj_z/0", "${PF}single_pair_logits_norm", n, True
        ),
        rule(
            f"{pf}/attention/proj_z/1/weight",
            lin("${PF}single_pair_logits_projection/weights"),
            n,
        ),
    ]
    rules += swiglu(f"{pf}/transition_s", "${PF}single_transition", 1536, n)
    rules += pair_block(pf, "${PF}", n)
    # ---- MSA module (4 blocks)
    ms, n = "msa_module/layers/{L}", 4
    rules += swiglu(f"{ms}/msa_transition", "${MSA}msa_transition", 256, n)
    # msa_transition has no pair part: replace the last tensors' names (swiglu already used the right prefix)
    opm = "${MSA}outer_product_mean"
    rules += [
        *norm_rules(
            f"{ms}/outer_product_mean/norm", f"{opm}/layer_norm_input", n, True
        ),
        rule(
            f"{ms}/outer_product_mean/proj_a/weight",
            lin(f"{opm}/left_projection/weights"),
            n,
        ),
        rule(
            f"{ms}/outer_product_mean/proj_b/weight",
            lin(f"{opm}/right_projection/weights"),
            n,
        ),
        rule(
            f"{ms}/outer_product_mean/proj_o/weight",
            leaf(f"{opm}/output_w", [["reshape", [1024, 128]], *T2], True),
            n,
        ),
        rule(
            f"{ms}/outer_product_mean/proj_o/bias",
            leaf(f"{opm}/output_b", layer=True),
            n,
        ),
    ]
    pw = "${MSA}msa_attention1"
    rules += [
        *norm_rules(f"{ms}/pair_weighted_averaging/norm_m", f"{pw}/act_norm", n, True),
        *norm_rules(f"{ms}/pair_weighted_averaging/norm_z", f"{pw}/pair_norm", n, True),
        rule(
            f"{ms}/pair_weighted_averaging/proj_z/weight",
            lin(f"{pw}/pair_logits/weights"),
            n,
        ),
        rule(
            f"{ms}/pair_weighted_averaging/proj_m/weight",
            lin(f"{pw}/v_projection/weights", [["reshape", [64, 256]]]),
            n,
        ),
        rule(
            f"{ms}/pair_weighted_averaging/proj_g/weight",
            lin(f"{pw}/gating_query/weights"),
            n,
        ),
        rule(
            f"{ms}/pair_weighted_averaging/proj_o/weight",
            lin(f"{pw}/output_projection/weights"),
            n,
        ),
    ]
    rules += pair_block(f"{ms}/pairformer_layer", "${MSA}", n)
    # ---- unstacked embeddings and recycling
    rules += [
        rule("s_init/weight", lin("${EV}single_activations/weights", layer=False)),
        rule("z_init_1/weight", lin("${EV}left_single/weights", layer=False)),
        rule("z_init_2/weight", lin("${EV}right_single/weights", layer=False)),
        rule("z_recycle/weight", lin("${EV}prev_embedding/weights", layer=False)),
        rule(
            "s_recycle/weight", lin("${EV}prev_single_embedding/weights", layer=False)
        ),
        rule("z_norm/weight", leaf("${EV}prev_embedding_layer_norm/scale")),
        rule("z_norm/bias", leaf("${EV}prev_embedding_layer_norm/offset")),
        rule("s_norm/weight", leaf("${EV}prev_single_embedding_layer_norm/scale")),
        rule("s_norm/bias", leaf("${EV}prev_single_embedding_layer_norm/offset")),
        rule("token_bonds/weight", lin("${EV}bond_embedding/weights", layer=False)),
        rule("token_bonds_type/weight", leaf("${EV}token_bonds_type_embed/weights")),
        rule(
            "rel_pos/linear_layer/weight",
            lin("${EV}~_relative_encoding/position_activations/weights", layer=False),
        ),
        rule(
            "msa_module/s_proj/weight",
            lin("${EV}extra_msa_target_feat/weights", layer=False),
        ),
        rule(
            "contact_conditioning/encoder/weight",
            lin("${EV}contact_encoder/weights", layer=False),
        ),
        rule("contact_conditioning/encoder/bias", leaf("${EV}contact_encoder/bias")),
        rule(
            "contact_conditioning/fourier_embedding/proj/weight",
            lin("${EV}contact_fourier/weights", layer=False),
        ),
        rule(
            "contact_conditioning/fourier_embedding/proj/bias",
            leaf("${EV}contact_fourier/bias"),
        ),
        rule(
            "contact_conditioning/encoding_unselected",
            leaf("${EV}contact_encoding_unselected"),
        ),
        rule(
            "contact_conditioning/encoding_unspecified",
            leaf("${EV}contact_encoding_unspecified"),
        ),
    ]
    # ---- input embedder: token conditioning
    ie = "input_embedder"
    rules += [
        rule(
            f"{ie}/method_conditioning_init/weight",
            leaf("${TOP}boltz2_method_conditioning/weights"),
        ),
        rule(
            f"{ie}/modified_conditioning_init/weight",
            leaf("${TOP}boltz2_modified_conditioning/weights"),
        ),
        rule(
            f"{ie}/mol_type_conditioning_init/weight",
            leaf("${TOP}boltz2_mol_type_conditioning/weights"),
        ),
        rule(
            f"{ie}/cyclic_conditioning_init/weight",
            lin("${TOP}boltz2_cyclic_conditioning/weights", layer=False),
        ),
    ]
    rules += token_class_rules()
    # ---- input embedder: atom encoder
    ae = f"{ie}/atom_encoder"
    ec = "${EC}"
    rules += [
        rule(
            f"{ae}/embed_atom_features/weight",
            {
                "concat": 0,
                "ops": T2,
                "parts": [
                    leaf(f"{ec}embed_ref_pos/weights"),
                    leaf(f"{ec}embed_ref_charge/weights"),
                    leaf(f"{ec}embed_ref_element/weights"),
                    leaf(f"{ec}embed_ref_atom_name/weights"),
                ],
            },
        ),
        rule(
            f"{ae}/embed_atom_features/bias",
            leaf("${TOP}evoformer_conditioning_embed_atom_features_bias"),
        ),
        # LocalFold's pair offsets are k - q where Boltz uses q - k: the weight is negated
        rule(
            f"{ae}/embed_atompair_ref_pos/weight",
            leaf(f"{ec}embed_pair_offsets/weights", [*T2, ["scale", -1]]),
        ),
        rule(
            f"{ae}/embed_atompair_ref_dist/weight",
            lin(f"{ec}embed_pair_distances/weights", layer=False),
        ),
        rule(
            f"{ae}/embed_atompair_mask/weight",
            lin(f"{ec}embed_pair_offsets_valid/weights", layer=False),
        ),
        rule(
            f"{ae}/c_to_p_trans_q/1/weight",
            lin(f"{ec}single_to_pair_cond_row/weights", layer=False),
        ),
        rule(
            f"{ae}/c_to_p_trans_k/1/weight",
            lin(f"{ec}single_to_pair_cond_col/weights", layer=False),
        ),
        rule(f"{ae}/p_mlp/1/weight", lin(f"{ec}pair_mlp_1/weights", layer=False)),
        rule(f"{ae}/p_mlp/3/weight", lin(f"{ec}pair_mlp_2/weights", layer=False)),
        rule(f"{ae}/p_mlp/5/weight", lin(f"{ec}pair_mlp_3/weights", layer=False)),
        rule(
            f"{ie}/atom_attention_encoder/atom_to_token_trans/0/weight",
            lin(f"{ec}project_atom_features_for_aggr/weights", layer=False),
        ),
        rule(
            f"{ie}/atom_enc_proj_z/0/weight", leaf("${ET}pair_input_layer_norm/scale")
        ),
        rule(
            f"{ie}/atom_enc_proj_z/0/bias",
            {"zeros": [16]},
            note="no counterpart in the bundle; the LayerNorm offset feeds a bias-free linear whose 12 outputs are "
            "constant attention-logit biases per head, which softmax is invariant to",
        ),
        rule(
            f"{ie}/atom_enc_proj_z/1/weight",
            leaf("${ET}pair_logits_projection/weights", [["reshape", [16, 12]], *T2]),
        ),
    ]
    # ---- atom transformer (3 layers)
    dt, n = (
        f"{ae.replace('atom_encoder', 'atom_attention_encoder/atom_encoder')}/diffusion_transformer/layers/{{L}}",
        3,
    )
    # q/k conditioning weights are identical in the bundle (tied); the query-side ones are used
    for prefix, group in ((dt, "q"), (f"{dt}/transition", "ffw_")):
        s = "${AT}" + (f"{group}single_cond" if group == "q" else "ffw_single_cond")
        rules += [
            rule(
                f"{prefix}/adaln/s_norm/weight",
                leaf(f"{s}_layer_norm/scale", layer=True),
                n,
            ),
            rule(f"{prefix}/adaln/s_scale/weight", lin(f"{s}_scale/weights"), n),
            rule(
                f"{prefix}/adaln/s_scale/bias", leaf(f"{s}_scale/bias", layer=True), n
            ),
            rule(f"{prefix}/adaln/s_bias/weight", lin(f"{s}_bias/weights"), n),
        ]
    flat = [["reshape", [128, 128]]]
    rules += [
        rule(
            f"{dt}/output_projection/0/weight",
            lin("${AT}adaptive_zero_cond/weights"),
            n,
        ),
        rule(
            f"{dt}/output_projection/0/bias",
            leaf("${AT}adaptive_zero_cond/bias", layer=True),
            n,
        ),
        rule(
            f"{dt}/output_projection_linear/weight",
            lin("${AT}adaptive_zero_cond/weights"),
            n,
        ),
        rule(
            f"{dt}/output_projection_linear/bias",
            leaf("${AT}adaptive_zero_cond/bias", layer=True),
            n,
        ),
        rule(
            f"{dt}/pair_bias_attn/proj_q/weight",
            lin("${AT}q_projection/weights", flat),
            n,
        ),
        rule(
            f"{dt}/pair_bias_attn/proj_q/bias",
            leaf("${AT}q_projection/bias", [["reshape", [128]]], True),
            n,
        ),
        rule(
            f"{dt}/pair_bias_attn/proj_k/weight",
            lin("${AT}k_projection/weights", flat),
            n,
        ),
        rule(
            f"{dt}/pair_bias_attn/proj_v/weight",
            lin("${AT}v_projection/weights", flat),
            n,
        ),
        rule(f"{dt}/pair_bias_attn/proj_g/weight", lin("${AT}gating_query/weights"), n),
        rule(f"{dt}/pair_bias_attn/proj_o/weight", lin("${AT}transition2/weights"), n),
        rule(
            f"{dt}/transition/output_projection/0/weight",
            lin("${AT}ffw_adaptive_zero_cond/weights"),
            n,
        ),
        rule(
            f"{dt}/transition/output_projection/0/bias",
            leaf("${AT}ffw_adaptive_zero_cond/bias", layer=True),
            n,
        ),
        rule(f"{dt}/transition/a_to_b/weight", lin("${AT}ffw_a_to_b/weights"), n),
        # the two 256-wide halves of the gate projection are stored swapped
        rule(
            f"{dt}/transition/swish_gate/0/weight",
            lin(
                "${AT}ffw_transition1/weights",
                [["reshape", [128, 2, 256]], ["flip", 1], ["reshape", [128, 512]]],
            ),
            n,
        ),
        rule(f"{dt}/transition/b_to_a/weight", lin("${AT}ffw_transition2/weights"), n),
    ]
    return rules


def token_class_rules() -> list:
    """Token-class columns: LocalFold has 31 classes (AF3 order, N for RNA, no pad, no separate DNA N).

    Boltz columns: 0 pad, 1 gap, 2..22 amino acids + UNK, 23..26 RNA A G C U, 27 RNA N, 28..31 DNA A G C T, 32 DNA N.
    LocalFold rows: 0..20 amino acids + UNK, 21 gap, 22..25 RNA, 26..29 DNA, 30 N. Boltz's DNA N column is stored as
    the delta ``boltz2_dn_*`` to the N row. The pad column (token class 0, never set for a real token) has no
    counterpart and is zero.
    """
    order = [
        21,
        *range(21),
        22,
        23,
        24,
        25,
        30,
        26,
        27,
        28,
        29,
    ]  # LocalFold rows for Boltz columns 1..31

    def rows(
        src: str,
    ) -> dict:  # (classes, d) -> (d, 31) columns for Boltz classes 1..31
        return leaf(src, [["take", 0, order], *T2])

    def dn(src: str, delta: str) -> dict:  # Boltz DNA N column (d, 1)
        return {
            "add": [
                leaf(src, [["slice", 0, 30, 31], *T2]),
                leaf(delta, [["reshape", [-1, 1]]]),
            ]
        }

    ie = "input_embedder"
    res, prof = (
        "${TOP}boltz2_res_type_encoding/weights",
        "${TOP}boltz2_msa_profile_encoding/weights",
    )
    note = "column 0 (pad token) has no counterpart in the bundle and is zero"
    return [
        rule(
            f"{ie}/res_type_encoding/weight",
            {
                "concat": 1,
                "parts": [
                    {"zeros": [384, 1]},
                    rows(res),
                    dn(res, "${TOP}boltz2_dn_res_type"),
                ],
            },
            note=note,
        ),
        rule(
            f"{ie}/msa_profile_encoding/weight",
            {
                "concat": 1,
                "parts": [
                    {"zeros": [384, 1]},
                    rows(prof),
                    dn(prof, "${TOP}boltz2_dn_msa_profile"),
                    leaf(prof, [["slice", 0, 31, 32], *T2]),
                ],
            },
            note=note,
        ),
        rule(
            "msa_module/msa_proj/weight",
            {
                "concat": 1,
                "parts": [
                    {"zeros": [64, 1]},
                    rows("${EV}msa_activations/weights"),
                    dn("${EV}msa_activations/weights", "${EV}boltz2_dn_msa"),
                    leaf("${EV}msa_activations/weights", [["slice", 0, 32, 35], *T2]),
                ],
            },
            note=note,
        ),
    ]


# --------------------------------------------------------------------------- verification and output


def load_pack(path: Path):
    doc = json.loads(path.read_text())
    return (
        doc,
        doc["tensors"],
        np.memmap(path.with_suffix(".bin"), dtype=np.uint8, mode="r"),
    )


def pack_tensor(records, blob, name: str) -> np.ndarray:
    rec = records[name]
    count = int(np.prod(rec["shape"]))
    return np.frombuffer(
        bytes(blob[rec["byteOffset"] : rec["byteOffset"] + 4 * count]), dtype="<f4"
    ).reshape(rec["shape"])


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def write_fixture(path: Path) -> None:
    rng = np.random.default_rng(7)
    values = (rng.standard_normal(5 * 32 + 11) * 3).astype(
        np.float32
    )  # 6 groups, the last one partial
    codes, scales, zeros = encode_int5(values)
    count = values.size
    groups = len(scales)
    # the bundle layout: codes, then scales, then zeros, all in one shard (offsets absolute)
    shard = bytearray(13)  # arbitrary leading bytes so byteOffset != 0
    code_at = len(shard)
    shard += codes
    scale_at = len(shard)
    shard += scales.tobytes()
    zero_at = len(shard)
    shard += zeros.tobytes()
    expected = decode_int5(
        np.frombuffer(bytes(shard), np.uint8), code_at, scale_at, zero_at, 32, count
    )
    doc = {
        "count": count,
        "block": 32,
        "groups": groups,
        "codeAt": code_at,
        "scaleAt": scale_at,
        "zeroAt": zero_at,
        "shard": base64.b64encode(bytes(shard)).decode(),
        "expected": base64.b64encode(expected.astype("<f4").tobytes()).decode(),
        "original": base64.b64encode(values.astype("<f4").tobytes()).decode(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    print(
        f"wrote {path} (max encode error {float(np.abs(expected - values).max()):.4f})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/private/tmp/claude-501/lf-boltz2/manifest.json"),
    )
    parser.add_argument(
        "--shards-dir",
        type=Path,
        default=Path("/private/tmp/claude-501/lf-boltz2/shards"),
    )
    parser.add_argument(
        "--pack",
        type=Path,
        default=WEBGPU / "packs/boltz2-trunk-own/boltz2-trunk-own.json",
    )
    parser.add_argument(
        "--out", type=Path, default=WEBGPU / "src/trunk/localfold-map.json"
    )
    parser.add_argument("--max-error", type=float, default=0.15)
    parser.add_argument("--write-fixture", type=Path, default=None)
    args = parser.parse_args()
    if args.write_fixture:
        write_fixture(args.write_fixture)

    bundle = Bundle(args.manifest, args.shards_dir)
    pack_doc, records, blob = load_pack(args.pack)
    rules = build_rules()

    seen: dict[str, dict] = {}
    errors: list[tuple[float, str]] = []
    unsourced: list[str] = []
    for r in rules:
        count = r["layers"] or 1
        for layer in range(count):
            name = r["pack"].replace("{L}", str(layer))
            if name in seen:
                raise SystemExit(f"duplicate rule for {name}")
            if name not in records:
                raise SystemExit(f"rule for {name}, which is not in the pack")
            seen[name] = r
            got = evaluate(r["recipe"], bundle, layer if r["layers"] else None)
            want = pack_tensor(records, blob, name)
            if list(got.shape) != list(want.shape):
                raise SystemExit(
                    f"{name}: recipe shape {got.shape}, pack shape {want.shape}"
                )
            errors.append((rel_l2(got, want), name))
            r.setdefault("shape", list(want.shape))
    missing = sorted(set(records) - set(seen))
    for r in rules:
        if "note" in r:
            unsourced.append(r["pack"])
    errs = np.array([e for e, _ in errors])
    print(
        f"pack tensors: {len(records)}, mapped: {len(seen)}, without rule: {len(missing)}"
    )
    print(
        f"relative L2 error: max {errs.max():.4f}, median {np.median(errs):.4f}, mean {errs.mean():.4f}"
    )
    for e, n in sorted(errors, reverse=True)[:8]:
        print(f"  worst {e:.4f} {n}")
    zero_filled = {r["pack"].replace("{L}", "") for r in rules if "note" in r}
    bad = [(e, n) for e, n in errors if e > args.max_error and n not in zero_filled]
    for e, n in sorted(errors, reverse=True):
        if e > args.max_error:
            print(
                f"error > {args.max_error}: {e:.4f} {n}"
                + (" (documented zero-fill)" if n in zero_filled else "")
            )
    for n in missing:
        print(f"MISSING (no source in the bundle): {n}")
    for n in unsourced:
        print(f"partly or fully without a source (zero-filled): {n}")

    doc = {
        "version": 1,
        "bundle": "boltz2-int5",
        "description": "Recipes deriving the PyTorch-named Boltz-2 trunk tensors from LocalFold's int5 bundle; "
        "written by tools/map_localfold_weights.py, applied by src/trunk/localfold-weights.js.",
        "meta": {
            "kind": "boltz-trunk-localfold",
            "model": "boltz2",
            "architecture": pack_doc["meta"]["architecture"],
        },
        "aliases": ALIASES,
        "rules": [
            {k: r[k] for k in ("pack", "layers", "shape", "recipe", "note") if k in r}
            for r in rules
        ],
    }
    args.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes, {len(rules)} rules)")
    return 1 if (bad or missing) else 0


if __name__ == "__main__":
    sys.exit(main())

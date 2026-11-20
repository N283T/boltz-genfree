"""Dump PyTorch AffinityModule intermediates for WebGPU parity tests.

Runs the real ``boltz.model.modules.affinity.AffinityModule`` (structure-free
path, ``coordinate_free=True``) on a synthetic trunk output and writes
 - ``<out>/<name>-weights.{json,bin}`` : the module weights as a tensor bundle
 - ``<out>/<name>-dump.{json,bin}``    : inputs and intermediates per mode

Two weight sources:
 - ``--checkpoint PATH``  : the real ``boltz2_aff.ckpt`` (both ensemble members)
 - ``--random-small SEED``: small random-initialised modules (no checkpoint)

The trunk is NOT run here: ``s_inputs`` and ``z`` are random tensors with the
shapes and masking of the real trunk output (see webgpu/DESIGN.md). That
isolates the affinity module, which is what the WebGPU port implements.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from tensor_bundle import write_bundle

from boltz.model.modules.affinity import AffinityModule
from boltz.model.modules.affinity_distogram import STRUCTURE_FREE_AFFINITY_MODES

MODES = sorted(STRUCTURE_FREE_AFFINITY_MODES)
DUMP_LAYERS = 2  # per-layer pairformer outputs are dumped for the first N layers


def small_module(seed: int) -> AffinityModule:
    torch.manual_seed(seed)
    module = AffinityModule(
        token_s=24,
        token_z=16,
        pairformer_args={
            "num_blocks": 2,
            "pairwise_head_width": 8,
            "pairwise_num_heads": 2,
        },
        transformer_args={
            "token_s": 24,
            "num_blocks": 1,
            "num_heads": 2,
            "activation_checkpointing": False,
        },
        num_dist_bins=64,
        max_dist=22,
    )
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("norm.weight") or ".norm_in.weight" in name or ".norm_out.weight" in name or "layer_norm.weight" in name:
                param.copy_(1 + 0.1 * torch.randn_like(param))
            else:
                param.copy_(0.2 * torch.randn_like(param))
    return module.eval()


def checkpoint_modules(path: Path) -> tuple[list[AffinityModule], dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    state = ckpt["state_dict"]
    modules = []
    for idx in (1, 2):
        args = dict(hp[f"affinity_model_args{idx}"])
        args.pop("groups", None)
        pf = {k: v for k, v in args["pairformer_args"].items() if k != "use_trifast"}
        pf["activation_checkpointing"] = False
        tf = dict(args["transformer_args"])
        tf["activation_checkpointing"] = False
        module = AffinityModule(
            token_s=384,
            token_z=128,
            pairformer_args=pf,
            transformer_args=tf,
            num_dist_bins=args["num_dist_bins"],
            max_dist=args["max_dist"],
        )
        prefix = f"affinity_module{idx}."
        sub = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        module.load_state_dict(sub, strict=True)
        modules.append(module.eval())
    return modules, {"affinity_mw_correction": bool(hp.get("affinity_mw_correction", False))}


def make_inputs(token_s: int, token_z: int, n_rec: int, n_lig: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    n = n_rec + n_lig + 1  # one padding token at the end
    s_inputs = torch.randn(1, n, token_s, generator=g)
    z_trunk = 0.5 * torch.randn(1, n, n, token_z, generator=g)
    pad = torch.ones(1, n)
    pad[0, -1] = 0
    mol_type = torch.zeros(1, n, dtype=torch.long)
    mol_type[0, n_rec : n_rec + n_lig] = 3
    aff = torch.zeros(1, n, dtype=torch.long)
    aff[0, n_rec : n_rec + n_lig] = 1
    feats = {"token_pad_mask": pad, "mol_type": mol_type, "affinity_token_mask": aff}
    # trunk-side masking applied by Boltz2.forward before the affinity module
    rec = (mol_type == 0) * pad.long()
    lig = aff.bool() * pad.bool()
    cross = (
        lig[:, :, None] * rec[:, None, :]
        + rec[:, :, None] * lig[:, None, :]
        + lig[:, :, None] * lig[:, None, :]
    )
    z_aff = z_trunk * cross[:, :, :, None]
    return feats, s_inputs, z_trunk, z_aff


def run_member(module, tag, s_inputs, z_aff, feats, mode, seed, out):
    captured: dict[str, torch.Tensor] = {}
    hooks = [
        module.z_linear.register_forward_hook(
            lambda _m, _i, o: captured.__setitem__("z_linear", o.detach())
        ),
        module.pairwise_conditioner.register_forward_hook(
            lambda _m, _i, o: captured.__setitem__("conditioner_update", o.detach())
        ),
        module.pairformer_stack.register_forward_hook(
            lambda _m, _i, o: captured.__setitem__("pairformer_out", o.detach())
        ),
    ]
    for i, layer in enumerate(module.pairformer_stack.layers[:DUMP_LAYERS]):
        hooks.append(
            layer.register_forward_hook(
                lambda _m, _i, o, i=i: captured.__setitem__(f"layer{i}", o.detach())
            )
        )
    with torch.no_grad():
        res = module(
            s_inputs=s_inputs,
            z=z_aff,
            x_pred=None,
            feats=feats,
            distogram_mode=mode,
            distogram_seed=seed,
            return_distogram=True,
            coordinate_free=True,
            capture_representations=True,
        )
    for h in hooks:
        h.remove()
    prefix = f"{tag}/{mode}/"
    for k, v in captured.items():
        out[prefix + k] = v
    for k in (
        "pooled_processed",
        "pooled_conditioner_update",
        "pooled_conditioned",
        "pooled_pairformer",
        "pooled_hidden",
        "affinity_pred_value",
        "affinity_logits_binary",
        "affinity_distogram_bins",
    ):
        out[prefix + k] = res[k]
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path)
    src.add_argument("--random-small", type=int, metavar="SEED")
    parser.add_argument("--out-dir", type=Path, default=Path("webgpu/oracle-dumps"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--n-rec", type=int, default=7)
    parser.add_argument("--n-lig", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="distogram + input seed")
    args = parser.parse_args()

    if args.checkpoint is not None:
        modules, extra = checkpoint_modules(args.checkpoint)
        name = args.name or "ckpt"
        token_s, token_z = 384, 128
    else:
        modules = [small_module(args.random_small), small_module(args.random_small + 1)]
        extra = {"affinity_mw_correction": False}
        name = args.name or "small"
        token_s, token_z = 24, 16

    feats, s_inputs, z_trunk, z_aff = make_inputs(
        token_s, token_z, args.n_rec, args.n_lig, args.seed
    )
    out: dict[str, Any] = {
        "input/s_inputs": s_inputs,
        "input/z_trunk": z_trunk,
        "input/z_affinity": z_aff,
        "input/token_pad_mask": feats["token_pad_mask"],
        "input/mol_type": feats["mol_type"],
        "input/affinity_token_mask": feats["affinity_token_mask"],
    }
    summary: dict[str, Any] = {}
    for mode in MODES:
        results = [
            run_member(m, f"affinity{i + 1}", s_inputs, z_aff, feats, mode, args.seed, out)
            for i, m in enumerate(modules)
        ]
        values = torch.stack([r["affinity_pred_value"] for r in results]).mean(0)
        probs = torch.stack([torch.sigmoid(r["affinity_logits_binary"]) for r in results]).mean(0)
        out[f"final/{mode}/affinity_pred_value"] = values
        out[f"final/{mode}/affinity_probability_binary"] = probs
        summary[mode] = (float(values), float(probs))

    meta = {
        "source": "checkpoint" if args.checkpoint else f"random-small:{args.random_small}",
        "modes": MODES,
        "seed": args.seed,
        "nRec": args.n_rec,
        "nLig": args.n_lig,
        "dumpLayers": DUMP_LAYERS,
        **extra,
    }
    path = write_bundle(args.out_dir, f"{name}-dump", out, meta)

    # weights for the JS side. The checkpoint pack comes from convert_weights.py;
    # for random modules, write the same naming scheme here.
    if args.random_small is not None:
        weights = {}
        config = {}
        for i, m in enumerate(modules):
            root = f"affinity{i + 1}"
            for k, v in m.state_dict().items():
                weights[root + "/" + k.replace(".", "/")] = v
            pf = m.pairformer_stack.layers[0]
            config[root] = {
                "numDistBins": 64,
                "tokenZ": token_z,
                "tokenS": token_s,
                "headsTokenS": 24,
                "pairformerBlocks": len(m.pairformer_stack.layers),
                "triHeads": pf.tri_att_start.no_heads,
                "triHeadWidth": pf.tri_att_start.c_hidden,
                "conditionerTransitions": 2,
                "boundaries": [float(x) for x in m.boundaries],
            }
        write_bundle(
            args.out_dir,
            f"{name}-weights",
            weights,
            {"kind": "boltz-genfree-affinity-pack", "modules": config, **extra},
        )
    print(path)
    for mode, (v, p) in summary.items():
        print(f"{mode:28s} value={v:+.6f} prob={p:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()

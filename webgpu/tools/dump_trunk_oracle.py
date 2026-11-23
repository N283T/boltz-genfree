"""Dump the real GenFree PyTorch chain for one protein + ligand input.

Runs the same route ``boltz-genfree screen`` uses (``stfree_runtime``): input
parsing, optional MSA, featurisation (``ligand_reference_conformer``), the
Boltz-2 trunk, and the structure-free affinity ensemble. Writes a float32 tensor
bundle ``<out>/<name>-chain.{json,bin}`` with

  feat/*                     the featuriser outputs the WebGPU side must reproduce
  trunk/s_inputs_trunk       input embedding the trunk used (full-MSA profile)
  trunk/atom_enc_{q_in,q,c,p,bias,a}, trunk/target_feat_sum   embedder pieces
  trunk/s_init, trunk/z_init, trunk/z_init_terms/*            initial s and z terms
  trunk/{s,z}_recycle_out    recycling terms as applied in the first pass
  trunk/pairformer_in_{s,z}, trunk/msa_in_z                   exact module inputs
  trunk/msa_embed_m, trunk/msa_block{0..3}_*                  MSA module stages
  trunk/pairformer_layer{0,1,2,7,31,63}_{s,z}                 pairformer layers
  (all taps record the first recycling pass only; trunk/z, trunk/s are final)
  trunk/s_inputs_affinity    input embedding the affinity module uses
  trunk/z                    pair representation after the pairformer (unmasked)
  trunk/s                    single representation after the pairformer
  input/*                    the affinity-module inputs (same names as dump_oracle.py)
  final/<mode>/*             affinity value / probability (mean of both members)

``meta`` records the input, options and per-member outputs. Runs on the CPU in
float32 by default, which is the reference precision for the parity numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from tensor_bundle import write_bundle

from boltz.stfree_runtime import (
    StFreeRuntimeOptions,
    create_stfree_runtime_session,
    prepare_stfree_inputs,
)

TAP_LAYERS = (0, 1, 2, 7, 31, 63)

FEATURE_KEYS = (
    "token_index",
    "residue_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "mol_type",
    "res_type",
    "token_pad_mask",
    "affinity_token_mask",
    "token_bonds",
    "type_bonds",
    "profile",
    "deletion_mean",
    "profile_affinity",
    "deletion_mean_affinity",
    "pocket_feature",
    "contact_conditioning",
    "contact_threshold",
    "method_feature",
    "modified",
    "cyclic_period",
    "atom_to_token",
    "ref_pos",
    "ref_element",
    "ref_charge",
    "ref_atom_name_chars",
    "ref_space_uid",
    "ref_chirality",
    "atom_pad_mask",
    "atom_backbone_feat",
    "ref_mask",
    "msa",
    "msa_paired",
    "msa_mask",
    "has_deletion",
    "deletion_value",
)


def _store(captured: dict[str, torch.Tensor], name: str, value: torch.Tensor) -> None:
    captured.setdefault(name, value.detach())


def write_input(path: Path, sequence: str, msa: str | None, smiles: str) -> None:
    protein: dict[str, Any] = {"id": "A", "sequence": sequence}
    protein["msa"] = msa or "empty"
    payload = {
        "version": 1,
        "sequences": [{"protein": protein}, {"ligand": {"id": "B", "smiles": smiles}}],
        "properties": [{"affinity": {"binder": "B"}}],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--msa", type=Path, default=None, help="A3M for the protein")
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--record-name",
        default=None,
        help="Input record id (seeds the reference-position noise); default --name",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("webgpu/oracle-dumps"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".boltz")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--ligand-reference-conformer",
        default="skip",
        choices=["skip", "etkdg", "rdkit_2d"],
    )
    parser.add_argument("--recycling-steps", type=int, default=0)
    parser.add_argument("--accelerator", default="cpu")
    args = parser.parse_args()

    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    from boltz.main import BoltzProcessedInput, filter_inputs_affinity

    work = (args.work_dir or args.out_dir / f"{args.name}-work").resolve()
    work.mkdir(parents=True, exist_ok=True)
    input_path = work / f"{args.record_name or args.name}.yaml"
    write_input(
        input_path,
        args.sequence,
        str(args.msa.resolve()) if args.msa else None,
        args.smiles,
    )
    options = StFreeRuntimeOptions(
        cache=args.cache,
        accelerator=args.accelerator,
        num_workers=0,
        seed=args.seed,
        no_kernels=True,
        ligand_reference_conformer=args.ligand_reference_conformer,
        affinity_recycling_steps=args.recycling_steps,
    )
    run_dir = work / "run"
    prepare_stfree_inputs(input_paths=[input_path], run_dir=run_dir, options=options)
    session = create_stfree_runtime_session(options)
    model = cast(Any, session.model)
    model.eval()

    processed_dir = run_dir / "processed"
    manifest = Manifest.load(processed_dir / "manifest.json")  # ty: ignore[invalid-argument-type]
    pending = filter_inputs_affinity(
        manifest=cast(Any, manifest),
        outdir=run_dir,
        override=True,
        require_pre_affinity=False,
    )
    processed = BoltzProcessedInput(
        manifest=pending,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=None,
        template_dir=None,
        extra_mols_dir=processed_dir / "mols"
        if (processed_dir / "mols").exists()
        else None,
    )
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=session.mol_dir,
        num_workers=0,
        batch_size=1,
        constraints_dir=processed.constraints_dir,
        template_dir=None,
        extra_mols_dir=processed.extra_mols_dir,
        override_method="other",
        affinity=True,
        affinity_structure_free=True,
        affinity_structure_free_single_sequence=False,
        affinity_ligand_ref_pos_mode=options.ligand_reference_conformer,
        affinity_ligand_ref_pos_seed=(
            options.etkdg_seed
            if options.ligand_reference_conformer == "etkdg"
            else None
        ),
        affinity_random_seed=options.seed,
        affinity_require_full_input=True,
        pin_memory=False,
    )
    batch = next(iter(data_module.predict_dataloader()))
    batch = data_module.transfer_batch_to_device(batch, torch.device("cpu"), 0)

    captured: dict[str, torch.Tensor] = {}
    hooks: list[Any] = []

    def tap(module: Any, name: str, pick: Any = None) -> None:
        """Forward hook storing the first call's output (``pick`` selects a part)."""

        def hook(_m: Any, _i: Any, o: Any) -> None:
            _store(captured, name, o if pick is None else pick(o))

        hooks.append(module.register_forward_hook(hook))

    def tap_in(module: Any, name: str, index: int) -> None:
        """Forward pre-hook storing the first call's positional input ``index``."""
        hooks.append(
            module.register_forward_pre_hook(
                lambda _m, args: _store(captured, name, args[index])
            )
        )

    embedder = model.input_embedder
    tap(embedder, "s_inputs_trunk")
    tap(model.s_init, "s_init")
    tap(model.rel_pos, "rel_pos")
    tap(model.token_bonds, "token_bonds")
    tap(model.token_bonds_type, "token_bonds_type")
    tap(model.contact_conditioning, "contact_conditioning")
    tap(model.msa_module, "msa_module")
    tap(model.s_recycle, "s_recycle_out")
    tap(model.z_recycle, "z_recycle_out")
    tap_in(model.pairformer_module, "pairformer_in_s", 0)
    tap_in(model.pairformer_module, "pairformer_in_z", 1)
    tap_in(model.msa_module, "msa_in_z", 0)

    # Input embedder pieces: (q, c, p, to_keys) from the atom encoder, the atom
    # attention encoder's (a, q, ...) and the six target-feature projections.
    tap(embedder.atom_encoder, "atom_enc_q_in", lambda o: o[0])
    tap(embedder.atom_encoder, "atom_enc_c", lambda o: o[1])
    tap(embedder.atom_encoder, "atom_enc_p", lambda o: o[2])
    tap(embedder.atom_enc_proj_z, "atom_enc_bias")
    tap(embedder.atom_attention_encoder, "atom_enc_a", lambda o: o[0])
    tap(embedder.atom_attention_encoder, "atom_enc_q", lambda o: o[1])
    target_feat_modules = {
        "res_type_encoding": embedder.res_type_encoding,
        "msa_profile_encoding": embedder.msa_profile_encoding,
        "method_conditioning_init": getattr(embedder, "method_conditioning_init", None),
        "modified_conditioning_init": getattr(
            embedder, "modified_conditioning_init", None
        ),
        "cyclic_conditioning_init": getattr(embedder, "cyclic_conditioning_init", None),
        "mol_type_conditioning_init": getattr(
            embedder, "mol_type_conditioning_init", None
        ),
    }
    for key, module in target_feat_modules.items():
        if module is not None:
            tap(module, f"target_feat/{key}")

    # Pairformer layers (both tracks) and MSA blocks, first call only.
    for index in TAP_LAYERS:
        layer = model.pairformer_module.layers[index]
        tap(layer, f"pairformer_layer{index}_s", lambda o: o[0])
        tap(layer, f"pairformer_layer{index}_z", lambda o: o[1])
    for index, block in enumerate(model.msa_module.layers):
        pre = f"msa_block{index}"
        tap_in(block, f"{pre}_m_in", 1)
        tap_in(block, f"{pre}_z_in", 0)
        tap(block.pair_weighted_averaging, f"{pre}_pwa_out")
        tap(block.msa_transition, f"{pre}_trans_out")
        tap(block.outer_product_mean, f"{pre}_opm_out")
        tap(block.pairformer_layer, f"{pre}_z_after_tri", lambda o: o)
        tap(block, f"{pre}_z_out", lambda o: o[0])
    tap_in_embed = model.msa_module.msa_proj
    tap(tap_in_embed, "msa_proj_out")
    tap(model.msa_module.s_proj, "msa_s_proj_out")
    started = time.perf_counter()
    with torch.no_grad():
        out = model(
            batch,
            recycling_steps=options.affinity_recycling_steps,
            num_sampling_steps=1,
            diffusion_samples=1,
            max_parallel_samples=1,
            run_confidence_sequentially=True,
        )
    elapsed = time.perf_counter() - started
    for hook in hooks:
        hook.remove()

    tensors: dict[str, Any] = {}
    for key in FEATURE_KEYS:
        if key in batch and isinstance(batch[key], torch.Tensor):
            tensors[f"feat/{key}"] = batch[key]
    cap = captured
    tensors["trunk/s_inputs_trunk"] = cap["s_inputs_trunk"]
    s_in = cap["s_inputs_trunk"]
    z_terms = {
        "z_init_1_2": model.z_init_1(s_in)[:, :, None]
        + model.z_init_2(s_in)[:, None, :],
        "rel_pos": cap["rel_pos"],
        "token_bonds": cap["token_bonds"],
        "type_bonds": cap["token_bonds_type"],
        "contact": cap["contact_conditioning"],
    }
    for key, value in z_terms.items():
        tensors[f"trunk/z_init_terms/{key}"] = value
    tensors["trunk/z_init"] = sum(z_terms.values())
    tensors["trunk/s_init"] = cap["s_init"]
    tensors["trunk/s_recycle_out"] = cap["s_recycle_out"]
    tensors["trunk/z_recycle_out"] = cap["z_recycle_out"]
    for key in ("atom_enc_q_in", "atom_enc_c", "atom_enc_p", "atom_enc_bias"):
        tensors[f"trunk/{key}"] = cap[key]
    tensors["trunk/atom_enc_q"] = cap["atom_enc_q"]
    tensors["trunk/atom_enc_a"] = cap["atom_enc_a"]
    tensors["trunk/target_feat_sum"] = sum(
        v for k, v in cap.items() if k.startswith("target_feat/")
    )
    tensors["trunk/pairformer_in_s"] = cap["pairformer_in_s"]
    tensors["trunk/pairformer_in_z"] = cap["pairformer_in_z"]
    tensors["trunk/msa_in_z"] = cap["msa_in_z"]
    tensors["trunk/z_msa_module_out"] = cap["msa_module"]
    tensors["trunk/msa_embed_m"] = cap["msa_proj_out"] + cap[
        "msa_s_proj_out"
    ].unsqueeze(1)
    for index in range(len(model.msa_module.layers)):
        pre = f"msa_block{index}"
        m_after_pwa = cap[f"{pre}_m_in"] + cap[f"{pre}_pwa_out"]
        tensors[f"trunk/{pre}_m_after_pwa"] = m_after_pwa
        tensors[f"trunk/{pre}_m_after_trans"] = m_after_pwa + cap[f"{pre}_trans_out"]
        tensors[f"trunk/{pre}_z_after_opm"] = cap[f"{pre}_z_in"] + cap[f"{pre}_opm_out"]
        tensors[f"trunk/{pre}_z_after_tri"] = cap[f"{pre}_z_after_tri"]
        tensors[f"trunk/{pre}_z_out"] = cap[f"{pre}_z_out"]
    for index in TAP_LAYERS:
        tensors[f"trunk/pairformer_layer{index}_s"] = cap[f"pairformer_layer{index}_s"]
        tensors[f"trunk/pairformer_layer{index}_z"] = cap[f"pairformer_layer{index}_z"]
    tensors["trunk/s_inputs_affinity"] = out["affinity_context_s_inputs"]
    tensors["trunk/z"] = out["z"]
    tensors["trunk/s"] = out["s"]
    tensors["input/s_inputs"] = out["affinity_context_s_inputs"]
    tensors["input/z_trunk"] = out["z"]
    tensors["input/z_affinity"] = out["affinity_context_z"]
    for key in ("token_pad_mask", "mol_type", "affinity_token_mask"):
        tensors[f"input/{key}"] = out[f"affinity_context_{key}"]
    mode = "zero_distogram_features"
    tensors[f"final/{mode}/affinity_pred_value"] = out["affinity_pred_value"]
    tensors[f"final/{mode}/affinity_probability_binary"] = out[
        "affinity_probability_binary"
    ]
    for member in (1, 2):
        tensors[f"final/{mode}/affinity_pred_value{member}"] = out[
            f"affinity_pred_value{member}"
        ]
        tensors[f"final/{mode}/affinity_probability_binary{member}"] = out[
            f"affinity_probability_binary{member}"
        ]

    meta = {
        "sequence": args.sequence,
        "smiles": args.smiles,
        "msa": str(args.msa) if args.msa else None,
        "tokens": int(batch["token_pad_mask"].shape[1]),
        "msaRows": int(batch["msa"].shape[1]),
        "seed": args.seed,
        "ligandReferenceConformer": args.ligand_reference_conformer,
        "recyclingSteps": args.recycling_steps,
        "accelerator": args.accelerator,
        "trunkAndAffinitySeconds": elapsed,
        "affinityMwCorrection": bool(model.affinity_mw_correction),
        "mode": mode,
        "affinity_pred_value": float(out["affinity_pred_value"]),
        "affinity_probability_binary": float(out["affinity_probability_binary"]),
        "affinity_pred_value1": float(out["affinity_pred_value1"]),
        "affinity_pred_value2": float(out["affinity_pred_value2"]),
        "affinity_probability_binary1": float(out["affinity_probability_binary1"]),
        "affinity_probability_binary2": float(out["affinity_probability_binary2"]),
    }
    path = write_bundle(args.out_dir, f"{args.name}-chain", tensors, meta)
    print(json.dumps(meta, indent=1))
    print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()

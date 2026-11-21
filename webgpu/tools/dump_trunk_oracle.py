"""Dump the real GenFree PyTorch chain for one protein + ligand input.

Runs the same route ``boltz-genfree screen`` uses (``stfree_runtime``): input
parsing, optional MSA, featurisation (``ligand_reference_conformer``), the
Boltz-2 trunk, and the structure-free affinity ensemble. Writes a float32 tensor
bundle ``<out>/<name>-chain.{json,bin}`` with

  feat/*                     the featuriser outputs the WebGPU side must reproduce
  trunk/s_inputs_trunk       input embedding the trunk used (full-MSA profile)
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

TAP_LAYERS = (0, 1, 7, 31, 63)

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
    input_path = work / f"{args.name}.yaml"
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
    taps = {
        "s_inputs_trunk": model.input_embedder,
        "rel_pos": model.rel_pos,
        "token_bonds": model.token_bonds,
        "token_bonds_type": model.token_bonds_type,
        "contact_conditioning": model.contact_conditioning,
        "msa_module": model.msa_module,
    }
    hooks = [
        module.register_forward_hook(
            lambda _m, _i, o, name=name: _store(captured, name, o)
        )
        for name, module in taps.items()
    ]
    for index in TAP_LAYERS:
        layer = model.pairformer_module.layers[index]
        hooks.append(
            layer.register_forward_hook(
                lambda _m, _i, o, index=index: _store(
                    captured, f"pairformer_layer{index}_z", o[1]
                )
            )
        )
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
    tensors["trunk/s_inputs_trunk"] = captured["s_inputs_trunk"]
    s_in = captured["s_inputs_trunk"]
    z_init = (
        model.z_init_1(s_in)[:, :, None]
        + model.z_init_2(s_in)[:, None, :]
        + captured["rel_pos"]
        + captured["token_bonds"]
        + captured["token_bonds_type"]
        + captured["contact_conditioning"]
    )
    tensors["trunk/z_init"] = z_init
    tensors["trunk/z_msa_module_out"] = captured["msa_module"]
    for index in TAP_LAYERS:
        tensors[f"trunk/pairformer_layer{index}_z"] = captured[
            f"pairformer_layer{index}_z"
        ]
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

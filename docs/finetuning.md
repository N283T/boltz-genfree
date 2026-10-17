# Optional GenFree fine-tuning: integration boundary

Status: versioned adapter import/inspection and screening integration are
implemented and CPU-tested, with a bounded two-target public-CLI GPU deployment smoke. Native context extraction and a general training
command remain pending. No `finetune` command or universal performance guarantee
is advertised.

GenFree describes per-candidate structure-generation-free inference. Released
weights are the no-label default. With target labels, affinity-module adaptation
can be optional without turning normal generated-pose features into candidate
inputs. The initial prototype updates both complete affinity modules on detached
GenFree contexts while leaving the trunk frozen.

## Implemented adapter interface

```sh
boltz-genfree adapter import-heads --checkpoint final_heads.pt \
  --metadata metadata.json --output adapter_dir
boltz-genfree adapter inspect adapter_dir
boltz-genfree screen screen.yaml --prepare-only
```

The schema-v1 directory contains `manifest.json` and `weights.pt`. The latter
contains only a tensor `state_dict`, loaded with `weights_only=True`; manifest
and weight hashes protect identity. The metadata input needs base checkpoint
SHA256, target name/sequence SHA256, `feature_route`, `receptor_context`, and
training seed/split/plan hashes with frozen-upstream independent-GenFree context
declarations. It must declare zero distogram and no candidate structure
sampling. The inspect command exposes the complete manifest.

`feature_route` pins recycling, conformer/conditioner mode, MW correction, MSA
subsampling/cap, input seed and ETKDG seed. For reference-cropped inputs,
`receptor_context` pins the reference-file SHA256 and crop-config SHA256: hash
the exact `target.crop` mapping excluding `reference`, serialized with sorted
keys and JSON separators `(',', ':')`. Changing chain/ligand selectors, budget
or reserved ligand tokens is rejected, even if the reference file is unchanged.
Target sequence hashing uses the normalized uppercase whitespace-free sequence.

Use `options.affinity_adapter` alongside the compatible full base checkpoint.
Model tensor names, shapes and dtypes for both affinity modules are validated
before either module is modified. The full base file is never rewritten.
Adapter changes reject resume, and a loaded adapted session cannot be used as a
no-FT session. Trusted Lightning checkpoint loading does not override the
adapter loader's explicit restricted-unpickler setting.

CPU validation includes the existing real NS5/GSK3B weight files (520 tensors
each), actual architecture settings from the SHA-checked released checkpoint,
and one previously saved independent context per target. Adapter-loaded and
original-weight head forwards were bitwise equal on these CPU inputs. At product
revision `9ae7e6f`, full public-CLI CUDA/kernel-enabled FT scoring passed for two
original-library inputs each on NS5/GSK3B (PBS2523982, exit0). Geometry-contract
hooks verified zero distogram, no generated coordinates, pruned structure
modules and all 520 adapter tensors. The preceding load failure (PBS2523931)
was retained and repaired by treating Lightning's unspecified `weights_only=None`
as the trusted full-checkpoint default, while preserving explicit `True` for
restricted adapter loads. This deployment smoke is not a new accuracy benchmark,
unseen-target guarantee or GPU numerical-equivalence claim.

## Remaining product work

1. Implemented: a versioned, head-only adapter bundle with the base checkpoint SHA256,
   architecture/tensor-shape compatibility, target identity, exact feature route,
   input/training seeds, label split hashes, final-weight hash and observed
   training settings. Do not silently treat a full checkpoint and an adapter as
   interchangeable.
2. Implemented: an explicit screening option for an adapter. Validate it before model
   loading/scoring, reject incompatible bundles, and put its identity into resume
   provenance. The current `affinity_checkpoint` option expects a full checkpoint;
   research `final_heads.pt` is supported through explicit conversion, not as a full checkpoint.
3. Pending: add GenFree-native, coordinate-free context extraction. Keep structure
   sampling off; never accept a generated-control pose/crop/tensor as an
   independent candidate context. Record receptor-selection and ligand-local
   geometry choices.
4. Pending: add a bounded training command with explicit label input, unique IDs,
   train/monitor/evaluation disjointness, seed, optimizer and stopping policy.
   Exclude training IDs from reported performance. Report a 300-label pool
   separately from the number of unique replacement draws actually used.
5. CPU fixtures and real-head replay passed, including module isolation, adapter
   round-trip, incompatible-base rejection and resume rejection when the
   adapter changes. GPU pilot/generalization studies require separate explicit
   experiment approval and existing-result checks.
6. Evaluate paired no-FT/FT scores from the same fresh GenFree contexts. Report
   AP, AUROC, enrichment and fixed-budget recovery together; adaptation can
   improve early recovery while worsening AUROC.

## Evidence and interpretation

The current research cases used 300-label pools, one training seed each, five
fixed epochs and final raw weights. Both improved AP/top-100 recovery; GSK3B
AUROC decreased. These were deliberately chosen targets, not a representative
sample. Author train-ID selection used normal Boltz rankings, so label acquisition
was not entirely generation-free even though candidate feature construction was.
Compound bootstrap intervals do not measure repeated-training or unseen-target
uncertainty. No-FT timing ratios exclude adaptation cost.

The frozen reproduction runtime and original pretrained checkpoint remain
unchanged. New product integration should use narrow hooks rather than migrating
the entire target-specific research controller or its scheduler assumptions.

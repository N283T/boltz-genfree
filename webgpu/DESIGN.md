# WebGPU Boltz-GenFree: design and reuse boundary

GenFree scores a protein-ligand pair without generating a structure: the Boltz-2
trunk runs, then the **affinity module** (two ensemble members) runs on the trunk
output with a coordinate-free distogram. This directory ports the affinity module
to WebGPU and defines how it connects to a WebGPU Boltz-2 trunk.

## Where the boundary is

```
                         (trunk side, NOT ported here)                  (this directory)
 tokens + MSA + ligand  ───────────────────────────────►  s_inputs [N,384]
 featurisation             Boltz-2 trunk, 0 recycles      z        [N,N,128]   ─►  AffinityGpu.score()
                                                          masks    [N]              ├ member 1 (8 blocks)
                                                                                    ├ member 2 (4 blocks)
                                                                                    └ mean of value / sigmoid(logit)
```

`Boltz2.forward` (src/boltz/model/models/boltz2.py, around line 880) hands the
affinity module exactly this, and nothing else in the structure-free path:

| tensor | shape | note |
|---|---|---|
| `s_inputs` | `[N, 384]` | `input_embedder(feats, affinity=True)`: uses `profile_affinity` / `deletion_mean_affinity`, **not** the profile the trunk used |
| `z` | `[N, N, 128]` | trunk pair output, multiplied by the protein-ligand pair mask before the module |
| `token_pad_mask`, `mol_type`, `affinity_token_mask` | `[N]` | derive receptor / ligand masks |

No coordinates, no `s`, no structure module and no confidence head are used.
`webgpu/src/trunk-input.js` is the one place that implements this contract
(`input/s_inputs`, `input/z_trunk`, masks, with the cross-pair masking applied in
JS); `webgpu/tools/dump_oracle.py` writes tensors under the same names.

## What is reused from LocalFold, and what is not

LocalFold (`LOCALFOLD_DIR`, never modified, nothing copied) provides a WebGPU
Boltz-2 trunk: `webgpu/af3/trunk/` (embedder, template, MSA stack, 64-block
pairformer, `Af3TrunkGpu`), `webgpu/af3/fold.js` (`foldBatch`, which builds the
batch from `shared/af3/featurise`, including ligands from SMILES via
`shared/chem/`), kernels in `webgpu/kernels/`, the `boltz2` weight bundle
(`shared/bundles/manifests/boltz2.js`, int5, 24 shards, ~382 MB, fetched over
HTTP) and the tensor-store format in `shared/bundles/`.

| piece | decision |
|---|---|
| Trunk (`s_inputs`-side embedder, MSA, pairformer, `z`) | **Reused via `LOCALFOLD_DIR`**, wired in `src/trunk/`. |
| Device creation | **Reused when available**: with `LOCALFOLD_DIR` set, `src/device.js` calls LocalFold's `requestAlphaFoldDevice` (features, raised limits). Otherwise a plain device with the adapter's maximum limits. |
| Weight pack layout | **Compatible**: the converter writes LocalFold's tensor-manifest layout (`{formatVersion, tensors:{name:{file,byteOffset,dtype,shape}}}`, read by `FileTensorStore`), float32. Not quantised: the module is 8.7 M parameters (35 MB). |
| Pair-track kernels (triangle mult., triangle attention, transition) | **Not reused; written new** (`src/gpu/wgsl.js`). Reason below. |
| Distogram modes, pooling, heads, ensemble | New. |

Why the pair-track kernels are new. LocalFold's `compilePairTrack` /
`encodePairTrack` / `Af3PairformerStackGpu` are heavily tuned for the AF3 pairformer
(128-channel pair, 4 heads of 32, int5-decoded resident weights, f16 staging,
chunking, device-profile tuning, pipeline-key conventions). They also assume a
pairformer block that carries a single track; the pair-only paths in that code are
PairMixer (no triangle attention) and ESMFold2, not Boltz's `PairformerNoSeq`
(triangle multiplications + both triangle attentions + transition, no single
track). Driving them from a bundle of Boltz-layout tensors means reproducing their
weight-source/thunk machinery and dialect flags, which is a larger and riskier
integration than the module warrants and was not done blind. The new kernels are
plain f32, one generic kernel per op, and match PyTorch at float32 round-off, so
they are the correctness baseline. Swapping the pair track for LocalFold's tuned
kernels is the main performance follow-up (see below).

## What is new in this directory

- `tools/convert_weights.py`: `affinity_module1.*` / `affinity_module2.*` from
  `boltz2_aff.ckpt` to a pack (`/`-separated names, config in `meta`).
- `tools/dump_oracle.py`: runs the real `AffinityModule` (`coordinate_free=True`)
  and dumps inputs and intermediates (post-`z_linear`, conditioner update,
  per-layer pairformer outputs, pooled stages, final outputs) per mode.
- `src/distogram.js`: the five structure-free modes, generated on the JS side
  (`uniform_random` uses an MT19937 stream and reproduces torch's bins exactly;
  `normal_random` uses Box-Muller and is *not* bit-identical to `torch.randn`).
- `src/reference.js`: an independent CPU implementation, also checked against
  the oracle, so GPU and CPU can be told apart from the oracle.
- `src/gpu/`: WGSL kernels and `AffinityGpu` (distance-bin embedding,
  `s_to_z_prod` projections, pairwise conditioner, PairformerNoSeq stack,
  protein-ligand pair pooling, affinity heads, ensemble).
- `src/cli.js`, `web/index.html`, `tools/serve.js`: entry points for the full chain (and for scoring one trunk dump).

## Module recap (what the kernels implement)

```
z  = z_linear(LN(z_affinity)) + W1·s[:,None] + W2·s[None,:]
d  = embed(bins)              # zero_distogram_features: zeros (GenFree default)
z += conditioner([z | d])     # LN(2c) -> Linear -> 2 x (z += SwiGLU(z)); skipped in zero_conditioning
repeat blocks: tri_mul_out, tri_mul_in, tri_att_start, tri_att_end, transition (4x), all residual,
               masked by the protein-ligand pair mask
g  = pool(z)                  # mean over protein-ligand and ligand-ligand pairs, diagonal excluded
value, logit = MLP heads(g)   # members averaged: value mean, probability = mean(sigmoid(logit))
```

## Trunk wiring (done) and open items

The trunk is wired in `src/trunk/` (see docs/webgpu.md for numbers):

- `localfold.js` imports LocalFold's featuriser, weight loaders, `Af3TrunkGpu` and `buildTargetFeat`
  from `LOCALFOLD_DIR` (a directory in Node, a served URL in a browser).
- `featurise.js` builds LocalFold's batch for GenFree's route (ligand from SMILES, `skip` conformer, bond
  type + 1, A3M parsing like `a3m.py`, size limits) and the affinity `profile_affinity` variant.
- `canonical.js` replaces RDKit for ligand order and names (exact for non-stereo molecules).
- `weights.js` opens a trunk weight set (the f32 pack from `tools/convert_trunk_weights.py`, or LocalFold's
  int5 bundle) and applies the two route adaptations (method row, zeroed template output).
- `run.js` is `foldHolding`'s trunk part: embedding, MSA draw, trunk passes and recycles; `chain.js` ties
  the trunk to `AffinityGpu`.

The three questions the earlier draft left open: (a) LocalFold's boltz2 weights are the same Boltz-2 weights
as `boltz2_aff.ckpt` (value-for-value up to int5 round-off; the checkpoint also carries the trunk), the f32 pack
removes the round-off; (b) `profile_affinity` is the one-hot of the query residue types with zero deletion mean
(`process_msa_features(..., max_seqs_batch=1, affinity=True)`), built in `affinityBatch`; (c) ligand features:
`skip` zeroes ligand `ref_pos`, and atom order and names come from RDKit's canonical SMILES and ranks.

Remaining gaps: protein `ref_pos` (Boltz rotates RDKit conformers with a seeded torch RNG; not reproducible
here, GenFree's own seed noise is ~0.02 in value), stereo ligands, the MSA convention differences above, kernel
performance of the affinity module beyond ~200 tokens (naive f32 kernels; LocalFold's tuned pair kernels are the
follow-up), and checks on GPUs other than the Apple M4.

# WebGPU Boltz-GenFree: design and reuse boundary

GenFree scores a protein-ligand pair without generating a structure: the Boltz-2
trunk runs, then the **affinity module** (two ensemble members) runs on the trunk
output with a coordinate-free distogram. This directory ports the whole chain to
WebGPU: featuriser, Boltz-2 trunk and affinity module, all owned here.

## Where the module boundary is (the contract between trunk and affinity module)

```
                         (src/trunk/own/)                               (src/gpu/)
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

## Reuse boundary: everything is owned

Every stage of the chain is implemented in this repository; nothing is imported from LocalFold at build or run time:

| piece | where |
|---|---|
| Featuriser (tokens, atoms, MSA, windows, ligand graph) | `src/trunk/own/{featurise,msa,windows,ligand,boltz-const,conformers}.js`, `src/trunk/canonical.js` |
| Trunk (embedder, recycling, MSA module, pairformer) | `src/trunk/own/{embedder,msa-module,pairformer,run}.js` (+ `*-wgsl.js`) |
| Chain | `src/trunk/own/chain.js` (`createOwnChain`) |
| Weights | PyTorch-named float32 pack (`tools/convert_boltz_trunk.py`), `src/trunk/own-weights.js`, lazy Range reads in the browser |
| Affinity module, distogram modes, pooling, heads, ensemble | `src/gpu/`, `src/distogram.js`, `src/reference.js` |
| Device creation | `src/device.js`: a plain device with the adapter's maximum limits, no optional features |

LocalFold (https://github.com/sokrypton/localfold, Beer-Ware licence) is a reference only. It was consulted to confirm
the Boltz-2 dialect flags of the MSA module and pairformer, and its protein conformer table was the first source of
`src/trunk/own/conformers.js` (attributed in that file). The bundle manifest layout (`{formatVersion, tensors:{name:{file,
byteOffset,dtype,shape}}}`) follows LocalFold's tensor store. Every kernel was written against Boltz's PyTorch source
and checked against GenFree's PyTorch run.

Numbers: with the oracle's atom features the own trunk's z agrees with PyTorch to relRMS ~1e-6 (the earlier
LocalFold-backed "f32" trunk was ~1e-3). Own chain, Apple M4: 124 tokens ~6 s, 211 tokens ~23 s, 395 tokens ~127 s
(LocalFold's tuned kernels: 211 tokens 8.3 s total). Naive triangle attention dominates; the pair-track kernels are
plain f32, one generic kernel per op (`src/gpu/wgsl.js`, `src/trunk/own/pairformer-wgsl.js`), the correctness baseline.

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

## Trunk (done) and open items

The trunk lives in `src/trunk/own/` (numbers in docs/webgpu.md):

- `featurise.js` builds the Boltz-2 `feat/*` tensors for GenFree's route (ligand from SMILES, `skip` conformer, A3M
  parsing like `a3m.py`, size limits, `profile_affinity` variant); `canonical.js` replaces RDKit for ligand order and
  names (exact for non-stereo molecules).
- `embedder.js`, `msa-module.js`, `pairformer.js`, `run.js` are Boltz2.forward up to the trunk outputs; weights stream
  per module, and for a URL pack each block / layer is fetched by HTTP Range just before use and released after.
- `chain.js` ties the trunk to `AffinityGpu`.

Remaining gaps: protein `ref_pos` (Boltz rotates RDKit conformers with a seeded torch RNG; not reproducible here,
GenFree's own seed noise is ~0.02 in value), stereo ligands, the dropped `etkdg` / `rdkit_2d` conformer modes, CCD
ligands and receptor cropping, speed (naive triangle attention; LocalFold's tuned kernels are 2.8x faster at 211
tokens), the browser path (verified once on one machine; the affinity pack is not streamed), and checks on GPUs other
than the Apple M4.

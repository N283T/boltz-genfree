# WebGPU chain (experimental)

`webgpu/` runs Boltz-GenFree's structure-free route end to end on WebGPU, in Node (Dawn,
the `webgpu` package) and in a browser. Everything is implemented in this repository:

```
sequence (+A3M), SMILES -> own featuriser -> own embedder -> recycling (0 by default)
                        -> own MSA module (4 blocks) -> own pairformer (64 blocks)
                        -> structure-free affinity module -> value, P(binder)
```

Design: [webgpu/DESIGN.md](../webgpu/DESIGN.md). [LocalFold](https://github.com/sokrypton/localfold)
(WebGPU AlphaFold 3 / Boltz-2, Beer-Ware licence) was consulted as a reference for the Boltz-2 dialect and
its protein conformer table was the first source of `src/trunk/own/conformers.js`; no LocalFold code or
checkout is used at build or run time.

## Setup

```bash
# weights GenFree itself uses (2,062,139,170 bytes, sha256 dcc5cd37...ca9e)
#   https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt -> ~/.boltz/boltz2_aff.ckpt
uv run python webgpu/tools/convert_weights.py        # affinity module -> webgpu/packs/genfree-affinity.*          (35 MB)
uv run python webgpu/tools/convert_boltz_trunk.py    # Boltz-2 trunk   -> webgpu/packs/boltz2-trunk-own/*          (805 MB, float32, PyTorch names)
cd webgpu && npm install                             # `webgpu` (Dawn node binding)
```

Both packs are float32 tensor bundles (a JSON manifest plus one `.bin`). Packs and oracle dumps are git-ignored.

## Scoring

```bash
cd webgpu
node src/cli.js --sequence QLEDSEV... --smiles 'Oc1ccccc1' [--a3m target.a3m] [--recycles 0] [--json]
node src/cli.js --trunk oracle-dumps/ckpt-dump.json [--cpu]    # affinity module only, from a stored trunk output
node tools/serve.js && open http://127.0.0.1:4180/             # browser page
```

Options: `--trunk-pack manifest.json|URL` (default `packs/boltz2-trunk-own/boltz2-trunk-own.json`), `--pack`
(affinity pack), `--allow-extended`, `--ligand-atoms ligand.json`, `--mode` (structure-free distogram mode,
default `zero_distogram_features` as in GenFree). Only the GenFree default ligand conformer (`skip`, ligand
`ref_pos` zeroed) exists.

Example, phenol with a 117-residue protein and no MSA (Apple M4, Metal):

```
mode         zero_distogram_features  (124 tokens: 117 receptor + 7 ligand)
pred value   1.278146        (GenFree PyTorch: 1.293631)
P(binder)    0.334760        (0.328219)
timings      featurise 5 ms, trunk 5212 ms (embed 29, msa 345, pairformer 4815), affinity 881 ms, total 6098 ms
```

### Screening a library (same config as `boltz-genfree screen`)

```bash
cd webgpu
node src/cli.js screen ../examples/screen.yaml --prepare-only    # validate the inputs
node src/cli.js screen ../examples/screen.yaml                    # writes scores.csv, screen_status.json
```

The config is the GenFree `screen.yaml`: `target.sequence`, `target.msa` (an A3M path, or `empty`), `ligands.csv|tsv|smiles|ccd`
(`id_column`, `smiles_column`, `delimiter`), `output_dir`, `chunk_size` and `options.{seed, affinity_recycling_steps,
ligand_reference_conformer, allow_extended_full_input}`. `scores.csv` has GenFree's columns. A re-run resumes from
`.webgpu-records.jsonl`; a changed target, MSA, ligand list or option is rejected. The trunk runs once per ligand
(about 2 s each for a 70-residue target and 6 s for 117 residues on an Apple M4). On `examples/screen.yaml` with `msa: empty`, values agree with
GenFree (float32, `accelerator: mps`) to 0.004 (glycine) and 0.0004 (phenol). GenFree's default `accelerator: gpu` runs bf16 on a Mac and differs more.

Deep MSA check (CTDSP1, 180 residues, 14,343-row A3M subsampled to 1024 rows, first 8 ligands of the paper's library, config
`balanced`/`minimal` of the full screen): GenFree on MPS float32 reproduces the stored CUDA scores to mean |d| 0.006, so the
reference is sound. WebGPU differs from GenFree by mean |d| 0.05-0.17 (max 0.57). GenFree itself moves by 0.08-0.19 between
seeds on the same input, because the 1024-row subsample and the protein `ref_pos` rotation are random and not reproducible
in JS, so the WebGPU values lie within GenFree's own seed spread. Rank-level agreement on a larger library was not measured. Not supported yet, and rejected or noted explicitly: `target.crop`
(reference-complex cropping; use `--receptor-indices` on the single-pair CLI), `msa: auto`, which is also the default when `msa` is omitted (scored without an MSA, with a
note; write `msa: empty` to say so explicitly), `ligands.sdf`, `affinity_adapter` / `affinity_checkpoint`, and `options` outside the list above (rejected with an error).
`affinity_conditioner_mode: folded` is accepted and computed as `original`; `subsample_msa` / `num_subsampled_msa` must keep GenFree's defaults.

### Browser page

`tools/serve.js` serves `webgpu/` with HTTP Range support. The page opens the trunk pack with
`OwnWeights.fromUrl(url, {lazy: true})`: only the manifest is fetched up front, and the embedder (about 6 MB), each
MSA block and each pairformer layer are fetched by byte range right before use (`ensure(prefix)`) and dropped after
(`release(prefix)`), so the 805 MB pack is never resident in JS memory. The affinity pack (35 MB) is fetched whole.
In the desktop app's built-in browser (Chrome 156, Apple Metal) the default input scored 1.27815, the same as Node,
in 8.3 s of compute with 82 Range requests (805 MB streamed once) and a peak JS heap of about 150 MB.

### What matches GenFree

* Route: one protein chain + one ligand, `override_method="other"`, no templates, no constraints,
  `ligand_reference_conformer: skip`, affinity input embedding from the one-hot query profile, the
  `crop_full_screen` size limits (receptor <= 200, ligand <= 128, total <= 256 tokens, <= 2048 atoms;
  `--allow-extended` lifts all but the ligand limit; nothing is cropped silently), A3M parsing as
  `boltz.data.parse.a3m` (duplicates dropped, 8192 rows; deeper alignments are subsampled to 1024 rows per pass with a
  seeded JS shuffle, not torch's stream).
* Ligand preparation as RDKit does it: `webgpu/src/trunk/canonical.js` reproduces `standardize()` + canonical SMILES
  atom order + canonical-rank atom names in pure JS (no RDKit). It matched RDKit on all 17,491 neutral non-stereo and
  1,040 charged non-stereo molecules of a 44,602-molecule CCD corpus (`tools/check_canonical.mjs`). Stereo molecules
  are not exact: produce the atoms with RDKit (`uv run python webgpu/tools/ligand_atoms.py 'SMILES' > ligand.json`,
  then `--ligand-atoms ligand.json`).

### Parity (Apple M4, Node/Dawn, float32; `BOLTZ_GENFREE_GPU_TESTS=1 node --test test/own-chain.test.js`)

PyTorch reference: GenFree's own pipeline on CPU float32 (`tools/dump_trunk_oracle.py`). "exact" imports the oracle's
protein reference positions and atom-name one-hots; "standalone" is everything from the own featuriser.

| input | trunk z relRMS (exact) | value diff, exact | value diff, standalone |
|---|---|---|---|
| phenol, 117 aa, no MSA (124 tokens) | 7.8e-7 | 5e-7 | 0.0155 |
| acetaminophen, same protein | 8.0e-7 | 0 | 0.0048 |
| salicylic acid | 7.8e-7 | 2e-7 | 0.0051 |
| phenol + MSA (75 rows, 143 tokens) | 1.1e-6 | 2.5e-6 | 0.047 |
| phenol, `recycling_steps=1` | 1.8e-6 | < 1e-6 | |
| acetaminophen, 200 aa (211 tokens) | 3.5e-3 (standalone) | | 1e-4 |

With the oracle's atom features the chain agrees to float32 round-off (z relRMS ~1e-6, 64 pairformer blocks
deep). The previous LocalFold-backed "f32" trunk was at ~1e-3 (cause not investigated), so this chain is about three
orders of magnitude closer to PyTorch.

The standalone featuriser cannot be exact for one reason: Boltz-2 rotates and translates every protein residue's
reference conformer with a per-record seeded torch RNG, and that stream is not reproduced (protein `ref_pos` random
rotation). GenFree itself moves by that noise when only its seed changes: phenol, seeds 1..6 give values 1.249 ...
1.309 (std ~0.02); standalone results stay within that spread.

### Timings and memory (Apple M4, 32 GB, float32, cold process)

| tokens | total | notes |
|---|---|---|
| 124 (117 + 7) | 6-11 s | pairformer ~4.8 s, MSA module 0.3 s, affinity 0.9 s |
| 211 (200 + 11) | ~23 s | pairformer ~19 s |
| 395 (384 + 11, `--allow-extended`) | ~127 s | |

Peak process RSS 0.25-0.4 GB (weights are read tensor by tensor). The LocalFold-backed trunk with its tuned kernels
needed 8.3 s total for 211 tokens, so the own chain is about 2.8x slower there. The dominant cost is the naive
triangle attention kernel (one generic f32 kernel); tuned/tiled attention and the f16 paths are the performance
follow-up. `node tools/bench_own_chain.js --case NAME | --long 384` reproduces the timings.

### Remaining gaps and what is not verified

* The `etkdg` / `rdkit_2d` ligand conformer modes (`ligandConformer`, `--ligand-conformer`) use the built-in embedder below; `skip` (GenFree's default) is the exact one. The embedded coordinates are not RDKit's, and the effect on the affinity value has not been measured against GenFree.
* CCD ligands (`ligandKind: "ccd"`, `smiles` = the code, `--ligand-ccd CODE`; the component is fetched from
  files.rcsb.org at run time) use the dictionary's heavy-atom order, `atom_id` names, charges and bond types
  (`pdbx_aromatic_flag` -> AROMATIC; a quadruple bond -> OTHER). Not verified against a Boltz oracle dump. Gaps: Boltz
  loads RDKit molecules from its own CCD pickle (aromaticity and charges come from RDKit's perception, ours from the
  dictionary flags), `ref_chirality` is 0 (Boltz uses the RDKit chiral tag), atom names longer than four characters are
  rejected, multi-residue CCD ligands are not supported.
* Receptor cropping (`receptorIndices`, `--receptor-indices 0-99,120`: sorted 0-based residues of the full sequence):
  only those residues are featurised, each keeps its original `residue_index` (as Boltz's crop; `token_index` is
  renumbered 0..T-1), the A3M is checked and de-duplicated on the full chain and then cut to the kept columns
  (insertions go with the column that follows), and the size limits apply to the cropped size. Boltz's own crop
  *selection* (spatial / contiguous sampling) is not reimplemented: the caller supplies the indices.
* Protein `ref_pos` random rotation (above); stereo ligands without `--ligand-atoms`.
* Speed against LocalFold's tuned kernels (above). Token limits: a pair of N tokens x 128 channels must fit
  `maxStorageBufferBindingSize` (N <= ~1000-2000 depending on device).
* Browser: verified once, on the same Apple M4 in the desktop app's built-in browser, one input. Not verified: other
  GPUs/OSes/browsers (NVIDIA, Windows, Safari, Firefox), hosted (non-local) serving, inputs above 256 tokens in the
  browser, recycling > 0 in the browser, multi-chain or non-protein targets, `affinity_mw_correction` beyond the
  formula in `cli.js`. The affinity module in the browser is not streamed (35 MB resident).

## Oracles and tests

```bash
uv run python webgpu/tools/dump_oracle.py --checkpoint ~/.boltz/boltz2_aff.ckpt     # affinity module on synthetic trunk output
uv run python webgpu/tools/dump_trunk_oracle.py --sequence SEQ --smiles SMILES [--msa x.a3m] --name NAME
cd webgpu && npm test                          # CPU tests; GPU / oracle tests skip visibly
cd webgpu && BOLTZ_GENFREE_GPU_TESTS=1 npm test   # adds the WebGPU tests (needs packs and oracle dumps)
```

Dumps and packs are git-ignored.

### Ligand conformers (`etkdg`, `rdkit_2d`): own embedder

`webgpu/src/trunk/own/conformer.js` (with `chem/bounds.js`) is a self-contained, LocalFold-free replacement for the
former LocalFold conformer modes; `featuriseOwn` calls it when `ligandConformer` is `etkdg` or `rdkit_2d`. `embedConformer(graph, {seed})` runs an ETKDG-like distance-geometry embedding on the
heavy-atom graph (hydrogens implicit): hybridisation-based bond lengths and angles (ring-size aware), 1-4 cis/trans
windows (fixed cis in planar 5/6/7 rings), triangle smoothing, random distances, 4D metric-matrix embedding, L-BFGS
against the bounds, then a 3D refinement with planarity terms at sp2 centres and in planar rings.
`ligandConformerPositions(graph, mode, seed)` applies Boltz's centring plus a seeded random rotation and translation
(`rigidMotion`); `rdkit_2d` first flattens onto the principal plane. `skip` stays with the caller.
Measured on phenol, acetaminophen, naphthalene, indole, cyclohexane, an ester and a 56-heavy-atom peptide: worst bond
length error 0.9% of ideal, worst angle error 5.3 degrees, planar rings within 0.01 A. Limits: no stereochemistry
(chiral centres and E/Z get whatever configuration the embedding finds), no torsional preferences beyond the 1-4
window, no RDKit parity (coordinates and the torch random stream differ, as for the previous LocalFold conformers).

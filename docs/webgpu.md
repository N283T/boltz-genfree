# WebGPU chain (experimental)

`webgpu/` runs Boltz-GenFree's structure-free route end to end on WebGPU, in Node (Dawn,
the `webgpu` package) and in a browser:

```
sequence (+A3M), SMILES -> featurisation -> Boltz-2 trunk (LocalFold, 0 recycles by default)
                        -> structure-free affinity module (this repo) -> value, P(binder)
```

The trunk is LocalFold's WebGPU Boltz-2 port, referenced through `LOCALFOLD_DIR` and never
copied or modified. Design and the reuse boundary: [webgpu/DESIGN.md](../webgpu/DESIGN.md).

## Setup

```bash
# weights GenFree itself uses (2,062,139,170 bytes, sha256 dcc5cd37...ca9e)
#   https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt -> ~/.boltz/boltz2_aff.ckpt
uv run python webgpu/tools/convert_weights.py          # affinity module  -> webgpu/packs/genfree-affinity.*   (35 MB)
uv run python webgpu/tools/convert_trunk_weights.py    # Boltz-2 trunk    -> webgpu/packs/boltz2-trunk-f32/   (770 MiB, float32)
cd webgpu && npm install                               # `webgpu` (Dawn node binding)
export LOCALFOLD_DIR=/path/to/localfold                # a LocalFold checkout (read-only use)
```

`convert_trunk_weights.py` writes the trunk of `boltz2_aff.ckpt` (the file GenFree loads the
whole model from) in LocalFold's boltz2 tensor layout. The layout rules were derived by
matching values against LocalFold's published `boltz2-int5` bundle;
`LOCALFOLD_DIR=... node webgpu/tools/compare_trunk_pack.mjs packs/boltz2-trunk-f32 <int5 dir>`
re-checks all 182 non-template tensors (median relRMS 0.036, worst 0.054, i.e. int5 round-off;
norms and biases exact). So LocalFold's Boltz-2 weights are the same weights as GenFree's.

Optional smaller weights: LocalFold's own int5 bundle, `https://huggingface.co/sokrypton/localfold/resolve/651a25ba8a57c6d18caacf50ab7180b255eb5a82/boltz2-int5/`
(`manifest.json` + 24 shards, 382,207,764 bytes, shard digests verified against LocalFold's compiled
manifest; downloaded here to `~/.cache/boltz-genfree-webgpu/localfold-boltz2-int5`, outside the repo). Pass it with
`--trunk-weights`. It carries about 4 % per-tensor weight round-off (see the table below).

## Scoring

```bash
cd webgpu
node src/cli.js --sequence QLEDSEV... --smiles 'Oc1ccccc1' [--a3m target.a3m] [--recycles 0] [--json]
node src/cli.js --trunk oracle-dumps/ckpt-dump.json [--cpu]    # affinity module only, from a stored trunk output
LOCALFOLD_DIR=... node tools/serve.js && open http://127.0.0.1:4180/   # browser page
```

Options: `--trunk-weights DIR|URL` (default `packs/boltz2-trunk-f32`), `--precision f32|auto`
(`auto` = f16 where the device has it: ~10 % faster, trunk z relRMS ~0.9 % instead of 0.1 %),
`--allow-extended`, `--ligand-atoms ligand.json`, `--ligand-conformer skip|conformer`, `--mode`
(structure-free distogram mode, default `zero_distogram_features` as in GenFree).

### What matches GenFree

* Route: one protein chain + one ligand, `override_method="other"`, no templates, no constraints,
  `ligand_reference_conformer: skip` (ligand `ref_pos` zeroed), affinity input embedding from the
  one-hot query profile (`profile_affinity`), the `crop_full_screen` size limits (receptor <= 200,
  ligand <= 128, total <= 256 tokens, <= 2048 atoms; `--allow-extended` lifts all but the ligand limit;
  nothing is cropped silently), A3M parsing as `boltz.data.parse.a3m` (duplicates dropped, 8192 rows).
* Ligand preparation as RDKit does it: `webgpu/src/trunk/canonical.js` reproduces `standardize()`
  + canonical SMILES atom order + canonical-rank atom names in pure JS (no RDKit). It matched RDKit
  on all 17,491 neutral non-stereo and 1,040 charged non-stereo molecules of a 44,602-molecule corpus
  built from the CCD (`tools/check_canonical.mjs`). Stereo molecules are **not** exact (names agree for
  ~20 %): the result is flagged in the output notes; produce the atoms with RDKit instead
  (`uv run python webgpu/tools/ligand_atoms.py 'SMILES' > ligand.json`, then `--ligand-atoms ligand.json`).
* Two LocalFold conventions are adapted (`src/trunk/weights.js`, `src/trunk/featurise.js`): LocalFold's
  bond-order plane is one lower than Boltz's `type_bonds` (+1 applied) and its method row is X-ray (row 4,
  "other", is used); its template module always runs, so its output projection is zeroed (the checkpoint has
  none).

### Parity (Apple M4, Node/Dawn, f32 trunk, `tools/check_chain.js`)

PyTorch reference: GenFree's own pipeline on CPU float32 (`tools/dump_trunk_oracle.py`, seed 20260718).

| input | features | trunk z relRMS | value (WebGPU / PyTorch) | P(binder) |
|---|---|---|---|---|
| phenol, 117 aa, no MSA | exact PyTorch atom features | 8.5e-4 | 1.29277 / 1.29363 | 0.32823 / 0.32822 |
| acetaminophen, same protein | exact | 8.2e-4 | 1.98065 / 1.98065 | 0.13269 / 0.13268 |
| salicylic acid | exact | 1.0e-3 | 1.33440 / 1.33386 | 0.27226 / 0.27198 |
| acetaminophen, 200 aa (211 tokens) | exact | 7.2e-4 | 2.35798 / 2.35818 | 0.25388 / 0.25384 |
| phenol + MSA (seq2.a3m, 136 aa, has X) | exact | 1.6e-2 | 1.37323 / 1.37862 | 0.2469 / 0.2448 |
| phenol, 117 aa | standalone JS featuriser | 3.5e-2 | 1.2722 / 1.2936 | 0.3378 / 0.3282 |
| salicylic, non-canonical SMILES | standalone | | 1.3368 / 1.3384 | 0.2652 / 0.2600 |
| acetaminophen, 200 aa | standalone | | 2.3712 / 2.3582 | 0.2507 / 0.2538 |
| phenol (int5 trunk weights, f16) | exact | 1.5e-1 | 1.2849 / 1.2936 | 0.3084 / 0.3282 |
| acetaminophen 200 aa (int5, f16) | exact | 5.5e-2 | 2.2989 / 2.3582 | 0.2679 / 0.2538 |

Reading it. With the very same atom features the whole chain agrees to |value diff| <= 1e-3
(P(binder) to 3e-4) in float32; the trunk is 64 pairformer blocks deep and still lands at z relRMS
~1e-3, the remainder being f32 summation order. The browser page (built-in browser, same Metal GPU)
gave 1.2731 for phenol with the standalone featuriser against 1.2722 in Node.

The standalone featuriser cannot be exact for one reason: Boltz-2 rotates and translates every
residue's reference conformer with a per-record seeded torch RNG, from RDKit-computed conformers that
differ from LocalFold's CCD coordinates (pairwise distances differ by up to 3.5 A). GenFree itself
moves by that noise when only its seed changes: phenol, seeds 1..6 and the default give values
1.249 ... 1.309 (std ~0.02). Standalone results are within that spread (largest observed difference
0.013 on the 200-aa case; phenol 0.021). Use `check_chain.js --features exact` to see the engine
without this noise.

The MSA case is looser (z 1.6 %, value 0.005): LocalFold keeps the query row in both its paired and its
unpaired block (76 rows against 75 in Boltz), and the example's two `X` residues are tokenised with other
atoms than Boltz's UNK. Boltz's seeded MSA row subsampling (> 1024 rows) is not reproduced: LocalFold's own
draw is used.

### Timings and memory (Apple M4, 32 GB, f32, one cold process; weights load 0.6-0.8 s included separately)

| tokens | trunk | affinity (naive kernels) | total | peak RSS |
|---|---|---|---|---|
| 124 (117 + 7) | 1.5 s | 0.9 s | 2.5 s | |
| 211 (200 + 11) | 4.4 s | 3.5 s | 8.3 s | 3.1 GB |
| 395 (384 + 11, MSA, `--allow-extended`) | 14.5 s | 19.1 s | 34.0 s | |

The affinity module's own kernels are plain f32 and dominate beyond ~200 tokens (DESIGN.md). Browser page,
124 tokens: trunk 2.1 s, affinity 1.0 s, 0.84 s weight load (f32 pack served locally).

### LocalFold-free chain (`webgpu/src/trunk/own/chain.js`)

`createOwnChain({trunkPack, affinityPack, device})` runs own featuriser -> embedder -> recycling -> MSA module ->
pairformer -> affinity module (no LocalFold, no `LOCALFOLD_DIR`); test: `BOLTZ_GENFREE_GPU_TESTS=1 node --test
test/own-chain.test.js`, timing/memory: `node tools/bench_own_chain.js --case NAME | --long 384`. With the oracle's atom
features the trunk z agrees to relRMS ~1e-6 and the value to <= 3e-6 (MSA included). The weight pack is read tensor by
tensor from the file (no whole-pack copy): peak RSS 0.25-0.4 GB. It is about 2.5x slower than the LocalFold trunk.

### Limits and what is not verified

* Token limits are GenFree's (above). The GPU limit is LocalFold's: a pair of N tokens x 128 channels must fit
  `maxStorageBufferBindingSize` (N <= ~1000-2000 depending on device); the f32 trunk weights are 770 MiB resident.
  The 770 MiB pack is a local file; for a hosted page use LocalFold's int5 bundle (382 MB) and accept its round-off.
* Verified: parity numbers above, on one Apple M4 / Metal; the browser page on the same machine in the desktop
  app's built-in browser. Not verified: other GPUs/OSes, other browsers, NVIDIA, recycling > 0 against PyTorch,
  the `etkdg`/`rdkit_2d` ligand conformer modes (not implemented; `--ligand-conformer conformer` uses LocalFold's
  JS conformer and is not comparable), protein inputs with unknown residues, multi-chain or non-protein targets,
  stereo ligands without `--ligand-atoms`, `affinity_mw_correction` beyond the formula in `cli.js`.
* Possible LocalFold issue found on the way (not changed there): its boltz2 bond-order plane is Boltz's
  `type_bonds` minus one for ligand bonds.

## Oracles and tests

```bash
uv run python webgpu/tools/dump_oracle.py --checkpoint ~/.boltz/boltz2_aff.ckpt     # affinity module on synthetic trunk output
uv run python webgpu/tools/dump_trunk_oracle.py --sequence SEQ --smiles SMILES [--msa x.a3m] --name NAME
cd webgpu && LOCALFOLD_DIR=... npm test    # 17 tests; chain tests skip without LOCALFOLD_DIR / packs / dumps
LOCALFOLD_DIR=... node tools/check_chain.js --dump oracle-dumps/NAME-chain.json --features exact|js
```

Dumps and packs are git-ignored.

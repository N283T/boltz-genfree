# WebGPU affinity scoring (experimental)

`webgpu/` holds a WebGPU implementation of the structure-free affinity module of
Boltz-GenFree: distance-bin embedding, pairwise conditioner, the PairformerNoSeq
stack, protein-ligand pair pooling, the affinity heads and the two-member
ensemble, plus the structure-free distogram modes. It runs in Node (the `webgpu`
package, Dawn) and in a browser. Design and the trunk boundary:
[webgpu/DESIGN.md](../webgpu/DESIGN.md).

What it is not: it does not run the Boltz-2 trunk. It scores trunk outputs
(`s_inputs`, `z`, token masks). Connecting LocalFold's WebGPU trunk is open work
(see DESIGN.md); LocalFold is referenced through `LOCALFOLD_DIR`, never copied
or modified.

## Setup

```bash
# weights: the standard GenFree/Boltz cache (2.1 GB, sha256 dcc5cd37...ca9e)
#   https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt -> ~/.boltz/boltz2_aff.ckpt
uv run python webgpu/tools/convert_weights.py          # -> webgpu/packs/genfree-affinity.{json,bin} (35 MB)
cd webgpu && npm install                               # `webgpu` (Dawn node binding), dev dependency
```

Optional: `export LOCALFOLD_DIR=/path/to/localfold` to request the device through
LocalFold's `requestAlphaFoldDevice`.

## Oracle and tests

```bash
uv run python webgpu/tools/dump_oracle.py --random-small 7                    # no checkpoint needed
uv run python webgpu/tools/dump_oracle.py --checkpoint ~/.boltz/boltz2_aff.ckpt
uv run python webgpu/tools/dump_oracle.py --checkpoint ~/.boltz/boltz2_aff.ckpt \
    --n-rec 120 --n-lig 10 --modes zero_distogram_features --name ckpt131
cd webgpu && npm test        # CPU reference, distogram and WebGPU-vs-oracle parity (GPU test skips without an adapter)
```

The oracle runs the real `AffinityModule` on a synthetic trunk output (random
`s_inputs` and `z` with the real shapes, masking and weights); dumps and packs
are git-ignored.

## Scoring

```bash
cd webgpu
node src/cli.js --pack packs/genfree-affinity.json --trunk oracle-dumps/ckpt-dump.json
node src/cli.js --cpu ...          # CPU reference instead of WebGPU
```

`--trunk` is a tensor bundle with `input/s_inputs`, `input/z_trunk`,
`input/token_pad_mask`, `input/mol_type`, `input/affinity_token_mask`. The default
mode is `zero_distogram_features`, which is what GenFree uses; the other
structure-free modes are `uniform_random`, `normal_random`, `uniform_mean_embedding`
and `zero_conditioning`. `web/index.html` is the browser entry point (untested in a
browser so far).

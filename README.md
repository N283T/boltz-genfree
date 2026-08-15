# Boltz-StFree

Structure-free Boltz-2 affinity screening without complex structure generation.

Boltz-StFree runs the released Boltz-2 affinity trunk and heads directly from
processed receptor and ligand inputs. It skips structure sampling, replaces the
local affinity distogram with coordinate-independent zero features, and writes
compact scores suitable for recoverable virtual screens.

This repository is an independent product fork. The research runtime used by
the associated manuscript remains frozen separately in
[`N283T/boltz2-affinity-interventions-boltz`](https://github.com/N283T/boltz2-affinity-interventions-boltz),
while the paper, analyses, and result artifacts remain in the associated
research repository. Changes made here do not alter those reproduction
artifacts.

## Status

`boltz-stfree screen` is the primary interface. It accepts one protein target,
a ligand library, output and chunking settings, and prediction options in one
YAML file. It generates ordinary Boltz affinity inputs, runs recoverable
structure-free chunks, and aggregates the compact JSON results into one
`scores.csv`. The lower-level `predict` command remains available for runtime
compatibility and research interventions.

Until the first tagged release, install from a pinned Git commit and expect the
interface to evolve.

The method requires a relevant receptor context. When the receptor does not fit
inside one affinity crop, provide a validated external crop. A structure-free
score is not evidence that Boltz generated or evaluated a physical ligand pose.

## Install

Python 3.10 or newer is required. Install the repository in an isolated
environment; the package is not yet published on PyPI.

```bash
git clone https://github.com/N283T/boltz-stfree.git
cd boltz-stfree
uv venv
uv pip install ".[cuda]"
```

This distribution installs `boltz-stfree`, not the upstream `boltz` command,
so the runtime used for a screen is explicit.

## Screen a ligand library

Create a screen config containing one protein and one ligand source:

```yaml
version: 1

target:
  sequence: YOUR_PROTEIN_SEQUENCE
  msa: ./target.a3m  # or auto / empty

ligands:
  csv: ./library.csv
  id_column: id
  smiles_column: smiles

output_dir: ./screen-output
chunk_size: 1000

options:
  seed: 20260718
  num_workers: 2
  affinity_recycling_steps: 0
  affinity_batch_size: 1
```

Then run:

```bash
boltz-stfree screen screen.yaml
```

Use `--prepare-only` to validate the config and inspect the generated Boltz YAML
chunks without loading checkpoints or running inference:

```bash
boltz-stfree screen screen.yaml --prepare-only
```

The generated inputs and immutable manifest live under
`<output_dir>/.boltz-stfree/`; per-chunk runtime artifacts live under
`<output_dir>/runs/`. Re-running the same config resumes through the existing
Boltz output checks. A changed config or ligand source is rejected rather than
mixed into existing screen state.

With `target.msa: auto`, `screen` calls the configured MSA server exactly once
for the target and records a checksum beside the shared MSA. Every ligand and
chunk then references that same file; automatic MSA generation is never
repeated per compound. `--prepare-only` does not make the network request.

### Ligand sources

`ligands` must contain exactly one of the following sources.

Inline SMILES:

```yaml
ligands:
  smiles:
    - id: compound-1
      smiles: CCO
    - id: compound-2
      smiles: CCN
```

An SDF library, using the molecule title or an explicit property as its ID:

```yaml
ligands:
  sdf: ./library.sdf
  id_property: compound_id
```

CCD components:

```yaml
ligands:
  ccd:
    - ATP
    - id: adp-reference
      ccd: ADP
```

CSV and TSV libraries accept configurable `id_column`, `smiles_column`, and
one-character `delimiter` fields. All source paths are resolved relative to the
screen config.

### Prediction options

Keys below `options` may use `snake_case` or CLI-style hyphenated names and are
validated against the existing `predict` command. Relative path options are
also resolved from the screen config directory.

The `screen` command manages these route-defining values and rejects different
overrides:

```text
model = boltz2
affinity_structure_free = true
affinity_structure_free_fast = true
affinity_screening_minimal_output = true
affinity_distogram_mode = zero_distogram_features
sampling_steps_affinity = 1
diffusion_samples_affinity = 1
```

Affinity recycling defaults to 0 but remains configurable from 0 through 5.
CUDA kernels remain enabled unless `options.no_kernels` is true. The paper's
complete 49,980-compound ALR screen used batch size 1, an MSA, ETKDG ligand
reference coordinates, and the other managed settings above.

Minimal output writes one score JSON per record plus shared run metadata. It
suppresses structure files, full distogram artifacts, and Lightning logs.

## Receptor crop requirements

The affinity crop can contain at most 256 tokens, including at most 128 ligand
tokens and at most 200 receptor tokens. If the complete receptor fits, the
default sequence-prefix crop can retain it. For longer receptors, supply
validated receptor-token indices and a provenance label:

```yaml
options:
  affinity_crop_indices_dir: ./crops
  affinity_structure_free_crop_label: known-pocket-v1
```

`screen` rejects targets longer than 200 residues unless this external crop is
configured. Crop files must follow the existing per-record runtime format.

Do not interpret an arbitrary sequence-prefix crop as pocket discovery. The
paper found that moving the receptor crop away from the relevant site can
damage screening even though removing compound-specific pose generation does
not.

## Measured scope

On one NVIDIA GeForce RTX 5080, the measured route completed 49,980 ALR
compounds in 18,290.09 seconds (2.733 compounds/s) and wrote 51.9 MB. A matched
16-compound operational benchmark was 11.08 times faster end to end than the
default generated-pose route. These measurements establish feasibility for the
tested hardware, target, inputs, and settings; they are not general performance
or accuracy guarantees.

Experimental modes such as all-zero ligand reference coordinates, affinity
batching, query-only inputs, and MSA-module bypass remain available for research
but are not the recommended default screening route.

## Provenance

The code history is preserved rather than squashed:

1. [Boltz](https://github.com/jwohlwend/boltz) is the official upstream.
2. [boltz-community](https://github.com/Novel-Therapeutics/boltz-community)
   supplied the compatibility and maintenance base.
3. [boltz2-affinity-interventions-boltz](https://github.com/N283T/boltz2-affinity-interventions-boltz)
   added the causal interventions and measured structure-free runtime.
4. Boltz-StFree starts from that runtime and develops the screening tool as an
   independent repository.

Checkpoint fallback URLs still refer to the Boltz and boltz-community model
hosts because this project does not redistribute the released model weights.

The inherited code is licensed under the [MIT License](LICENSE), and existing
copyright notices are retained. See [CITATION.cff](CITATION.cff) for the
software citation placeholder pending final manuscript metadata.

## Development safety

The only writable GitHub target for this repository is
`N283T/boltz-stfree`. Repository instructions prohibit pull requests and
writable remotes targeting the research runtime, boltz-community, or official
Boltz repositories. Package publishing is also disabled: the included workflow
builds and validates artifacts manually but cannot upload them to PyPI.

Run the CPU-focused tests with:

```bash
pytest -m "not slow and not regression" -q
```

Generic input, training, and prediction documentation inherited from the
runtime is available under [`docs/`](docs/).

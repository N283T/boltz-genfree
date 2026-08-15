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

`boltz-stfree screen` is the only prediction interface. It accepts one protein
target and a ligand library, expands the library into independent 1:1
protein-ligand records, runs recoverable structure-free chunks, and aggregates
the compact JSON results into one `scores.csv`.

Until the first tagged release, install from a pinned Git commit and expect the
interface to evolve.

The method requires a relevant receptor context. `screen` only accepts records
whose complete receptor and ligand fit inside one affinity crop. A
structure-free score is not evidence that Boltz generated or evaluated a
physical ligand pose.

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

### Runtime options

Keys below `options` may use `snake_case` or CLI-style hyphenated names.
Boltz-StFree deliberately exposes only settings that do not change the
structure-free method:

```text
cache, affinity_checkpoint
devices, accelerator, num_workers, preprocessing_threads
seed, override
max_msa_seqs, subsample_msa, num_subsampled_msa
no_kernels, flash_attn
affinity_mw_correction, affinity_recycling_steps
msa_server_url, msa_pairing_strategy and MSA authentication settings
```

Every run uses Boltz-2, the affinity checkpoint, complete-input receptor
selection, ETKDG ligand reference coordinates, zero distogram features, one
record per affinity batch, one nominal diffusion sample, no structure
sampling, and compact output. These route-defining values cannot be overridden.
Affinity recycling defaults to 0 but remains configurable from 0 through 5.
CUDA kernels remain enabled unless `options.no_kernels` is true.

Minimal output writes one score JSON per record plus shared run metadata. It
suppresses structure files, full distogram artifacts, and Lightning logs.

## Full-screen eligibility

`screen` validates every library record before generating inputs or starting a
GPU run. A record passes only when all three conditions hold:

```text
target_tokens <= 200
ligand_tokens <= 128
target_tokens + ligand_tokens <= 256
```

Small-molecule ligand tokens are counted after the same standardization and
hydrogen removal used by Boltz affinity parsing. Ligands from 57 through 128
tokens are accepted with a warning because 56 was the maximum during training.
Any hard-limit failure names the affected ligand and stops the whole screen;
`screen` does not crop or silently skip it. The runtime verifies after
tokenization that every token was retained and fails if an input would be
truncated.

CCD inputs are checked against the Boltz-2 component cache. A normal `screen`
run downloads that cache when needed. Offline `--prepare-only` validation
requires an existing cache, selected with `options.cache` or `BOLTZ_CACHE`.

## Measured scope

On one NVIDIA GeForce RTX 5080, the measured route completed 49,980 ALR
compounds in 18,290.09 seconds (2.733 compounds/s) and wrote 51.9 MB. A matched
16-compound operational benchmark was 11.08 times faster end to end than the
default generated-pose route. These measurements establish feasibility for the
tested hardware, target, inputs, and settings; they are not general performance
or accuracy guarantees.

Generated crops, external crops, complex structure sampling, structure-derived
distograms, structure files, Boltz-1, multi-chain complexes, and research
interventions are intentionally not exposed by this product runtime. They remain
available only in the separate research/reproduction repository.

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
The screen runtime downloads the Boltz-2 affinity checkpoint and component
dictionary, but not the structure checkpoint.

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

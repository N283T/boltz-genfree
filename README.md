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

The method requires a relevant receptor context. A short receptor can be used
in full; a longer receptor can be selected from an experimental or otherwise
supplied reference complex. The reference is used only to choose receptor
sequence blocks. It does not provide a ligand pose or a structure-derived
distogram to the model.

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
  crop:              # optional when the complete receptor fits
    reference: ./reference-complex.cif
    reference_protein: A
    reference_ligand: L
    budget: per_ligand

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
  ligand_reference_conformer: skip
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
`<output_dir>/runs/`. Re-running the same input and preprocessing configuration
resumes through provenance-checked outputs. A changed target, ligand source,
component cache, MSA source, or preprocessing limit is rejected rather than
mixed into existing screen state. Operational options such as `override` may be
changed safely.

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
Boltz-StFree keeps the method surface deliberately small:

```text
cache, affinity_checkpoint
devices, accelerator, num_workers, preprocessing_threads
seed, override
max_msa_seqs, subsample_msa, num_subsampled_msa
no_kernels, flash_attn
affinity_mw_correction, affinity_recycling_steps
ligand_reference_conformer, etkdg_seed
msa_server_url, msa_pairing_strategy and MSA authentication settings
```

Every run uses Boltz-2, the affinity checkpoint, complete-input receptor
selection or a reference-derived receptor crop, zero distogram features, one
record per affinity batch, one nominal diffusion sample, no structure
sampling, and compact output. Ligand conformer generation defaults to `skip`;
the ligand reference-position tensor is explicitly zeroed in this mode.
For comparison runs, deterministic ETKDGv3 plus UFF can be enabled with:

```yaml
options:
  ligand_reference_conformer: etkdg
  etkdg_seed: 42
```

The mode and effective ETKDG seed are recorded in run provenance, so outputs
from different conformer conditions cannot be resumed together. SDF records
are canonicalized to SMILES; their supplied coordinates are not reused.
Affinity recycling defaults to 0 but remains configurable from 0 through 5.
CUDA kernels remain enabled unless `options.no_kernels` is true.

Minimal output writes one score JSON per record plus shared run metadata. It
suppresses structure files, full distogram artifacts, and Lightning logs.

## Receptor selection and limits

`screen` validates every library record before generating inputs or starting a
GPU run. Without `target.crop`, a record passes only when all three conditions
hold:

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

For a longer receptor, set `target.crop.reference` to a PDB or mmCIF complex.
The observed sequence of `reference_protein` must exactly match `target.sequence`.
When `reference_ligand` is an author chain ID and that chain contains exactly
one non-polymer residue, the residue is selected automatically. If the chain
contains multiple candidates, specify one explicitly:

```yaml
reference_ligand:
  chain: L
  residue: 501
  ccd: ATP
```

The ligand may share an author chain ID with the protein. Zero or multiple
matches fail during preparation. Ligands represented by multiple or branched
residues, insertion codes, and alternate locations are intentionally
unsupported; preprocess the reference to one unambiguous ligand residue and
conformer first. Receptor residues are ranked by their minimum distance to that
residue's heavy atoms and added as contiguous sequence blocks using the
existing affinity crop behavior
(`neighborhood_size: 10`). The complete screened ligand is always reserved
first and is never truncated.

Two budget policies are supported:

- `per_ligand` (default) applies the 256-token, 2048-atom, and 200-protein-token
  limits separately for each ligand. All records share the same reference block
  priority, but smaller ligands may retain more receptor blocks.
- `library_fixed` reserves capacity for the largest ligand in the library and
  uses exactly the same receptor crop for every record.

The reference checksum and chain selection are stored in the screen manifest.
Each record also stores its receptor token indices, reserved ligand size, crop
policy, and retained receptor-token count under `.boltz-stfree/crops/`. These
files are included in resume provenance and summarized in `scores.csv`.

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

Arbitrary external token crops, complex structure sampling, structure-derived
distograms, structure files, Boltz-1, multi-chain prediction inputs, and
research interventions are intentionally not exposed by this product runtime.
They remain available only in the separate research/reproduction repository.

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
pytest -q
```

Generic structure prediction, training, and intervention workflows are outside
this product's scope. Their documentation and implementation remain in the
[separate research/reproduction repository](https://github.com/N283T/boltz2-affinity-interventions-boltz).

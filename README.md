# Boltz-GenFree

Structure-generation-free Boltz-2 affinity screening.

Boltz-GenFree (GenFree) runs the released Boltz-2 affinity trunk and heads directly from
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

`boltz-genfree screen` is the structure-free screening interface. It accepts one
protein target and a ligand library, expands the library into independent 1:1
protein-ligand records, runs recoverable structure-free chunks, and aggregates
the compact JSON results into one `scores.csv`. When no suitable holo reference
exists for receptor selection, the separate `bootstrap-reference` command can
predict one reusable reference from the target and an explicitly supplied
anchor ligand.

Until the first tagged release, install from a pinned Git commit and expect the
interface to evolve.

The method requires a relevant receptor context. A short receptor can be used
in full; a longer receptor can be selected from an experimental or otherwise
supplied reference complex. The reference is used only to choose receptor
sequence blocks. It does not provide a ligand pose or a structure-derived
distogram to the model.

## Naming and compatibility

GenFree means **structure-generation-free per-candidate screening**, not freedom
from all geometric or structural prior information. External receptor references,
MSA inputs and ligand-local coordinates remain explicit input choices. The
optional `bootstrap-reference` operation generates a reusable structure and must
be reported separately.

The repository and distribution are now `boltz-genfree`. The legacy commands
`boltz-stfree` and `boltz-stfree-fix-macos-libomp` remain compatibility aliases.
Existing `.boltz-stfree/` state directories, serialized runtime identifiers,
record-seed namespace and internal `boltz.stfree_runtime` API are deliberately
unchanged to preserve artifact identities and per-record random streams.
Resume still requires an exact runtime-code fingerprint: pre-migration runs must
be resumed with their original pinned revision, not silently mixed with this
revision. Command aliases do not bypass provenance checks. Local checkout
directories do not need to be renamed.

## Optional fine-tuning: current boundary

The default GenFree screen uses released pretrained affinity weights and needs
no target labels. With labels, target-specific affinity-module fine-tuning is
an optional extension, not a prerequisite for GenFree. The research prototype
has been evaluated on NS5 and GSK3B with a 300-label pool per target, detached
GenFree contexts and the two complete affinity modules updated while the trunk
remains frozen. AP and top-100 recovery increased on both targets, but GSK3B
AUROC decreased. These are exploratory results, not guarantees for other targets.

**Adapter import/loading is CPU-tested and has passed a bounded two-target GPU deployment smoke; the training command is
not yet integrated.** Convert research head-only weights with
`boltz-genfree adapter import-heads --checkpoint final_heads.pt --metadata metadata.json --output adapter_dir`,
and verify them with `boltz-genfree adapter inspect adapter_dir`. The conversion
uses CPU only, never loads a trunk and refuses to overwrite existing files.
Set `options.affinity_adapter: /path/to/adapter_dir` for screening; retain a
compatible full base checkpoint in `options.affinity_checkpoint` (or the default
released checkpoint). Do not pass raw `final_heads.pt` as a full checkpoint.

The adapter pins the base checkpoint SHA256, target sequence, exact feature
route, receptor reference/crop configuration and declared training split/seed.
Wrong target, changed route, malformed/nonfinite tensors or mismatched weights
are rejected. Adapter identity is included in the screen plan and strict resume
provenance. Only both complete affinity modules are replaced; the trunk and base
checkpoint file are unchanged. Metadata documents declared provenance; hashes
alone do not prove that arbitrary user-provided training provenance is true.

Existing NS5/GSK3B weights have passed CPU conversion and cached-context head
replay against their original serialization with zero output difference.
Full public-CLI CUDA/kernel-enabled scoring passed for two original-library inputs per target at product revision `9ae7e6f` (PBS2523982). This is not a full-screen accuracy benchmark or a general GPU parity guarantee. There is still
no `boltz-genfree finetune` command: native context extraction and the bounded
label-safe trainer remain next steps. See the [integration boundary](docs/finetuning.md).

## Install

Python 3.10 or newer is required. Install the repository in an isolated
environment; the package is not yet published on PyPI.

```bash
git clone https://github.com/N283T/boltz-genfree.git
cd boltz-genfree
uv sync --frozen --extra cuda
source .venv/bin/activate
```

This distribution installs `boltz-genfree`, not the upstream `boltz` command,
so the runtime used for a screen is explicit.

The committed `.python-version` and `uv.lock` reproduce the tested Python and
package set. The default `cuda` extra matches PyTorch's CUDA 13 wheels; systems
that deliberately use a CUDA 12 PyTorch build can select `--extra cuda12`
instead. For development, add `--extra test --extra lint`. Update the lock only
as an explicit maintenance action with `uv lock --upgrade`.

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
  affinity_batch_size: 1
  affinity_recycling_steps: 0
  ligand_reference_conformer: skip
```

Then run:

```bash
boltz-genfree screen screen.yaml
```

Use `--prepare-only` to validate the config and inspect the generated Boltz YAML
chunks without loading checkpoints or running inference:

```bash
boltz-genfree screen screen.yaml --prepare-only
```

Preparation also writes `screen_plan.json`. It records the target and MSA mode,
receptor-selection policy, ligand count and token range, chunk sizes, execution
settings, checkpoint path, and expected outputs. The same summary is printed to
the terminal, making `--prepare-only` a reviewable dry run before a large
screen.

The generated inputs and immutable manifest live under
`<output_dir>/.boltz-stfree/`; unified runtime artifacts live under
`<output_dir>/runs/`. Logical chunks are combined into one `process_inputs`
call, one processed manifest, and one prediction pass. The model is loaded once
after that complete preprocessing pass. Re-running the same input and
preprocessing configuration resumes through provenance-checked outputs. A
changed target, ligand source, component cache, MSA source, or preprocessing
limit is rejected rather than mixed into existing screen state. Operational
options such as `override` may be changed safely.

Set `options.seed` for reproducible screens. Boltz-GenFree derives independent
record-local seeds for reference-feature augmentation and MSA subsampling, so a
record receives the same score when chunk boundaries, worker scheduling, or a
resumed run change. Setting `seed: null` leaves those operations stochastic.

During execution, `screen_status.json` is atomically updated after every
prediction batch. It remains readable from another terminal and includes the
screen state, completed/failed/pending counts, percentage, current chunk, and
per-chunk counts. For example:

```bash
watch -n 1 cat screen-output/screen_status.json
```

The final states are `complete`, `failed`, or `interrupted`; a failed or
interrupted screen retains its completed record count for diagnosis and resume.

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
Boltz-GenFree keeps the method surface deliberately small:

```text
cache, affinity_checkpoint, affinity_adapter
devices, accelerator, num_workers, affinity_batch_size, preprocessing_threads
seed, override
max_msa_seqs, subsample_msa, num_subsampled_msa
no_kernels, flash_attn
affinity_mw_correction, affinity_recycling_steps
ligand_reference_conformer, etkdg_seed, affinity_conditioner_mode
allow_extended_full_input
msa_server_url, msa_pairing_strategy and MSA authentication settings
```

With a fixed `seed`, MSA rows are selected per record, independently of batch
size and chunk order. Batched MSA subsampling is supported.

Every run uses Boltz-2, the affinity checkpoint, complete-input receptor
selection or a reference-derived receptor crop, zero distogram features, one
or more records per affinity batch, one nominal diffusion sample, no structure
sampling, and compact output. Ligand conformer generation defaults to `skip`;
the ligand reference-position tensor is explicitly zeroed in this mode.
For comparison runs, deterministic ETKDGv3 plus UFF can be enabled with:

```yaml
options:
  ligand_reference_conformer: etkdg
  etkdg_seed: 42
```

Alternatively, `ligand_reference_conformer: rdkit_2d` generates ligand-local
2D coordinates without ETKDG. `affinity_conditioner_mode: folded` analytically
preserves LayerNorm over the full `[pair, zero]` width and both learned
affine/projection halves, while avoiding the zero-distogram embedding and
concatenation. The default is `original`; the altered-width `narrow` variant is
not supported. Folded equivalence is numerical, not guaranteed bitwise equality.

The modes and effective ETKDG seed are recorded in run provenance, so outputs
from different conformer conditions cannot be resumed together. SDF records
are canonicalized to SMILES; their supplied coordinates are not reused.
Affinity recycling defaults to 0 but remains configurable from 0 through 5.
CUDA kernels remain enabled unless `options.no_kernels` is true.

Minimal output writes one score JSON per record plus shared run metadata. It
suppresses structure files, full distogram artifacts, and Lightning logs.

## Receptor selection and limits

`screen` validates every library record before generating inputs or starting a
GPU run. By default, without `target.crop`, a record passes only when all three conditions
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

For an explicitly uncropped, extended full-receptor run, use:

```yaml
target:
  msa: empty  # query-only; the MSA module still runs
options:
  allow_extended_full_input: true
  affinity_recycling_steps: 0
  ligand_reference_conformer: skip
  affinity_conditioner_mode: folded
```

This opt-in removes receptor/combined-token and atom crop budgets, not the
128-token ligand limit. The runtime must retain every input token. It cannot
be combined with `target.crop`; memory use depends on the full input size.
Default limits remain unchanged when the option is absent.

Alternatively, set `target.crop.reference` to a PDB or mmCIF complex.
For mmCIF references, the declared polymer sequence of `reference_protein` must
exactly match `target.sequence`. Coordinate-observed residues may be incomplete;
their `_atom_site.label_seq_id` values map ligand-distance seeds back to the
declared full-sequence positions. References without declared polymer mapping,
such as plain PDB files, still require the observed sequence to exactly match
`target.sequence`.
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

### Bootstrap a missing holo reference

Reference bootstrapping is an explicit, separate operation so `screen` and
`screen --prepare-only` never start structure generation implicitly. Supply a
target sequence and one anchor ligand:

```yaml
version: 1

target:
  id: A
  sequence: YOUR_PROTEIN_SEQUENCE
  msa: auto  # or empty / a local MSA path

anchor:
  id: B
  smiles: YOUR_ANCHOR_SMILES

output_dir: ./target-reference

options:
  seed: 20260718
  diffusion_samples: 1
  recycling_steps: 3
  sampling_steps: 200
```

Stage and review the exact input without loading a checkpoint or using a GPU:

```bash
boltz-genfree bootstrap-reference bootstrap.yaml --prepare-only
```

After review, omit `--prepare-only` to run the single Boltz-2 holo prediction.
The highest-confidence diffusion sample is published as `reference.pdb` with a
checkpoint checksum, reference checksum, selected confidence, and chain IDs in
`bootstrap_result.json`. Re-running an unchanged completed bootstrap reuses the
reference; changed configuration is rejected against the immutable manifest.

Use the result as an ordinary crop reference. Chain `A` is the protein and
chain `B` is the unambiguous anchor ligand in this example:

```yaml
target:
  sequence: YOUR_PROTEIN_SEQUENCE
  crop:
    reference: ./target-reference/reference.pdb
    reference_protein: A
    reference_ligand: B
    budget: library_fixed
```

The anchor is a user-supplied pocket-localization input. Selecting a potent
compound is one possible policy, not a requirement or an inference performed
by the tool.

## Measured scope

On one NVIDIA GeForce RTX 5080, the measured route completed 49,980 ALR
compounds in 18,290.09 seconds (2.733 compounds/s) and wrote 51.9 MB. A matched
16-compound operational benchmark was 11.08 times faster end to end than the
default generated-pose route. These measurements establish feasibility for the
tested hardware, target, inputs, and settings; they are not general performance
or accuracy guarantees.

Arbitrary external token crops, structure-derived distograms, Boltz-1,
multi-chain screening inputs, and research interventions are intentionally not
exposed by this product runtime. Complex structure generation is limited to the
explicit single-target/single-anchor reference-bootstrap operation above.
They remain available only in the separate research/reproduction repository.

## Provenance

The code history is preserved rather than squashed:

1. [Boltz](https://github.com/jwohlwend/boltz) is the official upstream.
2. [boltz-community](https://github.com/Novel-Therapeutics/boltz-community)
   supplied the compatibility and maintenance base.
3. [boltz2-affinity-interventions-boltz](https://github.com/N283T/boltz2-affinity-interventions-boltz)
   added the causal interventions and measured structure-free runtime.
4. Boltz-GenFree starts from that runtime and develops the screening tool as an
   independent repository.

Checkpoint fallback URLs still refer to the Boltz and boltz-community model
hosts because this project does not redistribute the released model weights.
The screen runtime downloads the Boltz-2 affinity checkpoint and component
dictionary. The structure checkpoint is downloaded only by an explicitly
started `bootstrap-reference` run, never by preparation or screening.

The inherited code is licensed under the [MIT License](LICENSE), and existing
copyright notices are retained. See [CITATION.cff](CITATION.cff) for the
software citation placeholder pending final manuscript metadata.

## Development safety

The only writable GitHub target for this repository is
`N283T/boltz-genfree`. Repository instructions prohibit pull requests and
writable remotes targeting the research runtime, boltz-community, or official
Boltz repositories. Package publishing is also disabled: the included workflow
builds and validates artifacts manually but cannot upload them to PyPI.

Run the CPU-focused tests with:

```bash
pytest -q
```

Generic structure prediction and arbitrary intervention workflows are outside
this product's scope. Target-specific affinity-module fine-tuning is an optional
GenFree feature under integration, not a general-purpose training interface. Their documentation and implementation remain in the
[separate research/reproduction repository](https://github.com/N283T/boltz2-affinity-interventions-boltz).

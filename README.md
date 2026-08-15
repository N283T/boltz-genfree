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

Boltz-StFree currently exposes the measured structure-free route through the
`boltz-stfree predict` command and explicit options. A shorter screening-first
command is planned. Until the first tagged release, install from a pinned Git
commit and expect the interface to evolve.

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

## Run the measured structure-free route

Input files use the normal Boltz YAML format and must request the affinity
property. For example:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: YOUR_PROTEIN_SEQUENCE
  - ligand:
      id: B
      smiles: YOUR_LIGAND_SMILES
properties:
  - affinity:
      binder: B
```

Run one file or a directory of inputs:

```bash
boltz-stfree predict inputs/ \
  --out_dir results/ \
  --affinity_structure_free \
  --affinity_structure_free_fast \
  --affinity_screening_minimal_output \
  --affinity_distogram_mode zero_distogram_features \
  --affinity_recycling_steps 0 \
  --sampling_steps_affinity 1 \
  --diffusion_samples_affinity 1 \
  --num_workers 2
```

CUDA kernels are enabled by default. The paper's complete 49,980-compound ALR
screen used this route with batch size 1, the MSA enabled, ETKDG ligand
reference coordinates retained, and a fixed experimental seed. The affinity
sampling counts in the example are inert because structure sampling is skipped;
they make the recorded run configuration unambiguous.

Minimal output writes one score JSON per record plus shared run metadata. It
suppresses structure files, full distogram artifacts, and Lightning logs.

## Receptor crop requirements

The affinity crop can contain at most 256 tokens, including at most 128 ligand
tokens and at most 200 receptor tokens. If the complete receptor fits, the
default sequence-prefix crop can retain it. For longer receptors, supply
validated receptor-token indices and a provenance label:

```bash
boltz-stfree predict inputs/ \
  --out_dir results/ \
  --affinity_structure_free \
  --affinity_structure_free_fast \
  --affinity_screening_minimal_output \
  --affinity_distogram_mode zero_distogram_features \
  --affinity_recycling_steps 0 \
  --affinity_crop_indices_dir crops/ \
  --affinity_structure_free_crop_label known-pocket-v1
```

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

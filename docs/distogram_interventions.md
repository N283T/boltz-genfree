# Affinity distogram interventions

This repository evaluates controlled changes at the narrow
`AffinityModule` distogram boundary. The trunk, selected pose, affinity weights,
and affinity head are shared across all requested modes in one forward pass.

These options are research code for reproducing manuscript ablations. They are
retained and tested, but are not the supported screening interface documented
in the top-level README. Their CLI and artifact formats may evolve.

## Modes

- `control`: the unmodified coordinate-derived distogram. It is always run and
  written first, even when the CLI only requests intervention modes.
- `collapsed_coordinates`: ligand representative coordinates are set to the
  origin before distances are computed. Protein coordinates are unchanged.
- `constant_middle`: only symmetric protein-ligand pairs are replaced by bin 32
  of 64. Protein-protein and ligand-ligand bins remain coordinate-derived.
- `constant_far`: only symmetric protein-ligand pairs are replaced by the final
  bin (63 of 64).
- `zero_conditioning`: real bins are retained in the audit artifact, but the
  complete `PairwiseConditioning` contribution is omitted.

These are interventions, not claims that any mode approximates a learned
pose-free optimum. Any retained affinity or screening signal is a result to
measure, including negative and distribution-shift results.

## Running a comparison

Repeat `--affinity_distogram_mode` for every intervention to evaluate. Control
does not need to be listed explicitly:

```bash
boltz predict input.yaml \
  --write_embeddings \
  --affinity_distogram_mode collapsed_coordinates \
  --affinity_distogram_mode constant_middle \
  --affinity_distogram_mode constant_far \
  --affinity_distogram_mode zero_conditioning
```

The original `affinity_<id>.json` remains the control-compatible output. Each
mode also writes:

```text
affinity_<id>__distogram_<mode>.json
affinity_<id>__distogram_<mode>.npz
```

The JSON records scalar predictions, both ensemble members, checkpoint paths
and hashes, seed, recycle and sampling settings, the intervention mask count,
and a bin histogram. The NPZ stores the exact effective bins and intervention
mask. Large predictions and embeddings should remain outside the repository.

## Resuming from embeddings

`--write_embeddings` stores final trunk `s/z` arrays as float32. A later run can
start after the recycle count represented by that cache:

```bash
boltz predict input.yaml \
  --resume_embeddings_dir path/to/predictions \
  --resume_recycling_steps 3 \
  --recycling_steps 5
```

Both `predictions/<id>/embeddings_<id>.npz` and a flat directory of embedding
files are accepted. Shapes and token counts are validated. Affinity cropping
uses saved token indices to select the corresponding rows and pair matrix.

The saved recycle count is the CLI `--recycling_steps` value used to create the
cache. Resuming from `N` starts at recycle loop index `N + 1`. The requested
total must be at least `N`; affinity prediction currently has a fixed total of
5. Resume avoids recomputing completed trunk recycles, but it does not promise
bitwise-identical downstream diffusion samples: skipping trunk work changes the
execution path, and GPU sampling may consume randomness differently.

## Research option groups

### Distogram construction and random controls

- `--affinity_distogram_mode` selects one or more interventions and may be
  repeated. `control` is the default for coordinate-derived runs.
- `--affinity_distogram_seed` records and controls random distograms.
- `--affinity_random_distance_mean` and
  `--affinity_random_distance_std` parameterize `normal_random`; the standard
  deviation must be positive.
- `--affinity_random_pair_scope` chooses all token pairs or only
  protein-ligand pairs. Structure-free random distograms require the `all`
  scope because no coordinate-derived pairs are available.

Structure-free execution rejects coordinate-dependent modes. Its allowed modes
are `uniform_random`, `normal_random`, `zero_distogram_features`,
`uniform_mean_embedding`, and `zero_conditioning`.

### Exact affinity-head replay

`--affinity_head_only` replays only the affinity head from an exact captured
context and skips structure generation and affinity-trunk recycling. It is
Boltz-2-only and requires `--affinity_context_dir` pointing to an existing
directory of `affinity_head_context_<id>.npz` files. A context directory is not
accepted during an ordinary structure-free run unless head-only replay is also
selected.

### Prior-free sequence windows and explicit crops

`--affinity_sequence_window_start` selects the zero-based start of a
deterministic 200-protein-token structure-free window. A nonzero start requires
`--affinity_structure_free` and cannot be combined with an explicit crop.

`--affinity_crop_indices_dir` instead loads explicit `affinity_crop_<id>.npz`
token indices. In a structure-free run it requires
`--affinity_structure_free_crop_label`, a filename-safe provenance label. The
label is invalid without the external crop directory.

### Prepared geometry without structure generation

`--affinity_skip_structure` skips structure prediction but, unlike the fully
structure-free route, requires previously prepared `pre_affinity` artifacts.
`--affinity_prepared_geometry_label` records their provenance when supplied.
It is invalid with `--affinity_structure_free`, which does not load
coordinates.

### Embedding resume

`--resume_embeddings_dir` resumes Boltz-2 trunk state from an existing
embedding directory and requires `--resume_recycling_steps`. Supplying the
recycle count without an embedding directory is invalid. Outside head-only
replay, the saved recycle count cannot exceed the requested structure recycle
count; for affinity execution it also cannot exceed
`--affinity_recycling_steps`.

### Artifact provenance

`--affinity_output_tag` adds a filename-safe identifier to keep repeated
intervention outputs distinct. It and `--affinity_structure_free_crop_label`
must start with an alphanumeric character and may otherwise contain letters,
digits, `.`, `_`, or `-`. The crop label describes token selection; it does not
change the supplied crop.

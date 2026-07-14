# Affinity distogram interventions

This private research branch evaluates controlled changes at the narrow
`AffinityModule` distogram boundary. The trunk, selected pose, affinity weights,
and affinity head are shared across all requested modes in one forward pass.

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

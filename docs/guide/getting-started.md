# Getting started

Install, write a config, and rank a ligand library.

## Install

Python 3.10 or newer. The package is not on PyPI; install it from a tagged release.

```bash
git clone https://github.com/N283T/boltz-genfree.git
cd boltz-genfree
uv sync --frozen --extra cuda      # --extra cuda12 for CUDA 12 PyTorch builds
source .venv/bin/activate
```

The Boltz-2 affinity checkpoint is downloaded on first use. It is not redistributed with the package.

## Write a config

A screen config names one protein target and one ligand source.

```yaml
version: 1

target:
  sequence: YOUR_PROTEIN_SEQUENCE
  msa: auto                  # or a path to an A3M file, or empty

ligands:
  csv: ./library.csv
  id_column: id
  smiles_column: smiles

output_dir: ./screen-output
```

`library.csv` holds one ligand per row:

```text
id,smiles
glycine,NCC(=O)O
phenol,Oc1ccccc1
```

A complete example is in [examples/screen.yaml](../../examples/screen.yaml).

## Run a screen

```bash
boltz-genfree screen screen.yaml --prepare-only   # validate the inputs; no model is loaded
boltz-genfree screen screen.yaml                  # score every ligand
```

Scores are written to `output_dir/scores.csv`, one row per ligand. Rank the library by
`affinity_probability_binary` (higher is more likely a binder) or by
`affinity_pred_value` (Boltz-2's affinity value; lower predicts stronger binding). If the
run stops, run the same command again: it resumes where it stopped.

## Next steps

- [Configuration](configuration.md): every config key, receptor cropping, size limits, outputs and resume.
- [Browser tool](browser-tool.md): score a few ligands in the browser, with no install.
- [How it works](https://n283t.github.io/boltz-genfree/how-it-works.html): what GenFree replaces in Boltz-2, and why the ranking holds.

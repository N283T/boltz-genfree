# Boltz-GenFree

Structure-generation-free Boltz-2 affinity screening.

Boltz-GenFree (GenFree) runs the released Boltz-2 affinity trunk and heads directly
from processed receptor and ligand inputs. It runs no structure generation for any candidate and
writes compact scores for large, resumable virtual screens. It is intended as a cheap
first stage: rank the whole library with GenFree, then spend the full Boltz-2 pipeline
only on the top of the ranking.

![Boltz-GenFree overview](docs/images/genfree_overview.png)

## Performance

![Simulated two-stage screen: shortlist-share sweep](docs/images/performance.png)

## Install

Python 3.10 or newer. The package is not on PyPI; install from a tagged release.

```bash
git clone https://github.com/N283T/boltz-genfree.git
cd boltz-genfree
uv sync --frozen --extra cuda      # --extra cuda12 for CUDA 12 PyTorch builds
source .venv/bin/activate
```

The Boltz-2 affinity checkpoint is downloaded on first use; the checkpoint itself is not
redistributed. The one exception is the browser tool, which loads a 35 MB float32 conversion of
the checkpoint's affinity heads (the checkpoint's own values) from the release
[`webgpu-weights-1`](https://github.com/N283T/boltz-genfree/releases/tag/webgpu-weights-1)
and reads the trunk weights from LocalFold's public bundle on Hugging Face.

## Usage

Write a config with one protein and one ligand source:

```yaml
version: 1

target:
  sequence: YOUR_PROTEIN_SEQUENCE
  msa: ./target.a3m          # or auto / empty
  crop:                      # optional when the complete receptor fits
    reference: ./reference-complex.cif
    reference_protein: A
    reference_ligand: L

ligands:
  csv: ./library.csv
  id_column: id
  smiles_column: smiles

output_dir: ./screen-output
```

```bash
boltz-genfree screen screen.yaml --prepare-only   # dry run: validate the inputs
boltz-genfree screen screen.yaml                  # run; scores are written to scores.csv
```

A re-run with the same inputs resumes where it stopped. Ligand sources, options, size
limits, outputs, the optional reference bootstrap and fine-tuned heads are described in
[docs/guide/configuration.md](docs/guide/configuration.md).

## Website

The project page, the Docs, the article and the browser tool are published at
[n283t.github.io/boltz-genfree](https://n283t.github.io/boltz-genfree/). The site is assembled from `docs/site/` and
the Markdown in this repository by `docs/site/assemble.sh` (preview: `docs/site/assemble.sh && python3 -m http.server -d _site`).

## WebGPU (experimental)

`webgpu/` runs the full structure-free chain on WebGPU, in Node and in a browser:
featurisation, the Boltz-2 trunk (embedder, MSA module, pairformer) and the affinity module,
all implemented in this repository, from a protein sequence (+ optional A3M) and a ligand
SMILES (the [LocalFold](https://github.com/sokrypton/localfold) WebGPU port was used as a
reference only). With exact PyTorch features it agrees with GenFree to |value diff| <= 3e-6; with its
own featuriser to within GenFree's seed-to-seed spread. `node src/cli.js screen screen.yaml` reads the same config as `boltz-genfree screen` and writes `scores.csv`. See [docs/webgpu.md](docs/webgpu.md)
for setup, tolerances, timings and limits.

## License

[MIT License](LICENSE). Derived from [Boltz](https://github.com/jwohlwend/boltz) and
[boltz-community](https://github.com/Novel-Therapeutics/boltz-community); history is
preserved.

## Citation

Preprint: T. Nagae and K. Tomii, Boltz-GenFree: structure-generation-free ranking for
first-stage affinity screening, *ChemRxiv* (2026),
[10.26434/chemrxiv.15012606/v1](https://doi.org/10.26434/chemrxiv.15012606/v1).

Release v0.1.0 is archived on Zenodo:
[10.5281/zenodo.24400635](https://doi.org/10.5281/zenodo.24400635)
(all versions: [10.5281/zenodo.24400634](https://doi.org/10.5281/zenodo.24400634)).
The curated evidence and analysis code for the associated manuscript are archived at
[10.5281/zenodo.23949585](https://doi.org/10.5281/zenodo.23949585). See
[CITATION.cff](CITATION.cff) for citation metadata.

## Development

```bash
uv sync --frozen --extra test --extra lint
pytest -q
```

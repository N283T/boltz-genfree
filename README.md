# Boltz-GenFree

Structure-generation-free Boltz-2 affinity screening.

Boltz-GenFree (GenFree) runs the released Boltz-2 affinity trunk and heads directly
from processed receptor and ligand inputs. It builds no structure for any candidate and
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

The Boltz-2 affinity checkpoint is downloaded on first use; weights are not
redistributed.

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
limits, the optional reference bootstrap and fine-tuned heads are described in
[docs/usage.md](docs/usage.md) and [docs/finetuning.md](docs/finetuning.md).

## License

[MIT License](LICENSE). Derived from [Boltz](https://github.com/jwohlwend/boltz) and
[boltz-community](https://github.com/Novel-Therapeutics/boltz-community); history is
preserved. See [CITATION.cff](CITATION.cff) to cite.

## Development

```bash
uv sync --frozen --extra test --extra lint
pytest -q
```

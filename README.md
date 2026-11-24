# Boltz-GenFree

Rank a ligand library with the released Boltz-2 affinity checkpoint, without structure
generation for any candidate.

**Website: [n283t.github.io/boltz-genfree](https://n283t.github.io/boltz-genfree/)** ·
[Docs](https://n283t.github.io/boltz-genfree/docs/) ·
[How it works](https://n283t.github.io/boltz-genfree/how-it-works.html) ·
[Results](https://n283t.github.io/boltz-genfree/results.html) ·
[Try it in the browser](https://n283t.github.io/boltz-genfree/webgpu/web/screen.html)

![Boltz-GenFree overview](docs/images/genfree_overview.png)

## Install

Python 3.10 or newer. The package is not on PyPI; install from a tagged release.

```bash
git clone https://github.com/N283T/boltz-genfree.git
cd boltz-genfree
uv sync --frozen --extra cuda      # --extra cuda12 for CUDA 12 PyTorch builds
source .venv/bin/activate
```

The Boltz-2 affinity checkpoint is downloaded on first use and is not redistributed. The
browser tool loads a converted copy of its affinity heads from the release
[`webgpu-weights-1`](https://github.com/N283T/boltz-genfree/releases/tag/webgpu-weights-1).

## Run

```bash
boltz-genfree screen screen.yaml
```

Scores are written to `scores.csv`. The config format is in
[Getting started](docs/guide/getting-started.md) and
[Configuration](docs/guide/configuration.md); an example is in
[examples/screen.yaml](examples/screen.yaml).

## Citation

T. Nagae and K. Tomii, Boltz-GenFree: structure-generation-free ranking for first-stage
affinity screening, *ChemRxiv* (2026),
[10.26434/chemrxiv.15012606/v1](https://doi.org/10.26434/chemrxiv.15012606/v1).
Software: [10.5281/zenodo.24400635](https://doi.org/10.5281/zenodo.24400635).
Data and analysis: [10.5281/zenodo.23949585](https://doi.org/10.5281/zenodo.23949585).
See [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE). Derived from [Boltz](https://github.com/jwohlwend/boltz) and
[boltz-community](https://github.com/Novel-Therapeutics/boltz-community); history is
preserved.

## Development

```bash
uv sync --frozen --extra test --extra lint
pytest -q
```

The site is built by `docs/site/assemble.sh`; the WebGPU chain is described in
[docs/webgpu.md](docs/webgpu.md).

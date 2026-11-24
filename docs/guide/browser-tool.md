# Browser tool

Score a few ligands in the browser, with no install.

## In the browser

Open the [WebGPU tool](../../webgpu/web/screen.html) in a browser with WebGPU enabled.

1. Paste the receptor sequence. An A3M file is optional; without one the receptor is scored as a single sequence.
2. Add ligands: one SMILES per line, `id,SMILES` lines, or a CSV/TSV file with a `smiles` column.
3. Set recycling, seed and the ligand conformer mode if needed.
4. Press **Run**. Results appear per ligand, sorted by P(binder); **Download scores.csv** saves them with the same columns as `boltz-genfree screen`, plus `error`.

A ligand that fails, for example one over the size limit, is listed with its error and the
run continues.

The first run downloads the trunk weights (about 380 MB, from LocalFold's public bundle on
Hugging Face) and keeps them in the browser cache. Later visits read them from disk.
Each ligand then takes about 10 s for a 117-residue receptor on an Apple M4.

### Receptor crop

The page checks whether the receptor and every ligand fit the
[size limits](configuration.md#receptor-crop-and-size-limits). When the receptor is too
long, load a reference complex (PDB or mmCIF): the page picks the matching protein chain
and the reference ligand, and keeps the residues nearest to it, as `budget: library_fixed`
does. The chosen residues can be edited by hand.

## From the command line

The same WebGPU chain runs in Node. The CLI reads the same `screen.yaml` and writes the same `scores.csv`.

```bash
cd webgpu && npm install
node src/cli.js screen ../examples/screen.yaml --prepare-only   # validate the inputs
node src/cli.js screen ../examples/screen.yaml                  # score every ligand
```

It needs the weight packs converted from the Boltz-2 checkpoint; see
[docs/webgpu.md](../webgpu.md#setup).

## Differences from `boltz-genfree screen`

Values agree with the Python screen to within its own seed-to-seed spread, not exactly.
Use the Python screen for large libraries and for results you report.

Not supported in the browser tool or the Node CLI:

- `msa: auto` (no MSA server; the receptor is scored without an MSA unless an A3M is given)
- SDF ligand files
- `affinity_adapter` and `affinity_checkpoint`
- `target.crop` in the Node CLI (the browser page has its own crop)
- resume in the browser page

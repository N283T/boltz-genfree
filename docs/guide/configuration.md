# Configuration

Every key of the screen config, and what a screen writes.

## Top level

Paths in the config are relative to the config file.

| Key | Default | Meaning |
|---|---|---|
| `version` | `1` | Config version. Only `1` is supported. |
| `target` | required | The protein target. |
| `ligands` | required | One ligand source. |
| `output_dir` | `<config name>_screen` | Where scores and run state are written. |
| `chunk_size` | `1000` | Ligands per prepared chunk. |
| `options` | `{}` | Runtime options. |

## Target

| Key | Default | Meaning |
|---|---|---|
| `sequence` | required | Protein sequence (one chain). Whitespace is ignored. |
| `msa` | `auto` | `auto` (query the MSA server once per target), a path to an A3M file, or `empty` (single sequence). |
| `id` | `A` | Chain ID of the target. `B` is reserved for the ligand. |
| `crop` | none | Crop the receptor around a reference ligand. See [Receptor crop and size limits](#receptor-crop-and-size-limits). |

## Ligands

Give exactly one source.

| Source | Keys | Notes |
|---|---|---|
| CSV / TSV | `csv`, `id_column`, `smiles_column`, `delimiter` | A `.tsv` file is read tab-separated unless `delimiter` is set. |
| SDF | `sdf`, `id_property` | Coordinates in the file are not used. |
| SMILES | `smiles` | An inline list. |
| CCD | `ccd` | An inline list of component codes. |

## Options

| Key | Default | Meaning |
|---|---|---|
| `seed` | none | Fixed seed. A ligand gets the same score whatever the chunking, workers or resumption. |
| `accelerator` | `gpu` | `gpu`, `cpu` or `mps`. |
| `devices` | `1` | Number of GPUs. Must be `1` on `cpu` and `mps`. |
| `num_workers` | `2` | Data-loading workers. |
| `affinity_batch_size` | `1` | Ligands per forward pass. |
| `affinity_recycling_steps` | `0` | Trunk recycling steps, `0`–`5`. |
| `ligand_reference_conformer` | `skip` | `skip`, `etkdg` or `rdkit_2d`. |
| `affinity_conditioner_mode` | `original` | `original` or `folded`. |
| `allow_extended_full_input` | `false` | Allow an uncropped receptor over 200 tokens. Not with `target.crop`. |
| `affinity_checkpoint` | released | Path to another full Boltz-2 affinity checkpoint. |
| `affinity_adapter` | none | Directory of fine-tuned heads. See [Fine-tuned heads](#fine-tuned-heads). |
| `cache` | `~/.boltz` | Where the checkpoint and CCD data are kept (or `$BOLTZ_CACHE`). |
| `override` | `false` | Recompute earlier outputs instead of resuming. |
| `no_kernels` | `false` | Disable the fused GPU kernels. |
| `msa_server_url` | ColabFold | MSA server for `msa: auto`. Default `https://api.colabfold.com`. |
| `msa_pairing_strategy` | `greedy` | `greedy` or `complete`. |

Advanced options, rarely needed: `preprocessing_threads`, `max_msa_seqs` (8192),
`subsample_msa` (`true`), `num_subsampled_msa` (1024), `flash_attn`,
`affinity_mw_correction`, `etkdg_seed` (42), and the MSA server credentials
`msa_server_username`, `msa_server_password`, `api_key_header`, `api_key_value`.
An unknown key is an error.

## Receptor crop and size limits

Without `target.crop`, every ligand must fit these limits:

| Limit | Tokens |
|---|---|
| Receptor | 200 |
| Ligand | 128 |
| Receptor + ligand | 256 |

A ligand over a limit stops the screen with its name. Nothing is cropped or skipped
silently. For a longer receptor, either crop it or set
`options.allow_extended_full_input: true` (the ligand limit still applies; it cannot be combined with a crop).

To crop, give a reference complex (PDB or mmCIF) with a bound ligand. Receptor residues
are ranked by distance to that ligand and kept in contiguous blocks until the limits are
reached.

```yaml
target:
  sequence: YOUR_PROTEIN_SEQUENCE
  crop:
    reference: ./reference-complex.cif
    reference_protein: A        # optional: chain of the receptor in the reference
    reference_ligand: L         # required: chain, or {chain, residue, ccd}
    budget: per_ligand          # or library_fixed
```

| Key | Default | Meaning |
|---|---|---|
| `reference` | required | PDB or mmCIF complex. |
| `reference_protein` | detected | Receptor chain in the reference. By default the chain matching `target.sequence`. |
| `reference_ligand` | required | Ligand in the reference: a chain ID, or `{chain, residue, ccd}`. |
| `budget` | `per_ligand` | `per_ligand` crops for each ligand; `library_fixed` uses one crop for the whole library. |
| `library_ligand_tokens` | largest ligand | With `library_fixed`: ligand tokens to reserve, `1`–`128`, at least the largest ligand. |

The crop used for each ligand is recorded in the manifest and in `scores.csv`.

A reference crop needs a reference complex once per target but keeps the inputs small;
the full receptor needs no preparation but costs more as the receptor grows. GenFree's two
routes differ in this choice. See
[What GenFree does](https://n283t.github.io/boltz-genfree/how-it-works.html#design) and
[What the complete receptor costs](https://n283t.github.io/boltz-genfree/results.html#receptor-cost).

### No reference complex?

`bootstrap-reference` predicts one from the target sequence and one anchor ligand. This
runs structure generation once, so it is a separate command and `screen` never starts it.

```yaml
version: 1
target:
  sequence: YOUR_PROTEIN_SEQUENCE
  msa: auto
anchor:
  smiles: ANCHOR_LIGAND_SMILES
output_dir: ./reference
```

```bash
boltz-genfree bootstrap-reference bootstrap.yaml --prepare-only
boltz-genfree bootstrap-reference bootstrap.yaml
```

Use `output_dir/reference.pdb` as `target.crop.reference`, and report it apart from the structure-generation-free screen.

## Outputs and resume

| File in `output_dir` | Contents |
|---|---|
| `scores.csv` | One row per ligand. |
| `screen_summary.json` | Expected and completed counts, and any missing ligands. |
| `screen_status.json` | Progress, updated after every batch. Safe to read while the screen runs. |
| `runs/`, `.boltz-stfree/` | Prepared inputs, per-ligand scores and run state. |

`scores.csv` columns:

| Column | Meaning |
|---|---|
| `record_id`, `ligand_id` | Internal and input IDs. |
| `ligand_kind`, `ligand_value` | `smiles` or `ccd`, and the value. |
| `ligand_tokens`, `receptor_tokens` | Token counts used. |
| `chunk` | Chunk the ligand was prepared in. |
| `crop_budget` | `per_ligand`, `library_fixed` or `complete_input` (no crop). |
| `affinity_pred_value` | Boltz-2's affinity value; lower predicts stronger binding. |
| `affinity_probability_binary` | Boltz-2's binder probability. |

Run the same command again to resume. Finished ligands are not scored again. A changed
target, ligand source, MSA or preprocessing limit is rejected rather than mixed into the
earlier results; set `options.override: true` to recompute.

## Fine-tuned heads

The default screen uses the released weights and needs no labels. Affinity heads
fine-tuned elsewhere can be imported as an adapter and used in a screen.

```bash
boltz-genfree adapter import-heads --checkpoint final_heads.pt \
  --metadata metadata.json --output adapter_dir
boltz-genfree adapter inspect adapter_dir
```

```yaml
options:
  affinity_adapter: ./adapter_dir
```

The adapter is checked against the base checkpoint, the target and the feature settings
before scoring, and a changed adapter rejects resume. There is no training command.
The adapter format is described in [docs/finetuning.md](../finetuning.md).

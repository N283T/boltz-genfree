# Combined route integration smoke — 2026-09-07

## Scope and result

Added opt-in extended complete-input screening, exposed the existing RDKit 2D
reference-coordinate path, and integrated the previously validated folded
zero-distogram conditioner. Defaults remain unchanged. No narrow-width variant.

WSL RTX 5080: both combined routes completed for two existing library records,
with original and folded conditioner modes (eight total predictions). All six
outputs per record are finite and original/folded agree at atol=rtol=1e-5.
This is an integration/equivalence check, not a screening accuracy or speed
benchmark, and not an all-target integration claim.

| Route | Receptor | MSA | ref_pos | batch | max absolute difference across six outputs |
|---|---|---|---|---|---|
| minimal | all 580 input residues | query-only, MSA module retained | zero | 1 | 2.8312206268310547e-7 |
| balanced | fixed 159 residues | existing MSA, subsample 1024 | RDKit 2D | 2 | 4.172325134277344e-7 |

Both routes use FadD28 records `000001_2277079` and `000002_6520605` from the
existing canonical library, affinity recycling 0, seed 20260718, kernels ON,
MW correction OFF, and no structure prediction. The balanced route reuses the
existing ATP-bootstrap reference solely for this smoke; it does not fix the
seed-selection policy for the future full benchmark. Library ligand budget 96.

Checkpoint: `boltz2_aff.ckpt`, SHA256
`dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e`.
Runtime code SHA256 recorded for these predictions:
`0fd7cb2c79cf8afef1dafa8c4780dee703b6a7d069584269308aa6ac454845c6`.

## Implementation details

- `options.allow_extended_full_input: true` lifts receptor/combined-token and
  atom selection limits for complete input only. The 128-token ligand limit is
  retained. Crop overrides cannot be combined with it; every token must survive.
- `options.ligand_reference_conformer: rdkit_2d` uses the existing conformer path
  while parsing still bypasses 3D conformer generation.
- `options.affinity_conditioner_mode: folded` preserves full-width normalization
  statistics and all affine/projection contributions for `[pair, zero]`. It skips
  the zero-distogram embedding lookup and concatenation, preserving state-dict
  keys and checkpoint weights. Default `original` remains available for parity.
- Method options are written to plans/provenance and protect resumed outputs.

## Artifacts and reproduction

WSL repository: `/home/nagaet/ghq/github.com/N283T/boltz-stfree`.
Mac repository: `/Users/nagaet/ghq/github.com/N283T/boltz-stfree`.
Relative paths in both:

- `experiments/boltzina-multitarget/local/combined-route-smoke-20260907/`:
  four YAML configs/logs, `execution.json`, and `validation.json`.
- `experiments/boltzina-multitarget/outputs/combined-route-smoke-20260907-{minimal,balanced}-{original,folded}/`:
  score CSVs, plans/status, eight raw score JSONs and shared run metadata.

The Mac copy contains compact evidence, not processed caches. Original YAMLs
retain WSL paths. Existence check: read `validation.json` (`passed: true`,
24 score comparisons) and the four `scores.csv` files (two rows each).
The run command was `.venv/bin/boltz-stfree screen <route-mode.yaml>` for each
of the four configs. These outputs already exist: do not repeat the same
condition without a documented validation/recovery reason.

No embeddings were requested or produced. No dedicated speed benchmark or
ABCI job was launched. Full-target configs, one-shot seed selection and ABCI
batch sizing remain outside this completed smoke check.

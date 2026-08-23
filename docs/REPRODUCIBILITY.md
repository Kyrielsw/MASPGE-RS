# Reproducibility and evidence boundary

## Frozen protocol

- Seeds: 42, 43, 44, 45, and 46.
- Metrics: standardized MSE and MAE, reported as population mean ± standard
  deviation across seeds.
- Selection: lowest validation MSE.
- Data fitting: normalization statistics use training data only.
- Holdout: evaluation scripts load frozen checkpoints and contain no optimizer.
- Restarting: launchers skip completed seed/method pairs.

## Dataset status

| Dataset | Units | Active roles | Holdout interpretation |
|---|---:|---|---|
| SMARTEOLE | 7 turbines | A1–A3 | report-only; observed during development |
| SDWPF | 134 turbines | A1–A3 | frozen cross-dataset report; an old router had observed holdout |
| HAI 23.05 | thermal + hydro | A1–A4 | validation-frozen one-shot evaluation |

The repository therefore does not describe every reported holdout as strictly
unseen. No model is tuned from the frozen test results included in the paper
result files.

## Result provenance

`paper_results/main_results.csv` contains the three-dataset, eight-method main
table. `paper_results/multihorizon_results.csv` contains 5-, 15-, and 30-step
results for SOFTS, adapted MAFS, and MASPGE-RS on SMARTEOLE and SDWPF. Values
are copied from archived per-seed JSON summaries and are not recomputed by the
README.

Five seeds alone do not justify an automatic claim of statistical
significance. Paired window-level confidence intervals and hypothesis tests
should be generated from saved predictions before such a claim is made.

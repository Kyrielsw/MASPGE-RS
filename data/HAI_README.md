# HAI 23.05 normal-operation data

The HAI dataset is distributed by the official `icsdataset/hai` repository
under CC BY-SA 4.0.  MASPGE-RS uses only the four normal-operation files from
HAI 23.05.  Attack files, attack labels and HAIEnd controller-internal points
are outside the forecasting contract.

Official source:

- <https://github.com/icsdataset/hai/tree/master/hai-23.05>
- <https://github.com/icsdataset/hai/blob/master/hai_dataset_technical_details.pdf>

The raw files are intentionally excluded from Git and are expected under:

```text
data/raw/hai_23_05/
  hai-train1.csv
  hai-train2.csv
  hai-train3.csv
  hai-train4.csv
```

On the workstation, download with resumable HTTP/1.1 transfers and run the
schema/distribution audit:

```bash
cd /path/to/MASPGE-RS
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
bash scripts/download_hai_23_05.sh 2>&1 | tee logs/hai_23_05_download_audit.log
```

If the four files already exist, run only the audit:

```bash
python -u scripts/audit_hai_23_05.py \
  --raw-dir data/raw/hai_23_05 \
  --output data/hai_23_05_admission_v1.audit.json \
  2>&1 | tee logs/hai_23_05_admission.log
```

The audit performs no model training and no holdout model evaluation.  It
checks the actual schema, file hashes, one-second time continuity, missingness,
target variability and whether enough A4 cooling/water-treatment tags vary.
Only after `admitted_for_contract_freeze` is true may resampling and exact role
slots be frozen in a processed dataset.

After admission passes, create and verify the frozen two-unit Role-State
contract:

```bash
python -u scripts/prepare_hai_23_05.py \
  2>&1 | tee logs/hai_23_05_prepare.log

python scripts/verify_hai_role_state_setup.py
sha256sum data/hai_23_05_role_state_v1.npz
```

The frozen contract aggregates each contiguous ten-second block, uses 60
history steps to predict the mean of the next 15 steps, and therefore maps ten
minutes of history to a 2.5-minute forecast.  It contains two forecasting
units (`thermal` and `hydro`) with the same four-role, six-slot interface.  A4
is active for both units.  Missing slots are standardized zeros and are
recorded by the per-unit feature-slot mask.

Provisional file-level split, fixed before any model result is observed:

- train: `hai-train1.csv` and `hai-train2.csv`;
- validation: `hai-train3.csv`;
- holdout: `hai-train4.csv`.

No window may cross a source-file boundary.

## Validation-only MASPGE-RS experiment

Run the single-seed pipeline smoke test first.  This trains one epoch for each
of MAFS pretraining, adjacency finetuning and the frozen-base Role-State stage;
it never constructs the holdout dataset.

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_hai_role_state_formal.py \
  --seed 42 --device cuda --smoke \
  2>&1 | tee logs/hai_role_state_smoke_seed42.log
```

Only after the smoke result is structurally valid should the five-seed
validation-only launcher be used:

```bash
bash scripts/run_hai_role_state_validation_2gpu.sh
```

The launcher cannot read `holdout_end_indices`.  A separate frozen evaluator
will be added only if the preregistered five-seed validation rule accepts the
method.

After the five-seed validation summary records
`accepted_for_strict_frozen_holdout=true`, run the one-shot evaluator:

```bash
bash scripts/run_hai_role_state_holdout_2gpu.sh
```

This evaluator contains no training path.  It reports joint standardized
metrics and separate thermal/hydro physical metrics from the already frozen
checkpoints.  No post-holdout tuning is permitted.

## External paper baselines

The HAI main table uses Persistence, DLinear, FourierGNN, iTransformer, SOFTS,
MAFS-adapted Base and MASPGE-RS.  The four trainable external baselines use
power history only and reuse their frozen SMARTEOLE/SDWPF configurations.

Run the structural smoke test before the full five-seed validation:

```bash
bash scripts/run_hai_external_baseline_smoke_2gpu.sh
```

After auditing the smoke logs, run validation only:

```bash
bash scripts/run_hai_external_baseline_validation_2gpu.sh
```

External holdout evaluation remains a separate step and must load the
validation-selected checkpoints without training.

After all 20 trainable validation jobs and the five Persistence records are
complete, create the final seven-method holdout table with:

```bash
bash scripts/run_hai_external_baseline_holdout_2gpu.sh
```

The script only evaluates frozen checkpoints and merges them with the existing
MAFS Base/MASPGE-RS strict holdout records.

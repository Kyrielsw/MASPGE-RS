# MASPGE-RS

**Unit-wise Multivariate State-Augmented Collaborative Forecasting for
Heterogeneous Energy Systems**

MASPGE-RS is a PyTorch implementation for collaborative forecasting across
multiple physical units. This repository provides data preparation tools,
training and evaluation entry points, baseline implementations, and
reproducible launchers for SMARTEOLE, SDWPF, and HAI 23.05.

## Repository layout

```text
configs/       Dataset and experiment configurations
data/          Data contracts and dataset-specific instructions
docs/          Reproducibility notes
scripts/       Preprocessing, training, evaluation, and aggregation commands
src/maspge/    Models, data loaders, metrics, and baseline implementations
tests/         Model and protocol tests
```

Raw datasets, processed arrays, checkpoints, logs, and generated outputs are
not tracked by Git.

## Requirements

- Linux is recommended for the provided shell launchers.
- Python 3.10 or 3.11.
- PyTorch compatible with the local CPU or CUDA installation.
- Two CUDA GPUs for the parallel launchers. Individual jobs can run on one GPU
  or CPU.

## Installation

```bash
git clone https://github.com/Kyrielsw/MASPGE-RS.git
cd MASPGE-RS

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export PYTHONPATH="$PWD/src"
python -m pytest -q
```

For CUDA systems, install the PyTorch build recommended by the
[official PyTorch selector](https://pytorch.org/get-started/locally/) before
installing the remaining requirements. Export `PYTHONPATH` in every new shell.

## Data preparation

Detailed source links, expected file names, hashes, and split contracts are
listed in [`data/README.md`](data/README.md).

### SMARTEOLE

Download and extract the official archive to the paths specified in
`configs/smarteole_role_contract_v1.json`, then run:

```bash
python scripts/prepare_smarteole.py
python scripts/verify_setup.py
```

The processed contract is written to:

```text
data/s1_complete_case_v0.1.npz
```

### SDWPF

Download the public 245-day training file and turbine-location table:

```bash
bash scripts/download_sdwpf_kdd.sh
```

Create the dense per-turbine state contract:

```bash
python -u scripts/prepare_sdwpf.py \
  --contract dense_role_v2 \
  --csv data/raw/sdwpf_kddcup/sdwpf_245days_v1.csv \
  --locations data/raw/sdwpf_kddcup/sdwpf_baidukddcup2022_turb_location.csv
```

See [`data/SDWPF_README.md`](data/SDWPF_README.md) for the complete contract.

### HAI 23.05

Download and audit the four normal-operation files:

```bash
mkdir -p logs

bash scripts/download_hai_23_05.sh \
  2>&1 | tee logs/hai_23_05_download_audit.log
```

Create and verify the processed thermal/hydro contract:

```bash
python -u scripts/prepare_hai_23_05.py \
  2>&1 | tee logs/hai_23_05_prepare.log

python scripts/verify_hai_role_state_setup.py
```

See [`data/HAI_README.md`](data/HAI_README.md) for the admission audit and raw
directory structure.

## Quick smoke tests

Run the automated tests before training:

```bash
export PYTHONPATH="$PWD/src"
python -m pytest -q
```

Check the MAFS training path on one GPU with a reduced workload:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_mafs_base.py \
  --dataset smarteole \
  --seed 42 \
  --device cuda \
  --smoke
```

Check the complete HAI training path:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_hai_role_state_formal.py \
  --seed 42 \
  --device cuda \
  --smoke \
  2>&1 | tee logs/hai_role_state_smoke_seed42.log
```

Use `--device cpu` for a CPU-only structural check. Smoke outputs are stored in
separate directories and are not used by full runs.

## Training

### Stage 1: base forecasting models

SMARTEOLE and SDWPF use validation-selected MAFS base checkpoints:

```bash
bash scripts/run_mafs_base_2gpu.sh
```

To run one dataset and seed on a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_mafs_base.py \
  --dataset smarteole \
  --seed 42 \
  --device cuda \
  --resume
```

### Stage 2: state-augmented models

After preparing the datasets and required base checkpoints, run the
dataset-specific validation launchers:

```bash
# SMARTEOLE
bash scripts/run_role_state_full_validation_2gpu.sh

# SDWPF
bash scripts/run_sdwpf_role_state_formal_2gpu.sh

# HAI 23.05
bash scripts/run_hai_role_state_validation_2gpu.sh
```

The launchers distribute five seeds between `CUDA_VISIBLE_DEVICES=0` and
`CUDA_VISIBLE_DEVICES=1`, stream progress to `logs/`, and skip completed jobs
when restarted.

For a long-running job, redirect the master output while retaining the
per-GPU logs created by the launcher:

```bash
nohup bash scripts/run_hai_role_state_validation_2gpu.sh \
  > logs/hai_role_state_validation_master.log 2>&1 &

echo $!
tail -f logs/hai_role_state_validation_master.log
```

## Evaluation

Training and evaluation are separate. Complete the validation run first, keep
its selected checkpoints unchanged, and then use the matching read-only
evaluation launcher:

```bash
# SMARTEOLE
bash scripts/run_role_state_holdout_2gpu.sh

# SDWPF
bash scripts/run_sdwpf_role_state_holdout_2gpu.sh

# HAI 23.05
bash scripts/run_hai_role_state_holdout_2gpu.sh
```

The evaluation scripts load frozen checkpoints and do not contain an optimizer
or training path.

## Optional workflows

The repository also includes runnable workflows for:

```text
scripts/run_role_state_ablation_v1_2gpu.sh           Structural ablations
scripts/run_multihorizon_validation_2gpu.sh          Multi-horizon validation
scripts/run_multihorizon_holdout_2gpu.sh             Frozen multi-horizon evaluation
scripts/run_hai_external_baseline_validation_2gpu.sh External baseline validation
scripts/run_hai_external_baseline_holdout_2gpu.sh    Frozen baseline evaluation
scripts/run_timemixer_validation_2gpu.sh              TimeMixer validation
scripts/run_timemixer_holdout_2gpu.sh                 Frozen TimeMixer evaluation
```

Check the corresponding configuration under `configs/` before launching an
optional workflow.

## Outputs

Generated artifacts follow the same directory convention across workflows:

```text
checkpoints/<experiment>/...   Validation-selected model checkpoints
results/<experiment>/...       Per-seed JSON files and aggregate summaries
logs/...                       Live progress and aggregation logs
```

Preserve the configuration, processed-data contract, and checkpoint directory
together when moving an experiment between machines.

## License and citation

MASPGE-RS is released under the [MIT License](LICENSE). Third-party notices are
listed in [`THIRD_PARTY.md`](THIRD_PARTY.md) and [`licenses/`](licenses/).
Dataset licenses are governed by their official sources. Citation metadata is
provided in [`CITATION.cff`](CITATION.cff).

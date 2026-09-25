# XAI4HEAT data

The raw XAI4HEAT district-heating SCADA data are not redistributed in this
repository. Download version 1 from the official Mendeley Data record:

<https://data.mendeley.com/datasets/2mwc6x6kwb/1>

Save the downloaded archive as:

```text
data/raw/xai4heat_v1/xai4heat_scada_dataset_2024_v1.zip
```

The expected source-archive SHA-256 is recorded in
`configs/xai4heat_frozen_v1.json`. Build the processed contract with:

```bash
python -u scripts/prepare_xai4heat.py
python scripts/verify_xai4heat_setup.py
```

The preparation script aligns the five substations by timestamp, preserves
heating-season boundaries, fits normalization statistics on the training
season only, and writes `data/xai4heat_scada_2024_generic_v1.npz`. The raw and
processed data remain ignored by Git and are governed by the dataset's own
license and terms.

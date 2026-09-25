#!/usr/bin/env python3
"""Fast contract checks before XAI4HEAT GPU work."""

from pathlib import Path
import json
import numpy as np

from maspge.data import load_frozen_xai4heat


ROOT = Path(__file__).resolve().parent.parent
config = json.loads((ROOT / "configs/xai4heat_frozen_v1.json").read_text(encoding="utf-8"))
arrays = load_frozen_xai4heat(
    ROOT / config["dataset"]["path"], expected_sha256=config["dataset"]["sha256"]
)
audit = json.loads(
    (ROOT / "data/xai4heat_scada_2024_generic_v1.audit.json").read_text(encoding="utf-8")
)
assert audit["source_zip_sha256"] == config["dataset"]["source_zip_sha256"]
assert audit["output_sha256"] == str(arrays["sha256"].item())
assert np.isfinite(arrays["x_role"]).all()
assert np.isfinite(arrays["power"]).all()
assert arrays["train_end_indices"].max() < arrays["validation_end_indices"].min()
assert arrays["validation_end_indices"].max() < arrays["holdout_end_indices"].min()
for name in ("train", "validation", "holdout"):
    endpoints = arrays[f"{name}_end_indices"]
    for end in (int(endpoints[0]), int(endpoints[-1])):
        start = end - config["dataset"]["history_steps"] + 1
        stop = end + config["dataset"]["horizon_steps"]
        assert len(set(arrays["season_ids"][start : stop + 1].tolist())) == 1
        assert bool(arrays["power_valid"][start : stop + 1].all())
print(f"DATASET_OK sha256={arrays['sha256'].item()}")
print(f"SOURCE_ZIP_OK sha256={audit['source_zip_sha256']}")
print(f"x_role={arrays['x_role'].shape} power={arrays['power'].shape} units={arrays['unit_names'].tolist()}")
print("windows " + " ".join(f"{name}={len(arrays[f'{name}_end_indices'])}" for name in ("train", "validation", "holdout")))
print("XAI4HEAT_SETUP_OK")

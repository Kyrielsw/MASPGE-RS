from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_formal_configs_use_the_same_five_seeds() -> None:
    names = (
        "smarteole_role_state_full_validation_v1.json",
        "sdwpf_role_state_formal_v1.json",
        "hai_role_state_formal_v1.json",
    )
    for name in names:
        config = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
        assert config["seeds"] == [42, 43, 44, 45, 46]


def test_hai_training_script_does_not_construct_holdout() -> None:
    source = (ROOT / "scripts/run_hai_role_state_formal.py").read_text(
        encoding="utf-8"
    )
    assert "holdout_end_indices" not in source


def test_holdout_evaluators_have_no_training_path() -> None:
    names = (
        "evaluate_role_state_holdout.py",
        "evaluate_sdwpf_role_state_holdout.py",
        "evaluate_hai_role_state_holdout.py",
    )
    for name in names:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "train_stage" not in source
        assert "optimizer" not in source.lower()


def test_paper_table_contains_all_eight_methods() -> None:
    with (ROOT / "paper_results/main_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert rows[-1]["method"] == "maspge_rs"
    for dataset in ("smarteole", "sdwpf", "hai"):
        proposed = float(rows[-1][f"{dataset}_mse_mean"])
        mafs = float(rows[-2][f"{dataset}_mse_mean"])
        assert proposed < mafs

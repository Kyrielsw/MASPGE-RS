import json
from pathlib import Path


def test_generic_state_holdout_protocol_forbids_training_and_tuning() -> None:
    root = Path(__file__).resolve().parent.parent
    config = json.loads(
        (root / "configs/generic_state_control_holdout_v1.json").read_text(
            encoding="utf-8"
        )
    )
    protocol = config["protocol"]
    assert protocol["training_permitted"] is False
    assert protocol["checkpoint_selection_permitted"] is False
    assert protocol["hyperparameter_search_permitted"] is False
    assert protocol["post_holdout_tuning_permitted"] is False
    assert protocol["study_level_holdout_previously_observed"] is True
    assert protocol["eligible_as_strict_unseen_test"] is False


def test_generic_state_holdout_evaluator_has_no_training_path() -> None:
    root = Path(__file__).resolve().parent.parent
    source = (root / "scripts/evaluate_generic_state_control_holdout.py").read_text(
        encoding="utf-8"
    )
    assert "train_stage" not in source
    assert "optimizer" not in source.lower()
    assert '"training_performed": False' in source
    assert '"post_holdout_tuning_permitted": False' in source
    assert "study_level_holdout_previously_observed" in source

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.schemas import TrainingHistoryEntry, TrainingMetrics
from src.training_history import (
    TrainingHistoryPersistenceError,
    append_training_entry,
    load_training_history,
    save_training_history,
)


def make_entry(
    *,
    hour: int = 12,
    model_type: str = "logreg",
) -> TrainingHistoryEntry:
    return TrainingHistoryEntry(
        trained_at=datetime(2026, 9, 6, hour, 0, tzinfo=timezone.utc),
        model_type=model_type,  # type: ignore[arg-type]
        hyperparameters={
            "solver": "liblinear",
            "C": 0.5,
            "fit_intercept": True,
            "class_weight": None,
        },
        metrics=TrainingMetrics(accuracy=0.8, f1=0.5, roc_auc=0.72),
    )


def test_load_missing_training_history_returns_empty_list(tmp_path: Path) -> None:
    history_path = tmp_path / "missing.json"

    assert load_training_history(history_path) == []


def test_save_and_load_training_history_round_trip(tmp_path: Path) -> None:
    history_path = tmp_path / "nested" / "training_history.json"
    entries = [make_entry(hour=12), make_entry(hour=13, model_type="random_forest")]

    save_training_history(entries, history_path)
    restored = load_training_history(history_path)

    assert history_path.is_file()
    assert restored == entries
    assert [entry.trained_at.hour for entry in restored] == [12, 13]


def test_save_training_history_preserves_unicode(tmp_path: Path) -> None:
    history_path = tmp_path / "training_history.json"
    entry = make_entry()
    entry.hyperparameters["description"] = "модель оттока"

    save_training_history([entry], history_path)

    assert "модель оттока" in history_path.read_text(encoding="utf-8")
    assert load_training_history(history_path) == [entry]


def test_append_training_entry_preserves_existing_order(tmp_path: Path) -> None:
    history_path = tmp_path / "training_history.json"
    first = make_entry(hour=12)
    second = make_entry(hour=13, model_type="random_forest")

    append_training_entry(first, history_path)
    append_training_entry(second, history_path)

    assert load_training_history(history_path) == [first, second]


def test_load_training_history_rejects_corrupted_json(tmp_path: Path) -> None:
    history_path = tmp_path / "training_history.json"
    history_path.write_text("not valid JSON", encoding="utf-8")

    with pytest.raises(TrainingHistoryPersistenceError, match="Could not load"):
        load_training_history(history_path)


def test_load_training_history_rejects_non_array_root(tmp_path: Path) -> None:
    history_path = tmp_path / "training_history.json"
    history_path.write_text('{"history": []}', encoding="utf-8")

    with pytest.raises(TrainingHistoryPersistenceError, match="JSON array"):
        load_training_history(history_path)


def test_load_training_history_rejects_invalid_entry(tmp_path: Path) -> None:
    history_path = tmp_path / "training_history.json"
    invalid_entry = make_entry().model_dump(mode="json")
    invalid_entry["model_type"] = "svm"
    history_path.write_text(json.dumps([invalid_entry]), encoding="utf-8")

    with pytest.raises(TrainingHistoryPersistenceError, match="invalid entry"):
        load_training_history(history_path)


def test_save_training_history_keeps_previous_file_on_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "training_history.json"
    original = make_entry(hour=12)
    replacement = make_entry(hour=13, model_type="random_forest")
    save_training_history([original], history_path)
    original_contents = history_path.read_text(encoding="utf-8")

    def fail_to_replace(_self: Path, _target: Path) -> Path:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_to_replace)

    with pytest.raises(TrainingHistoryPersistenceError, match="Could not save"):
        save_training_history([replacement], history_path)

    assert history_path.read_text(encoding="utf-8") == original_contents
    assert not history_path.with_name(f".{history_path.name}.tmp").exists()


def test_save_training_history_rejects_wrong_entry_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="TrainingHistoryEntry"):
        save_training_history([object()], tmp_path / "training_history.json")  # type: ignore[list-item]


def test_append_training_entry_rejects_wrong_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="TrainingHistoryEntry"):
        append_training_entry(object(), tmp_path / "training_history.json")  # type: ignore[arg-type]


@pytest.mark.parametrize("operation", [load_training_history, save_training_history])
def test_training_history_rejects_non_json_path(
    operation: object,
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "training_history.txt"

    with pytest.raises(ValueError, match=r"\.json suffix"):
        if operation is load_training_history:
            load_training_history(history_path)
        else:
            save_training_history([], history_path)

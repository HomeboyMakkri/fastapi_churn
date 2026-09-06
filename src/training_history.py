"""Persist and restore churn-model training history as JSON."""

from collections.abc import Sequence
import json
from pathlib import Path

from pydantic import ValidationError

from .schemas import TrainingHistoryEntry


class TrainingHistoryPersistenceError(RuntimeError):
    """Raised when training history cannot be serialized or restored."""


def load_training_history(path: Path) -> list[TrainingHistoryEntry]:
    """Load and validate training history, or return an empty missing history."""
    _validate_history_suffix(path)
    if not path.exists():
        return []
    if not path.is_file():
        raise IsADirectoryError(f"Training history path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as history_file:
            payload = json.load(history_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingHistoryPersistenceError(
            f"Could not load training history from {path}"
        ) from error

    if not isinstance(payload, list):
        raise TrainingHistoryPersistenceError(
            f"Training history must contain a JSON array: {path}"
        )

    try:
        return [TrainingHistoryEntry.model_validate(item) for item in payload]
    except ValidationError as error:
        raise TrainingHistoryPersistenceError(
            f"Training history contains an invalid entry: {path}"
        ) from error


def save_training_history(
    entries: Sequence[TrainingHistoryEntry],
    path: Path,
) -> None:
    """Atomically save validated training-history entries to a JSON file."""
    _validate_history_suffix(path)
    if any(not isinstance(entry, TrainingHistoryEntry) for entry in entries):
        raise TypeError("entries must contain only TrainingHistoryEntry objects")

    temporary_path = path.with_name(f".{path.name}.tmp")
    payload = [entry.model_dump(mode="json") for entry in entries]

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("w", encoding="utf-8") as history_file:
            json.dump(payload, history_file, ensure_ascii=False, indent=2)
            history_file.write("\n")
        temporary_path.replace(path)
    except (OSError, TypeError, ValueError) as error:
        temporary_path.unlink(missing_ok=True)
        raise TrainingHistoryPersistenceError(
            f"Could not save training history to {path}"
        ) from error


def append_training_entry(entry: TrainingHistoryEntry, path: Path) -> None:
    """Append one entry while preserving the existing history."""
    if not isinstance(entry, TrainingHistoryEntry):
        raise TypeError("entry must be a TrainingHistoryEntry")

    entries = load_training_history(path)
    entries.append(entry)
    save_training_history(entries, path)


def _validate_history_suffix(path: Path) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError(f"Training history file must use the .json suffix: {path}")

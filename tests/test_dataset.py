from pathlib import Path
from typing import Callable

import pytest

from src.dataset import ChurnDataset
from src.schemas import DatasetRowChurn


CsvFactory = Callable[[list[dict[str, object]]], Path]


def test_loads_csv_and_converts_rows(
    csv_factory: CsvFactory,
    valid_record: dict[str, object],
) -> None:
    path = csv_factory([valid_record])
    uppercase_extension_path = path.with_name("churn.CSV")
    path.rename(uppercase_extension_path)
    dataset = ChurnDataset(str(uppercase_extension_path))

    dataframe = dataset.load()
    rows = dataset.to_rows()

    assert dataframe.shape == (1, 10)
    assert rows == [DatasetRowChurn.model_validate(valid_record)]


def test_dataframe_property_returns_a_copy(
    csv_factory: CsvFactory,
    valid_record: dict[str, object],
) -> None:
    dataset = ChurnDataset(csv_factory([valid_record]))
    dataset.load()

    external_dataframe = dataset.dataframe
    external_dataframe.loc[0, "churn"] = 1

    assert dataset.dataframe.loc[0, "churn"] == 0
    assert dataset.to_rows()[0].churn == 0


def test_access_before_load_is_rejected(tmp_path: Path) -> None:
    dataset = ChurnDataset(tmp_path / "dataset.csv")

    with pytest.raises(RuntimeError, match="Dataset is not loaded"):
        _ = dataset.dataframe

    with pytest.raises(RuntimeError, match="Dataset is not loaded"):
        dataset.to_rows()


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    dataset = ChurnDataset(tmp_path / "missing.csv")

    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        dataset.load()


def test_directory_path_is_rejected(tmp_path: Path) -> None:
    dataset = ChurnDataset(tmp_path)

    with pytest.raises(IsADirectoryError, match="Dataset path is not a file"):
        dataset.load()


def test_non_csv_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dataset.txt"
    path.write_text("churn\n0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset file is not a CSV"):
        ChurnDataset(path).load()


def test_completely_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    path.touch()

    with pytest.raises(ValueError, match="Dataset file is empty"):
        ChurnDataset(path).load()


def test_csv_without_data_rows_is_rejected(
    csv_factory: CsvFactory,
    valid_record: dict[str, object],
) -> None:
    path = csv_factory([])
    path.write_text(",".join(valid_record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset contains no data rows"):
        ChurnDataset(path).load()


@pytest.mark.parametrize("column_problem", ["missing", "unexpected"])
def test_invalid_columns_are_reported(
    csv_factory: CsvFactory,
    valid_record: dict[str, object],
    column_problem: str,
) -> None:
    if column_problem == "missing":
        valid_record.pop("region")
    else:
        valid_record["customer_id"] = 123

    path = csv_factory([valid_record])

    with pytest.raises(ValueError, match=column_problem):
        ChurnDataset(path).load()


def test_invalid_row_reports_csv_line(
    csv_factory: CsvFactory,
    valid_record: dict[str, object],
) -> None:
    second_record = valid_record.copy()
    second_record["churn"] = 2
    path = csv_factory([valid_record, second_record])

    with pytest.raises(ValueError, match="CSV line 3"):
        ChurnDataset(path).load()
